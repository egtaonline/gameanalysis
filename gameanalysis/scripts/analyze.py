"""Analyze a game"""
import argparse
import json
import sys

import numpy as np
from numpy import linalg

from gameanalysis import dominance
from gameanalysis import gameio
from gameanalysis import nash
from gameanalysis import reduction
from gameanalysis import regret
from gameanalysis import subgame


def add_parser(subparsers):
    parser = subparsers.add_parser('analyze', help="""Analyze games""",
                                   description="""Perform game analysis.""")
    parser.add_argument('--input', '-i', metavar='<input-file>',
                        default=sys.stdin, type=argparse.FileType('r'),
                        help="""Input file for script.  (default: stdin)""")
    parser.add_argument('--output', '-o', metavar='<output-file>',
                        default=sys.stdout, type=argparse.FileType('w'),
                        help="""Output file for script. (default: stdout)""")
    parser.add_argument('--dist-thresh', metavar='<distance-threshold>',
                        type=float, default=1e-3, help="""L2 norm threshold,
                        inside of which, equilibria are considered identical.
                        (default: %(default)g)""")
    parser.add_argument('--regret-thresh', '-r', metavar='<regret-threshold>',
                        type=float, default=1e-3, help="""Maximum regret to
                        consider an equilibrium confirmed. (default:
                        %(default)g)""")
    parser.add_argument('--supp-thresh', '-t', metavar='<support-threshold>',
                        type=float, default=1e-3, help="""Maximum probability
                        to consider a strategy in support. (default:
                        %(default)g)""")
    parser.add_argument('--rand-restarts', metavar='<random-restarts>',
                        type=int, default=0, help="""The number of random
                        points to add to nash equilibrium finding. (default:
                        %(default)d)""")
    parser.add_argument('--max-iters', '-m', metavar='<maximum-iterations>',
                        type=int, default=10000, help="""The maximum number of
                        iterations to run through replicator dynamics.
                        (default: %(default)d)""")
    parser.add_argument('--converge-thresh', '-c',
                        metavar='<convergence-threshold>', type=float,
                        default=1e-8, help="""The convergence threshold for
                        replicator dynamics. (default: %(default)g)""")
    parser.add_argument('--processes', '-p', metavar='<num-procs>', type=int,
                        help="""Number of processes to use to run nash finding.
                        (default: number of cores)""")
    parser.add_argument('--dpr', nargs='+', metavar='<role> <count>',
                        help="""Apply a DPR reduction to the game, with reduced
                        counts per role specified.""")
    parser.add_argument('--dominance', '-d', action='store_true',
                        help="""Remove dominated strategies.""")
    parser.add_argument('--subgames', '-s', action='store_true',
                        help="""Extract maximal subgames, and analyze each
                        individually instead of considering the game as a
                        whole.""")
    return parser


def main(args):
    game, serial = gameio.read_game(json.load(args.input))

    if args.dpr:
        red_players = serial.from_role_json(dict(zip(
            args.dpr[::2], map(int, args.dpr[1::2]))))
        red = reduction.DeviationPreserving(game.num_strategies,
                                            game.num_players, red_players)
        redgame = red.reduce_game(game, True)
    else:
        redgame = game
    redserial = serial

    if args.dominance:
        domsub = dominance.iterated_elimination(redgame, 'strictdom')
        redgame = subgame.subgame(redgame, domsub)
        redserial = subgame.subserializer(redserial, domsub)

    if args.subgames:
        subgames = subgame.maximal_subgames(redgame)
    else:
        subgames = np.ones(redgame.num_role_strats, bool)[None]

    methods = {
        'replicator': {
            'max_iters': args.max_iters,
            'converge_thresh': args.converge_thresh},
        'optimize': {}}
    noeq_subgames = []
    candidates = []
    for submask in subgames:
        subg = subgame.subgame(redgame, submask)
        subeqa = nash.mixed_nash(
            subg, regret_thresh=args.regret_thresh,
            dist_thresh=args.dist_thresh, processes=args.processes, **methods)
        eqa = subgame.translate(subg.trim_mixture_support(
            subeqa, supp_thresh=args.supp_thresh), submask)
        if eqa.size:
            for eqm in eqa:
                if not any(linalg.norm(eqm - eq) < args.dist_thresh
                           for eq in candidates):
                    candidates.append(eqm)
        else:
            noeq_subgames.append(submask)

    equilibria = []
    unconfirmed = []
    unexplored = []
    for eqm in candidates:
        support = eqm > 0
        gains = regret.mixture_deviation_gains(redgame, eqm)
        role_gains = redgame.role_reduce(gains, ufunc=np.fmax)
        gain = np.nanmax(role_gains)

        if np.isnan(gains).any() and gain <= args.regret_thresh:
            # Not fully explored but might be good
            unconfirmed.append((eqm, gain))

        elif np.any(role_gains > args.regret_thresh):
            # There are deviations, did we explore them?
            dev_inds = ([np.argmax(gs == mg) for gs, mg
                         in zip(redgame.role_split(gains), role_gains)] +
                        redgame.role_starts)[role_gains > args.regret_thresh]
            for dind in dev_inds:
                devsupp = support.copy()
                devsupp[dind] = True
                if not np.all(devsupp <= subgames, -1).any():
                    unexplored.append((devsupp, dind, gains[dind], eqm))

        else:
            # Equilibrium!
            equilibria.append((eqm, np.max(gains)))

    # Output Game
    args.output.write('Game Analysis\n')
    args.output.write('=============\n')
    args.output.write(serial.to_game_printstring(game))
    args.output.write('\n\n')
    if args.dpr is not None:
        args.output.write('With DPR reduction: ')
        args.output.write(' '.join(args.dpr))
        args.output.write('\n\n')
    if args.dominance:
        num = np.sum(~domsub)
        if num:
            args.output.write('Found {:d} dominated strateg{}\n'.format(
                num, 'y' if num == 1 else 'ies'))
            args.output.write(serial.to_prof_printstring(~domsub))
            args.output.write('\n')
        else:
            args.output.write('Found no dominated strategies\n\n')
    if args.subgames:
        num = subgames.shape[0]
        if num:
            args.output.write(
                'Found {:d} maximal complete subgame{}\n\n'.format(
                    num, '' if num == 1 else 's'))
        else:
            args.output.write('Found no complete subgames\n\n')
    args.output.write('\n')

    # Output social welfare
    args.output.write('Social Welfare\n')
    args.output.write('--------------\n')
    welfare, profile = regret.max_pure_social_welfare(game)
    if profile is None:
        args.output.write('There was no profile with complete payoff data\n\n')
    else:
        args.output.write('\nMaximum social welfare profile:\n')
        args.output.write(serial.to_prof_printstring(profile))
        args.output.write('Welfare: {:.4f}\n\n'.format(welfare))

        if game.num_roles > 1:
            for role, welfare, profile in zip(
                    serial.role_names,
                    *regret.max_pure_social_welfare(game, True)):
                args.output.write('Maximum "{}" welfare profile:\n'.format(
                    role))
                args.output.write(serial.to_prof_printstring(profile))
                args.output.write('Welfare: {:.4f}\n\n'.format(welfare))

    args.output.write('\n')

    # Output Equilibria
    args.output.write('Equilibria\n')
    args.output.write('----------\n')
    if equilibria:
        args.output.write('Found {:d} equilibri{}\n\n'.format(
            len(equilibria), 'um' if len(equilibria) == 1 else 'a'))
        for i, (eqm, reg) in enumerate(equilibria, 1):
            args.output.write('Equilibrium {:d}:\n'.format(i))
            args.output.write(redserial.to_prof_printstring(eqm))
            args.output.write('Regret: {:.4f}\n\n'.format(reg))
    else:
        args.output.write('Found no equilibria\n\n')
    args.output.write('\n')

    # Output No-equilibria Subgames
    args.output.write('No-equilibria Subgames\n')
    args.output.write('----------------------\n')
    if noeq_subgames:
        args.output.write('Found {:d} no-equilibria subgame{}\n\n'.format(
            len(noeq_subgames), '' if len(noeq_subgames) == 1 else 's'))
        noeq_subgames.sort(key=lambda x: x.sum())
        for i, subg in enumerate(noeq_subgames, 1):
            args.output.write('No-equilibria subgame {:d}:\n'.format(i))
            args.output.write(redserial.to_prof_printstring(subg))
            args.output.write('\n')
    else:
        args.output.write('Found no no-equilibria subgames\n\n')
    args.output.write('\n')

    # Output Unconfirmed Candidates
    args.output.write('Unconfirmed Candidate Equilibria\n')
    args.output.write('--------------------------------\n')
    if unconfirmed:
        args.output.write('Found {:d} unconfirmed candidate{}\n\n'.format(
            len(unconfirmed), '' if len(unconfirmed) == 1 else 's'))
        unconfirmed.sort(key=lambda x: ((x[0] > 0).sum(), x[1]))
        for i, (eqm, reg_bound) in enumerate(unconfirmed, 1):
            args.output.write('Unconfirmed candidate {:d}:\n'.format(i))
            args.output.write(redserial.to_prof_printstring(eqm))
            args.output.write('Regret at least: {:.4f}\n\n'.format(reg_bound))
    else:
        args.output.write('Found no unconfirmed candidate equilibria\n\n')
    args.output.write('\n')

    # Output Unexplored Subgames
    args.output.write('Unexplored Best-response Subgames\n')
    args.output.write('---------------------------------\n')
    if unexplored:
        min_supp = min(supp.sum() for supp, _, _, _ in unexplored)
        args.output.write(
            'Found {:d} unexplored best-response subgame{}\n'.format(
                len(unexplored), '' if len(unexplored) == 1 else 's'))
        args.output.write(
            'Smallest unexplored subgame has support {:d}\n\n'.format(
                min_supp))

        unexplored.sort(key=lambda x: (x[0].sum(), -x[2]))
        for i, (sub, dev, gain, eqm) in enumerate(unexplored, 1):
            args.output.write('Unexplored subgame {:d}:\n'.format(i))
            args.output.write(redserial.to_prof_printstring(sub))
            args.output.write('{:.4f} for deviating to {} from:\n'.format(
                gain, redserial.strat_name(dev)))
            args.output.write(redserial.to_prof_printstring(eqm))
            args.output.write('\n')
    else:
        args.output.write('Found no unexplored best-response subgames\n\n')
    args.output.write('\n')

    # Output json data
    args.output.write('Json Data\n')
    args.output.write('=========\n')
    json_data = {
        'equilibria': [redserial.to_prof_json(eqm) for eqm, _ in equilibria]}
    json.dump(json_data, args.output)
    args.output.write('\n')
