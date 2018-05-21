"""Analyze a game"""
import argparse
import json
import sys

import numpy as np

from gameanalysis import collect
from gameanalysis import dominance
from gameanalysis import gamereader
from gameanalysis import nash
from gameanalysis import reduction
from gameanalysis import regret
from gameanalysis import restrict


def add_parser(subparsers):
    """Create analysis parser"""
    parser = subparsers.add_parser(
        'analyze', help="""Analyze games""", description="""Perform game
        analysis.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        '--dist-thresh', metavar='<distance-threshold>', type=float,
        default=0.1, help="""L2 norm threshold, inside of which, equilibria
        are considered identical.  (default: %(default)g)""")
    parser.add_argument(
        '--regret-thresh', '-r', metavar='<regret-threshold>', type=float,
        default=1e-3, help="""Maximum regret to consider an equilibrium
        confirmed. (default: %(default)g)""")
    parser.add_argument(
        '--supp-thresh', '-t', metavar='<support-threshold>', type=float,
        default=1e-3, help="""Maximum probability to consider a strategy in
        support. (default: %(default)g)""")
    parser.add_argument(
        '--rand-restarts', metavar='<random-restarts>', type=int, default=0,
        help="""The number of random points to add to nash equilibrium finding.
        (default: %(default)d)""")
    parser.add_argument(
        '--max-iters', '-m', metavar='<maximum-iterations>', type=int,
        default=10000, help="""The maximum number of iterations to run through
        replicator dynamics.  (default: %(default)d)""")
    parser.add_argument(
        '--converge-thresh', '-c', metavar='<convergence-threshold>',
        type=float, default=1e-8, help="""The convergence threshold for
        replicator dynamics. (default: %(default)g)""")
    parser.add_argument(
        '--processes', '-p', metavar='<num-procs>', type=int, help="""Number of
        processes to use to run nash finding.  (default: number of cores)""")
    parser.add_argument(
        '--one', action='store_true', help="""If specified, run a potentially
        expensive algorithm to guarantee an approximate equilibrium, if none
        are found via other methods.""")
    parser.add_argument(
        '--dominance', '-d', action='store_true', help="""Remove dominated
        strategies.""")
    parser.add_argument(
        '--restrictions', '-s', action='store_true', help="""Extract maximal
        restricted games, and analyze each individually instead of considering
        the game as a whole.""")
    reductions = parser.add_mutually_exclusive_group()
    reductions.add_argument(
        '--dpr', metavar='<role:count;role:count,...>', help="""Specify a
        deviation preserving reduction.""")
    reductions.add_argument(
        '--hr', metavar='<role:count;role:count,...>', help="""Specify a
        hierarchical reduction.""")
    return parser


def main(args): # pylint: disable=too-many-statements,too-many-branches,too-many-locals
    """Entry point for analysis"""
    game = gamereader.load(args.input)

    if args.dpr is not None:
        red_players = game.role_from_repr(args.dpr, dtype=int)
        game = reduction.deviation_preserving.reduce_game(game, red_players)
    elif args.hr is not None:
        red_players = game.role_from_repr(args.hr, dtype=int)
        game = reduction.hierarchical.reduce_game(game, red_players)

    if args.dominance:
        domsub = dominance.iterated_elimination(game, 'strictdom')
        game = game.restrict(domsub)

    if args.restrictions:
        restrictions = restrict.maximal_restrictions(game)
    else:
        restrictions = np.ones((1, game.num_strats), bool)

    methods = {
        'replicator': {
            'max_iters': args.max_iters,
            'converge_thresh': args.converge_thresh},
        'optimize': {}}
    noeq_restrictions = []
    candidates = []
    for rest in restrictions:
        rgame = game.restrict(rest)
        reqa = nash.mixed_nash(
            rgame, regret_thresh=args.regret_thresh,
            dist_thresh=args.dist_thresh, processes=args.processes,
            at_least_one=args.one, **methods)
        eqa = restrict.translate(rgame.trim_mixture_support(
            reqa, thresh=args.supp_thresh), rest)
        if eqa.size:
            candidates.extend(eqa)
        else:
            noeq_restrictions.append(rest)

    equilibria = collect.mcces(args.dist_thresh)
    unconfirmed = collect.mcces(args.dist_thresh)
    unexplored = {}
    for eqm in candidates:
        support = eqm > 0
        gains = regret.mixture_deviation_gains(game, eqm)
        role_gains = np.fmax.reduceat(gains, game.role_starts)
        gain = np.nanmax(role_gains)

        if np.isnan(gains).any() and gain <= args.regret_thresh:
            # Not fully explored but might be good
            unconfirmed.add(eqm, gain)

        elif np.any(role_gains > args.regret_thresh):
            # There are deviations, did we explore them?
            dev_inds = ([np.argmax(gs == mg) for gs, mg
                         in zip(np.split(gains, game.role_starts[1:]),
                                role_gains)] +
                        game.role_starts)[role_gains > args.regret_thresh]
            for dind in dev_inds:
                devsupp = support.copy()
                devsupp[dind] = True
                if not np.all(devsupp <= restrictions, -1).any():
                    ind = restrict.to_id(game, devsupp)
                    new_info = (gains[dind], dind, eqm)
                    old_info = unexplored.get(ind, (0, 0, None))
                    unexplored[ind] = max(new_info, old_info)

        else:
            # Equilibrium!
            equilibria.add(eqm, np.max(gains))

    # Output Game
    args.output.write('Game Analysis\n')
    args.output.write('=============\n')
    args.output.write(str(game))
    args.output.write('\n\n')
    if args.dpr is not None:
        args.output.write('With deviation preserving reduction: ')
        args.output.write(args.dpr.replace(';', ' '))
        args.output.write('\n\n')
    elif args.hr is not None:
        args.output.write('With hierarchical reduction: ')
        args.output.write(args.hr.replace(';', ' '))
        args.output.write('\n\n')
    if args.dominance:
        num = np.sum(~domsub)
        if num:
            args.output.write('Found {:d} dominated strateg{}\n'.format(
                num, 'y' if num == 1 else 'ies'))
            args.output.write(game.restriction_to_str(~domsub))
            args.output.write('\n\n')
        else:
            args.output.write('Found no dominated strategies\n\n')
    if args.restrictions:
        num = restrictions.shape[0]
        if num:
            args.output.write(
                'Found {:d} maximal complete restricted game{}\n\n'.format(
                    num, '' if num == 1 else 's'))
        else:
            args.output.write('Found no complete restricted games\n\n')
    args.output.write('\n')

    # Output social welfare
    args.output.write('Social Welfare\n')
    args.output.write('--------------\n')
    welfare, profile = regret.max_pure_social_welfare(game)
    if profile is None:
        args.output.write('There was no profile with complete payoff data\n\n')
    else:
        args.output.write('\nMaximum social welfare profile:\n')
        args.output.write(game.profile_to_str(profile))
        args.output.write('\nWelfare: {:.4f}\n\n'.format(welfare))

        if game.num_roles > 1:
            for role, welfare, profile in zip(
                    game.role_names,
                    *regret.max_pure_social_welfare(game, by_role=True)):
                args.output.write('Maximum "{}" welfare profile:\n'.format(
                    role))
                args.output.write(game.profile_to_str(profile))
                args.output.write('\nWelfare: {:.4f}\n\n'.format(welfare))

    args.output.write('\n')

    # Output Equilibria
    args.output.write('Equilibria\n')
    args.output.write('----------\n')
    if equilibria:
        args.output.write('Found {:d} equilibri{}\n\n'.format(
            len(equilibria), 'um' if len(equilibria) == 1 else 'a'))
        for i, (eqm, reg) in enumerate(equilibria, 1):
            args.output.write('Equilibrium {:d}:\n'.format(i))
            args.output.write(game.mixture_to_str(eqm))
            args.output.write('\nRegret: {:.4f}\n\n'.format(reg))
    else:
        args.output.write('Found no equilibria\n\n')
    args.output.write('\n')

    # Output No-equilibria Subgames
    args.output.write('No-equilibria Subgames\n')
    args.output.write('----------------------\n')
    if noeq_restrictions:
        args.output.write(
            'Found {:d} no-equilibria restricted game{}\n\n'.format(
                len(noeq_restrictions),
                '' if len(noeq_restrictions) == 1 else 's'))
        noeq_restrictions.sort(key=lambda x: x.sum())
        for i, subg in enumerate(noeq_restrictions, 1):
            args.output.write(
                'No-equilibria restricted game {:d}:\n'.format(i))
            args.output.write(game.restriction_to_str(subg))
            args.output.write('\n\n')
    else:
        args.output.write('Found no no-equilibria restricted games\n\n')
    args.output.write('\n')

    # Output Unconfirmed Candidates
    args.output.write('Unconfirmed Candidate Equilibria\n')
    args.output.write('--------------------------------\n')
    if unconfirmed:
        args.output.write('Found {:d} unconfirmed candidate{}\n\n'.format(
            len(unconfirmed), '' if len(unconfirmed) == 1 else 's'))
        ordered = sorted(
            (sum(e > 0 for e in m), r, m) for m, r in unconfirmed)
        for i, (_, reg_bound, eqm) in enumerate(ordered, 1):
            args.output.write('Unconfirmed candidate {:d}:\n'.format(i))
            args.output.write(game.mixture_to_str(eqm))
            args.output.write('\nRegret at least: {:.4f}\n\n'.format(
                reg_bound))
    else:
        args.output.write('Found no unconfirmed candidate equilibria\n\n')
    args.output.write('\n')

    # Output Unexplored Subgames
    args.output.write('Unexplored Best-response Subgames\n')
    args.output.write('---------------------------------\n')
    if unexplored:
        min_supp = min(restrict.from_id(game, sid).sum() for sid in unexplored)
        args.output.write(
            'Found {:d} unexplored best-response restricted game{}\n'.format(
                len(unexplored), '' if len(unexplored) == 1 else 's'))
        args.output.write(
            'Smallest unexplored restricted game has support {:d}\n\n'.format(
                min_supp))

        ordered = sorted((
            restrict.from_id(game, sind).sum(),
            -gain, dev,
            restrict.from_id(game, sind),
            eqm,
        ) for sind, (gain, dev, eqm) in unexplored.items())
        for i, (_, ngain, dev, sub, eqm) in enumerate(ordered, 1):
            args.output.write('Unexplored restricted game {:d}:\n'.format(i))
            args.output.write(game.restriction_to_str(sub))
            args.output.write('\n{:.4f} for deviating to {} from:\n'.format(
                -ngain, game.strat_name(dev)))
            args.output.write(game.mixture_to_str(eqm))
            args.output.write('\n\n')
    else:
        args.output.write(
            'Found no unexplored best-response restricted games\n\n')
    args.output.write('\n')

    # Output json data
    args.output.write('Json Data\n')
    args.output.write('=========\n')
    json_data = {
        'equilibria': [game.mixture_to_json(eqm) for eqm, _ in equilibria]}
    json.dump(json_data, args.output)
    args.output.write('\n')
