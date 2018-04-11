"""Analyze a game using gp learn"""
import argparse
import json
import sys
import warnings

from gameanalysis import learning
from gameanalysis import gamereader
from gameanalysis import nash
from gameanalysis import regret


def add_parser(subparsers):
    """Parser for learning script"""
    parser = subparsers.add_parser(
        'learning', help="""Analyze game using learning""",
        description="""Perform game analysis with learned model""")
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
        default=1e-3, help="""L2 norm threshold, inside of which, equilibria
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
    return parser


def main(args):
    """Entry point for learning script"""
    with warnings.catch_warnings(record=True) as warns:
        game = learning.rbfgame_train(gamereader.load(args.input))
    methods = {'replicator': {'max_iters': args.max_iters,
                              'converge_thresh': args.converge_thresh},
               'optimize': {}}

    mixed_equilibria = game.trim_mixture_support(
        nash.mixed_nash(game, regret_thresh=args.regret_thresh,
                        dist_thresh=args.dist_thresh, processes=args.processes,
                        at_least_one=args.one, **methods),
        thresh=args.supp_thresh)

    equilibria = [(eqm, regret.mixture_regret(game, eqm))
                  for eqm in mixed_equilibria]

    # Output game
    args.output.write('Game Learning\n')
    args.output.write('=============\n')
    args.output.write(str(game))
    args.output.write('\n\n')

    if any(w.category == UserWarning and
           w.message.args[0] == (
               'some lengths were at their bounds, this may indicate a poor '
               'fit') for w in warns):
        args.output.write('Warning\n')
        args.output.write('=======\n')
        args.output.write(
            'Some length scales were at their limit. This is a strong\n'
            'indication that a good representation was not found.\n')
        args.output.write('\n\n')

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

    # Output json data
    args.output.write('Json Data\n')
    args.output.write('=========\n')
    json_data = {
        'equilibria': [game.mixture_to_json(eqm) for eqm, _ in equilibria]}
    json.dump(json_data, args.output)
    args.output.write('\n')
