"""find nash equilibria"""
import argparse
import json
import sys

from gameanalysis import gamereader
from gameanalysis import nash


def add_parser(subparsers):
    """Add nash parser"""
    parser = subparsers.add_parser(
        'nash', help="""Compute nash equilibria""", description="""Computes
        Nash equilibria from the input file and creates a json file of the
        results.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        '--regret', '-r', metavar='<thresh>', type=float, default=1e-3,
        help="""Max allowed regret for approximate Nash equilibria. (default:
        %(default)g)""")
    parser.add_argument(
        '--distance', '-d', metavar='<distance>', type=float, default=0.1,
        help="""Average normalized per-role L2-norm threshold to consider
        equilibria distinct. Valid in [0, 1]. (default: %(default)g)""")
    parser.add_argument(
        '--support', '-s', metavar='<support>', type=float, default=1e-3,
        help="""Min probability for a strategy to be considered in support.
        (default: %(default)g)""")
    parser.add_argument(
        '--processes', '-p', type=int, metavar='<num-processes>', default=None,
        help="""The number of processes to use when finding a mixed nash.
        (default: num-cores)""")
    parser.add_argument(
        '--style', default='best',
        choices=['fast', 'fast*', 'more', 'more*', 'best', 'best*', 'one'],
        help="""The `style` of mixed equilibrium finding. `fast` runs the
        fastest algorithms that should find an equilibrium. `more` will try
        slower ones until it finds one. `best` is more but will do an
        exhaustive search with a timeout of a half hour. `one` is the same as
        best with no timeout. The starred* versions do the same, but will
        return the minimum regret mixture if no equilibria were found.
        (default: %(default)s)""")

    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        '--pure', action='store_true', help="""Compute pure equilibria.""")
    type_group.add_argument(
        '--mixed', action='store_false', dest='pure', help="""Compute mixed
        equilibria. (default)""")
    return parser


def main(args):
    """Entry point for nash finding"""
    game = gamereader.load(args.input)

    if args.pure:
        equilibria = nash.pure_equilibria(game, epsilon=args.regret)
        if args.style.endswith('*') and not equilibria:
            equilibria = nash.min_regret_profile(game)[None]
        json.dump([game.profile_to_json(eqm) for eqm in equilibria],
                  args.output)

    else:
        equilibria = nash.mixed_equilibria(
            game, style=args.style, regret_thresh=args.regret,
            dist_thresh=args.distance, processes=args.processes)
        json.dump(
            [game.mixture_to_json(eqm) for eqm
             in game.trim_mixture_support(equilibria, thresh=args.support)],
            args.output)

    args.output.write('\n')
