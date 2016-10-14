"""find nash equilibria"""
import argparse
import json
import sys

from gameanalysis import gameio
from gameanalysis import nash


def add_parser(subparsers):
    parser = subparsers.add_parser('nash', help="""Compute nash equilibria""",
                                   description="""Computes Nash equilibria from
                                   the input file and creates a json file of
                                   the results.""")
    parser.add_argument('--input', '-i', metavar='<input-file>',
                        default=sys.stdin, type=argparse.FileType('r'),
                        help="""Input file for script.  (default: stdin)""")
    parser.add_argument('--output', '-o', metavar='<output-file>',
                        default=sys.stdout, type=argparse.FileType('w'),
                        help="""Output file for script. (default: stdout)""")
    parser.add_argument('--regret', '-r', metavar='<thresh>', type=float,
                        default=1e-3, help="""Max allowed regret for
                        approximate Nash equilibria; default=1e-3""")
    parser.add_argument('--distance', '-d', metavar='<distance>', type=float,
                        default=1e-3, help="""L2-distance threshold to consider
                        equilibria distinct; default=1e-3""")
    parser.add_argument('--convergence', '-c', metavar='<convergence>',
                        type=float, default=1e-8, help="""Replicator dynamics
                        convergence thrshold; default=1e-8""")
    parser.add_argument('--max-iterations', '-x', metavar='<iterations>',
                        type=int, default=10000, help="""Max replicator
                        dynamics iterations; default=10000""")
    parser.add_argument('--support', '-s', metavar='<support>', type=float,
                        default=1e-3, help="""Min probability for a strategy to
                        be considered in support. default=1e-3""")
    parser.add_argument('--type', '-t', metavar='<type>', default='mixed',
                        choices=('mixed', 'pure', 'min-reg-prof',
                                 'min-reg-grid', 'min-reg-rand', 'rand'),
                        help="""Type of equilibrium to compute: `mixed`
                            - role-symmetric mixed-strategy Nash. `pure` -
    pure-strategy Nash.  `min-reg-prof` - minimum regret profile.
    `min-reg-grid` - minimum regret mixture over a grid search with
    `grid-points` points along each dimension.  `min-reg-rand`
                        - minimum regret mixture over `random-mixtures` number
                        of random mixtures. `rand` - simply returns
                        `random-mixtures` number of random mixtures. (default
                        %(default)s)""")
    parser.add_argument('--random-mixtures', '-m', metavar='<num-mixtures>',
                        type=int, default=0, help="""Number of random mixtures
                        to use when finding the minimum regret random profile
                        or when initializing replicator dynamics.  (default:
                        %(default)d)""")
    parser.add_argument('--one', '-n', action='store_true', help="""Always
                        report at least one equilibrium per game. This will
                        return the minimum regret equilibrium found, regardless
                        of whether it was below the regret threshold""")
    parser.add_argument('--processes', '-p', type=int,
                        metavar='<num-processes>', default=None, help="""The
                        number of processes to use when finding a mixed nahs
                        using replicator dynamics.  (default: num-cores)""")
    parser.add_argument('--grid-points', '-g', metavar='<num-grid-points>',
                        type=int, default=2, help="""Number of grid points to
                        use per dimension on the grid search of mixed
                        strategies / Nash finding. 2 is the same as only
                        searching pure profiles.  (default: %(default)d)""")
    return parser


def main(args):
    game, serial = gameio.read_game(json.load(args.input))

    if args.type == 'pure':
        equilibria = nash.pure_nash(game, args.regret)
        if args.one and not equilibria:
            equilibria = nash.min_regret_profile(game)[None]

    elif args.type == 'mixed':
        rep_args = {
            'max_iters': args.max_iterations,
            'converge_thresh': args.convergence
        }
        equilibria = nash.mixed_nash(game, args.regret, args.distance,
                                     random_restarts=args.random_mixtures,
                                     grid_points=args.grid_points,
                                     at_least_one=args.one,
                                     processes=args.processes,
                                     replicator=rep_args, optimize={})
        equilibria = game.trim_mixture_support(equilibria, args.support)

    elif args.type == 'min-reg-prof':
        equilibria = nash.min_regret_profile(game)[None]

    elif args.type == 'min-reg-grid':
        equilibria = nash.min_regret_grid_mixture(
            game, args.grid_points)[None]
        equilibria = game.trim_mixture_support(equilibria, args.support)

    elif args.type == 'min-reg-rand':
        equilibria = nash.min_regret_rand_mixture(
            game, args.random_mixtures)[None]
        equilibria = game.trim_mixture_support(equilibria, args.support)

    elif args.type == 'rand':
        equilibria = game.random_mixtures(args.random_mixtures)
        equilibria = game.trim_mixture_support(equilibria, args.support)

    else:
        raise ValueError('Unknown command given: {0}'.format(args.type))  # pragma: no cover # noqa

    json.dump([serial.to_prof_json(eqm) for eqm in equilibria], args.output)
    args.output.write('\n')
