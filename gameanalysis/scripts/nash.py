"""Module for finding nash equilibria"""
import json

from gameanalysis import rsgame
from gameanalysis import nash


def update_parser(parser):
    parser.description = """Computes Nash equilibria from the input file and
creates a json file of the results."""
    parser.add_argument('--regret', '-r', metavar='<thresh>', type=float,
                        default=1e-3, help="""Max allowed regret for
                        approximate Nash equilibria; default=1e-3""")
    parser.add_argument('--distance', '-d', metavar='<distance>', type=float,
                        default=1e-3, help="""L2-distance threshold to consider
                        equilibria distinct; default=1e-3""")
    parser.add_argument('--convergence', '-c', metavar='<convergence>',
                        type=float, default=1e-8, help="""Replicator dynamics
                        convergence thrshold; default=1e-8""")
    parser.add_argument('--max-iterations', '-m', metavar='<iterations>',
                        type=int, default=10000, help="""Max replicator
                        dynamics iterations; default=10000""")
    parser.add_argument('--support', '-s', metavar='<support>', type=float,
                        default=1e-3, help="""Min probability for a strategy to
                        be considered in support. default=1e-3""")
    parser.add_argument('--type', '-t', choices=('mixed', 'pure', 'mrp'),
                        default='mixed', help="""Type of approximate
                        equilibrium to compute: role-symmetric mixed-strategy
                        Nash, pure-strategy Nash, or min-regret profile;
                        default=mixed""")
    parser.add_argument('--random-points', '-p', metavar='<num-points>',
                        type=int, default=0, help="""Number of random points
                        from which to initialize replicator dynamics in
                        addition to the default set of uniform and
                        heavily-biased mixtures; default=0""")
    parser.add_argument('--one', '-n', action='store_true', help="""Always
                        report at least one equilibrium per game. This will
                        return the minimum regret equilibrium found, regardless
                        of whether it was below the regret threshold""")


def main(args):
    game = rsgame.Game.from_json(json.load(args.input))

    if args.type == 'pure':
        equilibria = list(nash.pure_nash(game, args.regret))
        if args.one and not equilibria:
            equilibria = [nash.min_regret_profile(game)]
    elif args.type == 'mixed':
        equilibria = [eq.trim_support(args.support) for eq
                      in nash.mixed_nash(game, args.regret, args.distance,
                                         args.random_points, args.one,
                                         max_iters=args.max_iterations,
                                         converge_thresh=args.convergence)]
    elif args.type == 'mrp':
        equilibria = [nash.min_regret_profile(game)]
    else:
        raise ValueError('Unknown command given: {0}'.format(args.type))

    json.dump(equilibria, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
