#!/usr/bin/env python3
"""Module for finding nash equilibria"""
import sys

if not hasattr(sys, 'real_prefix'):
    sys.stderr.write('Could not detect virtualenv. Make sure that you\'ve '
                     'activated the virtual env\n(`. bin/activate`).\n')
    sys.exit(1)

import argparse
import json

from gameanalysis import rsgame, nash


_PARSER = argparse.ArgumentParser(description='''Compute nash equilibria in a
game.''')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='eq-file', default=sys.stdout,
                     type=argparse.FileType('w'), help='''Output equilibria
                     file. This file will contain a json list of mixed
                     profiles. (default: stdout)''')
_PARSER.add_argument('--regret', '-r', metavar='thresh', type=float,
                     default=1e-3, help='''Max allowed regret for approximate
                     Nash equilibria; default=1e-3''')
_PARSER.add_argument('--distance', '-d', metavar='distance', type=float,
                     default=1e-3, help='''L2-distance threshold to consider
                     equilibria distinct; default=1e-3''')
_PARSER.add_argument('--convergence', '-c', metavar='convergence', type=float,
                     default=1e-8, help='''Replicator dynamics convergence
                     thrshold; default=1e-8''')
_PARSER.add_argument('--max-iterations', '-m', metavar='iterations', type=int,
                     default=10000, help='''Max replicator dynamics iterations;
                     default=10000''')
_PARSER.add_argument('--support', '-s', metavar='support', type=float,
                     default=1e-3, help='''Min probability for a strategy to be
                     considered in support. default=1e-3''')
_PARSER.add_argument('--type', '-t', choices=('mixed', 'pure', 'mrp'),
                     default='mixed', help='''Type of approximate equilibrium to
                     compute: role-symmetric mixed-strategy Nash, pure-strategy
                     Nash, or min-regret profile; default=mixed''')
_PARSER.add_argument('--random-points', '-p', metavar='points', type=int,
                     default=0, help='''Number of random points from which to
                     initialize replicator dynamics in addition to the default
                     set of uniform and heavily-biased mixtures; default=0''')
_PARSER.add_argument('--one', '-n', action='store_true', help='''Always report
at least one equilibrium per game. This will return the minimum regret
equilibrium found, regardless of whether it was below the regret threshold''')


def main():
    args = _PARSER.parse_args()
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
        raise ValueError('Unknown command given: {}'.format(args.type))

    json.dump(equilibria, args.output, default=lambda x: x.to_json())
    args.output.write('\n')


if __name__ == '__main__':
    main()
else:
    raise ImportError('This module should not be imported')
