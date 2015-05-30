import sys
import itertools
import argparse
import json
import numpy as np
import numpy.linalg as linalg

from gameanalysis import regret, rsgame


_TINY = np.finfo(float).tiny


def pure_nash(game, epsilon=0):
    """Returns a generator of all pure-strategy epsilon-Nash equilibria."""
    return (profile for profile in game
            if regret.pure_strategy_regret(game, profile) <= epsilon)


def min_regret_profile(game):
    """Finds the profile with the confirmed lowest regret.

    """
    return min((regret.pure_strategy_regret(game, prof), i, prof)
               for i, prof in enumerate(game))[2]


def mixed_nash(game, regret_thresh=1e-3, dist_thresh=1e-3, random_restarts=0,
               at_least_one=False, as_array=False, *rd_args, **rd_kwargs):
    """Finds role-symmetric, mixed Nash equilibria using replicator dynamics

    Returns a generator of mixed profiles

    regret_thresh:   The threshold to consider an equilibrium found
    dist_thresh:     The threshold for considering equilibria distinct
    random_restarts: The number of random initializations for replicator
                     dynamics
    at_least_one:    Returns the minimum regret mixture found by replicator
                     dynamics if no equilibria were within the regret threshold
    as_array:        If true returns equilibria in array form.
    rd_*:            Extra arguments to pass through to replicator dynamics

    """
    wrap = (lambda x: x) if as_array else game.to_profile
    equilibria = []  # TODO More efficient way to check distinctness
    best = (np.inf, -1, None)  # Best convergence so far

    for i, mix in enumerate(itertools.chain(
            game.pure_mixtures(as_array=True),
            game.biased_mixtures(as_array=True),
            [game.uniform_mixture(as_array=True)],
            (game.random_mixture(as_array=True)
             for _ in range(random_restarts)))):
        eq = _replicator_dynamics(game, mix, *rd_args, **rd_kwargs)
        reg = regret.mixture_regret(game, eq)
        if (reg <= regret_thresh and all(linalg.norm(e - eq, 2) >= dist_thresh
                                         for e in equilibria)):
            equilibria.append(eq)
            yield wrap(eq)
        best = min(best, (reg, i, eq))
    if at_least_one and not equilibria:
        yield wrap(best[2])


def _replicator_dynamics(game, mix, max_iters=10000, converge_thresh=1e-8,
                         verbose=False):
    """Replicator dynamics

    This will run at most max_iters of replicators dynamics and return unless
    the difference between successive mixtures is less than converge_thresh.

    """
    for i in range(max_iters):
        old_mix = mix
        mix = (game.expected_values(mix) - game.min_payoffs[:, np.newaxis] +
               _TINY) * mix
        mix = mix / mix.sum(1)[:, np.newaxis]
        if linalg.norm(mix - old_mix) <= converge_thresh:
            break
        if verbose:
            sys.stderr.write('{:d}: mix = {}, regret = {:f}\n'.format(
                i + 1,
                mix,
                regret.mixture_regret(game, mix)))
    return np.maximum(mix, 0)  # Probabilities are occasionally negative


_PARSER = argparse.ArgumentParser(add_help=False, description='''Compute nash
equilibria in a game.''')
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


def command(args, prog, print_help=False):
    _PARSER.prog = '{} {}'.format(_PARSER.prog, prog)
    if print_help:
        _PARSER.print_help()
        return
    args = _PARSER.parse_args(args)
    game = rsgame.Game.from_json(json.load(args.input))

    if args.type == 'pure':
        equilibria = list(pure_nash(game, args.regret))
        if args.one and not equilibria:
            equilibria = [min_regret_profile(game)]
    elif args.type == 'mixed':
        equilibria = [eq.trim_support(args.support) for eq
                      in mixed_nash(game, args.regret, args.distance,
                                    args.random_points, args.one,
                                    max_iters=args.max_iterations,
                                    converge_thresh=args.convergence)]
    elif args.type == 'mrp':
        equilibria = [min_regret_profile(game)]
    else:
        raise ValueError('Unknown command given: {}'.format(args.type))

    json.dump(equilibria, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
