#!/usr/bin/env python3
import sys
import itertools
import numpy as np
import numpy.linalg as linalg

import regret


def pure_nash(game, epsilon=0):
    '''Returns a generator of all pure-strategy epsilon-Nash equilibria.'''
    return (profile for profile in game
            if regret.pure_strategy_regret(game, profile) <= epsilon)


def min_regret_profile(game):
    '''Finds the profile with the confirmed lowest regret.

    Returns a tuple of the minimum regret and the the profile with the minimum
    regret.

    '''
    return min((regret.pure_strategy_regret(game, prof), prof)
               for prof in game)


def mixed_nash(game, regret_thresh=1e-3, dist_thresh=1e-3, random_restarts=0,
               at_least_one=False, *rd_args, **rd_kwargs):
    '''Runs replicator dynamics from multiple starting mixtures.

    regret_thresh:   The threshold to consider an equilibrium found
    dist_thresh:     The threshold for considering equilibria distinct
    random_restarts: The number of random initializations for replicator
                     dynamics
    at_least_one:    Returns the minimum regret mixture found by replicator
                     dynamics if no equilibria were within the regret threshold
    rd_*:            Extra arguments to pass through to replicator dynamics

    '''
    # TODO Add pure strategies to set
    equilibria = []
    best = (np.inf, None)  # Best convergence so far

    for mix in itertools.chain(
            game.biasedMixtures(),
            (game.uniformMixture(),),
            (game.randomMixture() for _ in range(random_restarts))):
        eq = replicator_dynamics(game, mix, *rd_args, **rd_kwargs)
        # TODO More efficient way to check distinctness
        reg = regret.mixture_regret(game, eq)
        if reg <= regret_thresh and \
           all(linalg.norm(e - eq, 2) >= dist_thresh for e in equilibria):
            equilibria.append(eq)
            yield eq
        best = min(best, (reg, eq))
    if at_least_one and not equilibria:
        yield best[1]


def _replicator_dynamics(game, mix, max_iters=10000, converge_thresh=1e-8,
                         verbose=False):
    '''Replicator dynamics

    This will run at most max_iters of replicators dynamics and return unless
    the difference between successive mixtures is less than converge_thresh.

    '''
    tiny = np.finfo(float).tiny
    for i in range(max_iters):
        old_mix = mix
        mix = (game.expectedValues(mix) - game.minPayoffs + tiny) * mix
        mix = mix / mix.sum(1)[:, np.newaxis]
        if linalg.norm(mix - old_mix) <= converge_thresh:
            break
        if verbose:
            sys.stderr.write('%d: mix = %s, regret = %f\n' % (
                i + 1,
                mix,
                regret.mixture_regret(game, mix)))
    return np.maximum(mix, 0)  # Probabilities are occasionally negative


def parse_args():
    parser = io_parser()
    parser.add_argument('-r', metavar='REGRET', type=float, default=1e-3, \
            help='Max allowed regret for approximate Nash equilibria. ' + \
            'default=1e-3')
    parser.add_argument('-d', metavar="DISTANCE", type=float, default=1e-3, \
            help="L2-distance threshold to consider equilibria distinct. " + \
            "default=1e-3")
    parser.add_argument("-c", metavar="CONVERGENCE", type=float, default=1e-8, \
            help="Replicator dynamics convergence thrshold. default=1e-8")
    parser.add_argument("-i", metavar="ITERATIONS", type=int, default=10000, \
            help="Max replicator dynamics iterations. default=1e4")
    parser.add_argument("-s", metavar="SUPPORT", type=float, default=1e-3, \
            help="Min probability for a strategy to be considered in " + \
            "support. default=1e-3")
    parser.add_argument("-type", choices=["mixed", "pure", "mrp"], default= \
            "mixed", help="Type of approximate equilibrium to compute: " + \
            "role-symmetric mixed-strategy Nash, pure-strategy Nash, or " + \
            "min-regret profile. default=mixed")
    parser.add_argument("-p", metavar="POINTS", type=int, default=0, \
            help="Number of random points from which to initialize " + \
            "replicator dynamics in addition to the default set of uniform " +\
            "and heavily-biased mixtures.")
    parser.add_argument("--one", action="store_true", help="Always report " +\
            "at least one equilibrium per game.")
    args = parser.parse_args()
    games = args.input
    if not isinstance(games, list):
        games = [games]
    return games, args


def main():
    games, args = parse_args()
    if args.type == 'pure':
        equilibria = [pure_nash(g, args.r) for g in games]
        if args.one:
            for i in range(len(games)):
                if len(equilibria[i]) == 0:
                    equilibria[i] = min_regret_profile(games[i])
    elif args.type == 'mixed':
        equilibria = [[g.toProfile(eq, args.s) for eq in mixed_nash(g, \
                args.r, args.d, args.p, args.one, iters=args.i, \
                converge_thresh=args.c)] for g in games]
    elif args.type == 'mrp':
        equilibria = map(min_regret_profile, games)
    if len(equilibria) > 1:
        print to_JSON_str(equilibria)
    else:
        print to_JSON_str(equilibria[0])


if __name__ == '__main__':
    main()
