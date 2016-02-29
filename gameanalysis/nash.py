"""Module for computing nash equilibria"""
import sys
import itertools
import numpy as np
import numpy.linalg as linalg

from gameanalysis import regret


_TINY = np.finfo(float).tiny


def pure_nash(game, epsilon=0):
    """Returns a generator of all pure-strategy epsilon-Nash equilibria."""
    return (profile for profile in game
            if regret.pure_strategy_regret(game, profile) <= epsilon)


def min_regret_profile(game):
    """Finds the profile with the confirmed lowest regret."""
    return min((regret.pure_strategy_regret(game, prof), i, prof)
               for i, prof in enumerate(game))[2]


def min_regret_grid_mixture(game, points):
    """Finds the mixed profile with the confirmed lowest regret

    The search is done over a grid with `points` per dimensions.

    Arguments
    ---------
    points : int > 1
        Number of points per dimension to search.
    """
    return game.as_mixture(
        min((regret.mixture_regret(game, mix), i, mix)
            for i, mix
            in enumerate(game.grid_mixtures(points, as_array=True)))[2])


def min_regret_rand_mixture(game, mixtures):
    """Finds the mixed profile with the confirmed lowest regret

    The search is done over a random sampling of `mixtures` mixed profiles.

    Arguments
    ---------
    mixtures : int > 0
        Number of mixtures to evaluate the regret of.
    """
    return game.as_mixture(
        min((regret.mixture_regret(game, mix), i, mix)
            for i, mix
            in enumerate(game.random_mixtures(mixtures, as_array=True)))[2])


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
    wrap = (lambda x: x) if as_array else game.as_mixture
    equilibria = []
    best = (np.inf, -1, None)  # Best convergence so far

    # TODO parallelize this loop
    for i, mix in enumerate(itertools.chain(
            game.pure_mixtures(as_array=True),
            game.biased_mixtures(as_array=True),
            game.role_biased_mixtures(as_array=True),
            [game.uniform_mixture(as_array=True)],
            game.random_mixtures(random_restarts, as_array=True))):
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
        mix = (game.expected_values(mix, as_array=True)
               - game.min_payoffs(True)[:, np.newaxis] + _TINY) * mix
        mix = mix / mix.sum(1, keepdims=True)
        if linalg.norm(mix - old_mix) <= converge_thresh:
            break
        if verbose:
            # TODO This should probably be switched to a logging utility
            sys.stderr.write('{0:d}: mix = {1}, regret = {2:f}\n'.format(
                i + 1,
                mix,
                regret.mixture_regret(game, mix)))
    return np.maximum(mix, 0)  # Probabilities are occasionally negative
