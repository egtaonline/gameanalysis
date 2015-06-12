"""Module for computing nash equilibria"""
import sys
import itertools
import numpy as np
import numpy.linalg as linalg

from gameanalysis import regret


_TINY = np.finfo(float).tiny


# TODO Maybe have all equilibria methods return regret too. Easier to throw
# away then to recompute.
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
            # TODO This should probably be switched to a logging utility
            sys.stderr.write('{0:d}: mix = {1}, regret = {2:f}\n'.format(
                i + 1,
                mix,
                regret.mixture_regret(game, mix)))
    return np.maximum(mix, 0)  # Probabilities are occasionally negative
