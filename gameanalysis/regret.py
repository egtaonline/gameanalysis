"""A module for computing regret and social welfare of profiles"""
import itertools
import multiprocessing

import numpy as np
from scipy import optimize


def pure_strategy_deviation_gains(game, profile):
    """Returns the pure strategy deviations gains

    The result is a compact array of deviation gains. Each element corresponds
    to the deviation from strategy i to strategy j ordered by (i, j) for all
    valid deviations."""
    profile = np.asarray(profile, int)
    dev_profs = profile[None].repeat(game.num_devs, 0)
    dev_profs[np.arange(game.num_devs), game.dev_from_indices] -= 1
    dev_profs[np.arange(game.num_devs), game.dev_to_indices] += 1

    pays = game.get_payoffs(profile)
    return np.fromiter(
        (game.get_payoffs(prof)[t] - pays[f] if np.all(prof >= 0) else 0
         for prof, f, t
         in zip(dev_profs, game.dev_from_indices, game.dev_to_indices)),
        float, game.num_devs)


def pure_strategy_regret(game, prof):
    """Returns the regret of a pure strategy profile

    If prof has more than one dimension, the last dimension is taken as a set
    of profiles and returned as a new array."""
    return max(pure_strategy_deviation_gains(game, prof).max(), 0)


def mixture_deviation_gains(game, mix):
    """Returns all the gains from deviation from a mixed strategy

    The result is ordered by role, then strategy."""
    mix = np.asarray(mix, float)
    strategy_evs = game.deviation_payoffs(mix)
    # strategy_evs is nan where there's no data, however, if it's not played in
    # the mix, it doesn't effect the role_evs
    masked = strategy_evs.copy()
    masked[mix == 0] = 0
    role_evs = np.add.reduceat(
        masked * mix, game.role_starts).repeat(game.num_role_strats)
    return strategy_evs - role_evs


def mixture_regret(game, mix):
    """Return the regret of a mixture profile"""
    mix = np.asarray(mix, float)
    return mixture_deviation_gains(game, mix).max()


def pure_social_welfare(game, profile):
    """Returns the social welfare of a pure strategy profile in game"""
    profile = np.asarray(profile, int)
    return game.get_payoffs(profile).dot(profile)


def mixed_social_welfare(game, mix):
    """Returns the social welfare of a mixed strategy profile"""
    return game.expected_payoffs(mix).dot(game.num_role_players)


class _SocialWelfareOptimizer(object):
    """A pickleable object to find maximal social welfare

    This method uses constrained convex optimization to to attempt to maximize
    mixed social welfare."""

    def __init__(self, game, gtol=1e-8):
        self.game = game
        self.scale = game.max_role_payoffs() - game.min_role_payoffs()
        self.scale[self.scale == 0] = 1  # In case payoffs are the same
        self.offset = game.min_role_payoffs()
        self.gtol = gtol

    def obj_func(self, mix, penalty):  # pragma: no cover
        # We assume that the initial point is in a constant sum subspace, and
        # so project the gradient so that any gradient step maintains that
        # constant step. Thus, sum to 1 is not one of the penalty terms

        # Because deviation payoffs uses log space, we max with 0 just for the
        # payoff calculation
        ep, ep_jac = self.game.expected_payoffs(
            np.maximum(0, mix), jacobian=True)
        # Normalize so payoffs are effectively in [0, 1]
        ep = (ep - self.offset) / self.scale
        ep_jac /= self.scale[:, None]

        # Compute normalized negative walfare (minimization)
        welfare = -self.game.num_role_players.dot(ep)
        dwelfare = -self.game.num_role_players.dot(ep_jac)

        # Add penalty for negative mixtures
        welfare += penalty * np.sum(np.minimum(mix, 0) ** 2) / 2
        dwelfare += penalty * np.minimum(mix, 0)

        # Project grad so steps stay in the simplex (more or less)
        dwelfare -= np.repeat(np.add.reduceat(dwelfare, self.game.role_starts)
                              / self.game.num_role_strats,
                              self.game.num_role_strats)
        return welfare, dwelfare

    def __call__(self, mix):  # pragma: no cover
        # Pass in lambda, and make penalty not a member

        result = None
        penalty = np.sum(self.game.num_role_players)
        for _ in range(30):
            # First get an unconstrained result from the optimization
            with np.errstate(over='raise', invalid='raise'):
                try:
                    opt = optimize.minimize(
                        lambda m: self.obj_func(m, penalty), mix, method='CG',
                        jac=True, options={'gtol': self.gtol})
                except FloatingPointError:  # pragma: no cover
                    penalty *= 2
                    continue

            mix = opt.x
            # Project it onto the simplex, it might not be due to the penalty
            result = self.game.mixture_project(mix)
            if np.allclose(mix, result):
                break
            # Increase constraint penalty
            penalty *= 2

        return result


def max_mixed_social_welfare(game, *, grid_points=2, random_restarts=0,
                             processes=None, **swopt_args):
    """Returns the maximum role symmetric mixed social welfare profile

    Arguments
    ---------
    grid_points : int > 1, optional
        The number of grid points to use for mixture seeds. two implies just
        pure mixtures, more will be denser, but scales exponentially with the
        dimension.
    random_restarts : int, optional
        The number of random initializations.
    processes : int, optional
        Number of processes to use when finding Nash equilibria. The game needs
        to be pickleable.
    """
    assert game.is_complete(), \
        "Max welfare finding only works on complete games"""

    initial_points = list(itertools.chain(
        [game.uniform_mixture()],
        game.grid_mixtures(grid_points),
        game.biased_mixtures(),
        game.role_biased_mixtures(),
        game.random_mixtures(random_restarts)))
    chunksize = len(initial_points) if processes == 1 else 4

    best = (-np.inf, -1, None)

    opt = _SocialWelfareOptimizer(game, **swopt_args)
    with multiprocessing.Pool(processes) as pool:
        for i, mix in enumerate(pool.imap_unordered(
                opt, initial_points, chunksize=chunksize)):
            welfare = mixed_social_welfare(game, mix)
            best = max(best, (welfare, i, mix))

    return best[0], best[2]


def max_pure_social_welfare(game, *, by_role=False):
    """Returns the maximum social welfare over the known profiles.

    If by_role is specified, then max social welfare applies to each role
    independently."""
    if by_role:
        if game.num_complete_profiles:
            # TODO technically you could have no complete profiles, but full
            # payoff data for all roles
            welfares = np.add.reduceat(
                game.profiles() * game.payoffs(), game.role_starts, 1)
            prof_inds = np.nanargmax(welfares, 0)
            return (welfares[prof_inds, np.arange(game.num_roles)],
                    game.profiles()[prof_inds])
        else:
            welfares = np.empty(game.num_roles)
            welfares.fill(np.nan)
            profiles = np.empty(game.num_roles, dtype=object)
            profiles.fill(None)
            return welfares, profiles

    else:
        if game.num_complete_profiles:
            welfares = np.sum(game.profiles() * game.payoffs(), 1)
            prof_ind = np.nanargmax(welfares)
            return welfares[prof_ind], game.profiles()[prof_ind]
        else:
            return np.nan, None
