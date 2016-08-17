"""A module for computing regret and social welfare of profiles"""
import itertools
import multiprocessing

import numpy as np
from scipy import optimize


def pure_strategy_deviation_gains(game, prof):
    """Returns the pure strategy deviations gains

    The result is a compact array of deviation gains. Each element corresponds
    to the deviation from strategy i to strategy j ordered by (i, j) for all
    valid deviations."""
    prof = np.asarray(prof, int)
    supp = prof > 0
    num_supp = game.role_reduce(supp)
    from_inds = np.arange(game.num_role_strats)[supp]
    reps = game.num_strategies[game.role_index[from_inds]]
    num_devs = np.sum(num_supp * (game.num_strategies - 1))

    to_inds = np.ones(reps.sum(), int)
    to_inds[0] = 0
    to_inds[reps[:-1].cumsum()] -= reps[:-1]
    role_inds = (num_supp * game.num_strategies)[:-1].cumsum()
    to_inds[role_inds] += game.num_strategies[:-1]
    to_inds = to_inds.cumsum()
    to_inds = to_inds[to_inds != from_inds.repeat(reps)]
    from_inds = from_inds.repeat(reps - 1)

    pays = game.get_payoffs(prof)[from_inds]
    dev_profs = prof[None].repeat(num_devs, 0)
    dev_profs[np.arange(num_devs), from_inds] -= 1
    dev_profs[np.arange(num_devs), to_inds] += 1
    dev_pays = np.array([game.get_payoffs(dprof)[to]
                         for dprof, to in zip(dev_profs, to_inds)])
    return dev_pays - pays


def pure_strategy_regret(game, prof):
    """Returns the regret of a pure strategy profile

    If prof has more than one dimension, the last dimension is taken as a set
    of profiles and returned as a new array."""
    prof = np.asarray(prof, int)
    return max(pure_strategy_deviation_gains(game, prof).max(), 0)


def mixture_deviation_gains(game, mix, assume_complete=False):
    """Returns all the gains from deviation from a mixed strategy

    The result is ordered by role, then strategy."""
    mix = np.asarray(mix, float)
    strategy_evs = game.deviation_payoffs(mix, assume_complete=assume_complete)
    # strategy_evs is nan where there's no data, however, if it's not played in
    # the mix, it doesn't effect the role_evs
    masked = strategy_evs.copy()
    masked[mix == 0] = 0
    role_evs = game.role_reduce(masked * mix, keepdims=True)
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
    return game.get_expected_payoffs(mix).dot(game.num_players)


class SocialWelfareOptimizer(object):
    """A pickleable object to find Nash equilibria

    This method uses constrained convex optimization to to attempt to solve a
    proxy for the nonconvex regret minimization."""
    def __init__(self, game, gtol=1e-8):
        self.game = game
        self.scale = game.max_payoffs() - game.min_payoffs()
        self.scale[self.scale == 0] = 1  # In case payoffs are the same
        self.offset = game.min_payoffs()
        self.gtol = gtol

    def obj_func(self, mix, penalty):
        # We assume that the initial point is in a constant sum subspace, and
        # so project the gradient so that any gradient step maintains that
        # constant step. Thus, sum to 1 is not one of the penalty terms

        # Because deviation payoffs uses log space, we max with 0 just for the
        # payoff calculation
        ep, ep_jac = self.game.get_expected_payoffs(
            np.maximum(0, mix), assume_complete=True, jacobian=True)
        # Normalize so payoffs are effectively in [0, 1]
        ep = (ep - self.offset) / self.scale
        ep_jac /= self.scale[:, None]

        # Compute normalized negative walfare (minimization)
        welfare = -self.game.num_players.dot(ep)
        dwelfare = -self.game.num_players.dot(ep_jac)

        # Add penalty for negative mixtures
        welfare += penalty * np.sum(np.minimum(mix, 0) ** 2) / 2
        dwelfare += penalty * np.minimum(mix, 0)

        # Project grad so steps stay in the simplex (more or less)
        dwelfare -= self.game.role_repeat(self.game.role_reduce(dwelfare) /
                                          self.game.num_strategies)
        return welfare, dwelfare

    def __call__(self, mix):
        # Pass in lambda, and make penalty not a member

        result = None
        penalty = np.sum(self.game.num_players)
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
            result = self.game.simplex_project(mix)
            if np.allclose(mix, result):
                break
            # Increase constraint penalty
            penalty *= 2

        return result


def max_mixed_social_welfare(game, grid_points=2, random_restarts=0,
                             processes=None, **swopt_args):
    """Returns the maximum role symmetric mixed social welfare profile

    Arguments
    ---------
    grid_points : int > 1
        The number of grid points to use for mixture seeds. two implies just
        pure mixtures, more will be denser, but scales exponentially with the
        dimension.
    random_restarts : int
        The number of random initializations.
    processes : int
        Number of processes to use when finding Nash equilibria. If greater
        than one, the game will need to be pickleable.
    """
    # XXX The code for this in game is rather complicated because of its
    # generality. This might be faster if that were not the case.
    assert game.is_complete(), \
        "Max welfare finding only works on complete games"""

    initial_points = itertools.chain(
        [game.uniform_mixture()],
        game.grid_mixtures(grid_points),
        game.biased_mixtures(),
        game.role_biased_mixtures(),
        game.random_mixtures(random_restarts))

    best = [-np.inf, None]  # Need a pointer for closure

    def process(mix):
        welfare = mixed_social_welfare(game, mix)
        if welfare > best[0]:
            best[0] = welfare
            best[1] = mix

    opt = SocialWelfareOptimizer(game, **swopt_args)
    if processes == 1:
        for mix in initial_points:
            process(opt(mix))
    else:
        with multiprocessing.Pool(processes) as pool:
            for mix in pool.imap_unordered(opt, initial_points):
                process(mix)

    return tuple(best)


def max_pure_social_welfare(game):
    """Get the max social welfare pure profile

    Returns a tuple of the max welfare and the corresponding profile"""
    mask = np.sum(game.profiles > 0, 1) == game.num_roles
    if mask.any():
        profiles = game.profiles[mask]
        welfares = np.sum(profiles * game.payoffs[mask], 1)
        prof_ind = np.nanargmax(welfares)
        return welfares[prof_ind], profiles[prof_ind]
    else:
        return np.nan, None


# def neighbors(game, p, *args, **kwargs):
#     if isinstance(p, Profile):
#         return profile_neighbors(game, p, *args, **kwargs)
#     elif isinstance(p, np.ndarray):
#         return mixture_neighbors(game, p, *args, **kwargs)
#     raise TypeError('unrecognized argument type: ' + type(p).__name__)


# def profile_neighbors(game, profile, role=None, strategy=None,
#                       deviation=None):
#     if role is None:
#         return list(chain(*[profile_neighbors(game, profile, r, strategy, \
#                 deviation) for r in game.roles]))
#     if strategy is None:
#         return list(chain(*[profile_neighbors(game, profile, role, s, \
#                 deviation) for s in profile[role]]))
#     if deviation is None:
#         return list(chain(*[profile_neighbors(game, profile, role, strategy, \ # noqa
#                 d) for d in set(game.strategies[role]) - {strategy}]))
#     return [profile.deviate(role, strategy, deviation)]


# def mixture_neighbors(game, mix, role=None, deviation=None):
#     n = set()
#     for profile in feasible_profiles(game, mix):
#         n.update(profile_neighbors(game, profile, role, deviation=deviation))
#     return n


# def feasible_profiles(game, mix, thresh=1e-3):
#     return [Profile({r:{s:p[game.index(r)].count(s) for s in set(p[ \
#             game.index(r)])} for r in game.roles}) for p in product(*[ \
#             CwR(filter(lambda s: mix[game.index(r), game.index(r,s)] >= \
#             thresh, game.strategies[r]), game.players[r]) for r \
#             in game.roles])]


# def symmetric_profile_regrets(game):
#     assert game.is_symmetric(), 'Game must be symmetric'
#     role = next(iter(game.strategies))
#     return {s: regret(game, rsgame.Profile({role:{s:game.players[role]}})) for s \ # noqa
#             in game.strategies[role]}
