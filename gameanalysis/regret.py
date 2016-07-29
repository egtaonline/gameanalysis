"""A module for computing regret and social welfare of profiles"""
import numpy as np


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
