"""A module for computing regret and social welfare of profiles"""
import math

import numpy as np


def pure_strategy_deviation_gains(game, prof):
    """Returns a nested dict containing all of the gains from deviation

    The nested dict maps role to strategy to deviation to gain. The profile
    must have data.
    """
    prof = game.as_profile(prof)
    return {role:
            {strat:
             {dev: game.get_payoff(
                 prof.deviate(role, strat, dev),  # noqa
                 role, dev, default=np.nan) - payoff  # noqa
              for dev in game.strategies[role]}  # noqa
             for strat, payoff in strat_payoffs.items()}
            for role, strat_payoffs in game.get_payoffs(prof).items()}


def pure_strategy_regret(game, prof):
    """Returns the regret of a pure strategy profile in a game"""
    gains = pure_strategy_deviation_gains(game, prof)
    if any(any(any(math.isnan(gain) for gain in dev_gain.values())
               for dev_gain in strat_gain.values())
           for strat_gain in gains.values()):
        return float('nan')
    else:
        return max(max(max(gain for gain in dev_gain.values())
                       for dev_gain in strat_gain.values())
                   for strat_gain in gains.values())


def mixture_deviation_gains(game, mix, assume_complete=False, as_array=False):
    """Returns all the gains from deviation from a mixed strategy

    Return type is a dict mapping role to deviation to gain. This is equivalent
    to what is sometimes called equilibrium regret."""
    mix = game.as_mixture(mix, as_array=True)
    strategy_evs = game.deviation_payoffs(mix, assume_complete=assume_complete,
                                          as_array=True)
    # strategy_evs is nan where there's no data, however, if it's not played in
    # the mix, it doesn't effect the role_evs
    masked = strategy_evs.copy()
    masked[mix == 0] = 0
    role_evs = game.role_reduce(masked * mix, keepdims=True)
    gains = strategy_evs - role_evs

    if as_array:
        return gains
    else:
        return game.as_dict(gains, filter_zeros=False)


def mixture_regret(game, mix):
    """Return the regret of a mixture profile"""
    return mixture_deviation_gains(game, mix, as_array=True).max()


def pure_social_welfare(game, profile):
    """Returns the social welfare of a pure strategy profile in game"""
    profile = game.as_profile(profile, as_array=True)
    return np.sum(profile * game.get_payoffs(profile, as_array=True))


def mixed_social_welfare(game, mix):
    """Returns the social welfare of a mixed strategy profile"""
    return game.get_expected_payoff(mix, as_array=True).dot(game.aplayers)


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
