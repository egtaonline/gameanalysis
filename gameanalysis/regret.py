"""A module for computing regret and social welfare of profiles"""
import numpy as np


def pure_strategy_deviation_pays(game, profile):
    """Returns the pure strategy deviation payoffs

    The result is a compact array of deviation payoffs. Each element
    corresponds to the payoff of deviating to strategy i from strategy j for
    all valid deviations."""
    profile = np.asarray(profile, int)
    pays = game.get_payoffs(profile)
    devs = np.empty(game.num_devs)

    for dev_ind, (from_ind, to_ind) in enumerate(zip(
            game.dev_from_indices, game.dev_to_indices)):
        if profile[from_ind] == 0:
            devs[dev_ind] = 0
        elif from_ind == to_ind:
            devs[dev_ind] = pays[from_ind]
        else:
            prof_copy = profile.copy()
            prof_copy[from_ind] -= 1
            prof_copy[to_ind] += 1
            devs[dev_ind] = game.get_payoffs(prof_copy)[to_ind]
    return devs


def pure_strategy_deviation_gains(game, profile):
    """Returns the pure strategy deviations gains"""
    profile = np.asarray(profile, int)
    pays = game.get_payoffs(profile)
    devs = pure_strategy_deviation_pays(game, profile)
    return devs - pays.repeat(game.num_strat_devs)


def pure_strategy_regret(game, prof):
    """Returns the regret of a pure strategy profile

    If prof has more than one dimension, the last dimension is taken as a set
    of profiles and returned as a new array."""
    with np.errstate(invalid='ignore'): # keep nans
        return pure_strategy_deviation_gains(game, prof).max()


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


def max_pure_social_welfare(game, *, by_role=False):
    """Returns the maximum social welfare over the known profiles.

    If by_role is specified, then max social welfare applies to each role
    independently. If there are no profiles with full payoff data for a role,
    an arbitrary profile will be returned."""
    if by_role: # pylint: disable=no-else-return
        if game.num_profiles: # pylint: disable=no-else-return
            welfares = np.add.reduceat(
                game.profiles() * game.payoffs(), game.role_starts, 1)
            prof_inds = np.nanargmax(welfares, 0)
            return (welfares[prof_inds, np.arange(game.num_roles)],
                    game.profiles()[prof_inds])
        else:
            welfares = np.full(game.num_roles, np.nan)
            profiles = np.full(game.num_roles, None)
            return welfares, profiles

    else:
        if game.num_complete_profiles: # pylint: disable=no-else-return
            welfares = np.einsum('ij,ij->i', game.profiles(), game.payoffs())
            prof_ind = np.nanargmax(welfares)
            return welfares[prof_ind], game.profiles()[prof_ind]
        else:
            return np.nan, None
