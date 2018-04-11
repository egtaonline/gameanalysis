"""A module for computing regret and social welfare of profiles"""
import numpy as np


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
    return np.fromiter(  # pragma: no branch
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
