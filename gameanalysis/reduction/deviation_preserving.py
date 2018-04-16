"""Deviation preserving reduction"""
import numpy as np

from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import _common
from gameanalysis.reduction import hierarchical


def _devs(game, num_profs):
    """Return an array of the player counts after deviation"""
    return np.tile(np.repeat(
        game.num_role_players - np.eye(game.num_roles, dtype=int),
        game.num_role_strats, 0), (num_profs, 1))


def reduce_game(full_game, red_players): # pylint: disable=too-many-locals
    """Reduce a game using deviation preserving reduction

    Parameters
    ----------
    full_game : Game
        The game to reduce.
    red_players : ndarray-like
        The reduced number of players for each role. This will be coerced
        into the proper shape if necessary.
    """
    red_game = rsgame.empty_names(
        full_game.role_names, red_players, full_game.strat_names)
    utils.check(
        np.all((red_game.num_role_players > 1) |
               (full_game.num_role_players == 1)),
        'all reduced players must be greater than zero')
    utils.check(
        np.all(full_game.num_role_players >= red_game.num_role_players),
        'all full counts must not be less than reduced counts')

    if full_game.is_empty():
        return red_game
    elif full_game.num_profiles < red_game.num_all_dpr_profiles:
        full_profiles = full_game.profiles()
        full_payoffs = full_game.payoffs()
    else:
        full_profiles = expand_profiles(
            full_game, red_game.all_profiles())
        full_payoffs = full_game.get_payoffs(full_profiles)
        valid = ~np.all(np.isnan(full_payoffs) |
                        (full_profiles == 0), 1)
        full_profiles = full_profiles[valid]
        full_payoffs = full_payoffs[valid]

    # Reduce
    red_profiles, red_inds, full_inds, strat_inds = _reduce_profiles(
        red_game, full_profiles, True)

    if red_profiles.size == 0:  # Empty reduction
        return red_game

    # Build mapping from payoffs to reduced profiles, and use bincount
    # to count the number of payoffs mapped to a specific location, and
    # sum the number of payoffs mapped to a specific location
    cum_inds = red_inds * full_game.num_strats + strat_inds
    payoff_vals = full_payoffs[full_inds, strat_inds]
    red_payoffs = np.bincount(
        cum_inds, payoff_vals, red_profiles.size).reshape(
            red_profiles.shape)
    red_payoff_counts = np.bincount(
        cum_inds, minlength=red_profiles.size).reshape(
            red_profiles.shape)
    mask = red_payoff_counts > 1
    red_payoffs[mask] /= red_payoff_counts[mask]

    unknown = (red_profiles > 0) & (red_payoff_counts == 0)
    red_payoffs[unknown] = np.nan
    valid = ~np.all((red_profiles == 0) | np.isnan(red_payoffs), 1)
    return paygame.game_replace(red_game, red_profiles[valid],
                                red_payoffs[valid])


def expand_profiles(full_game, profiles): # pylint: disable=too-many-locals
    """Expand profiles using dpr

    Parameters
    ----------
    full_game : Game
        Game that expanded profiles will be valid for.
    profiles : ndarray-like
        The profiles to expand
    return_contributions : bool, optional
        If specified, returns a boolean array matching the shape is
        returned indicating the payoffs that are needed for the initial
        profiles.
    """
    profiles = np.asarray(profiles, int)
    utils.check(
        profiles.shape[-1] == full_game.num_strats,
        'profiles not a valid shape')
    if not profiles.size:
        return np.empty((0, full_game.num_strats), int)
    profiles = profiles.reshape((-1, full_game.num_strats))
    all_red_players = np.add.reduceat(profiles, full_game.role_starts, 1)
    red_players = all_red_players[0]
    utils.check(
        np.all(all_red_players == red_players), 'profiles must be valid')

    num_profs = profiles.shape[0]
    dev_profs = profiles[:, None] - np.eye(full_game.num_strats, dtype=int)
    dev_profs = np.reshape(dev_profs, (-1, full_game.num_strats))
    dev_full_players = _devs(full_game, num_profs)

    mask = ~np.any(dev_profs < 0, 1)
    devs = (np.eye(full_game.num_strats, dtype=bool)[None]
            .repeat(num_profs, 0)
            .reshape((-1, full_game.num_strats))[mask])
    dev_full_profs = _common.expand_profiles(
        full_game, dev_full_players[mask], dev_profs[mask]) + devs
    ids = utils.axis_to_elem(dev_full_profs)
    return dev_full_profs[np.unique(ids, return_index=True)[1]]


def reduce_profiles(red_game, profiles):
    """Reduce profiles using dpr

    Parameters
    ----------
    red_game : Game
        Game that reduced profiles will be profiles for.
    profiles : ndarray-like
        The profiles to reduce.
    """
    return _reduce_profiles(red_game, profiles, False)


def _reduce_profiles(red_game, profiles, return_contributions): # pylint: disable=too-many-locals
    """Reduce profiles using dpr

    Parameters
    ----------
    red_game : Game
        Game that reduced profiles will be profiles for.
    profiles : ndarray-like
        The profiles to reduce.
    return_contributions : bool, optional
        If true return ancillary information about where the payoffs come
        from.
    """
    profiles = np.asarray(profiles, int)
    utils.check(
        profiles.shape[-1] == red_game.num_strats,
        'profiles not a valid shape')
    if not profiles.size:
        return np.empty((0, red_game.num_strats), int)

    profiles = profiles.reshape((-1, red_game.num_strats))
    all_full_players = np.add.reduceat(profiles, red_game.role_starts, 1)
    full_players = all_full_players[0]
    utils.check(
        np.all(all_full_players == full_players), 'profiles must be valid')

    num_profs = profiles.shape[0]
    dev_profs = profiles[:, None] - np.eye(red_game.num_strats, dtype=int)
    dev_profs = np.reshape(dev_profs, (-1, red_game.num_strats))
    dev_red_players = _devs(red_game, num_profs)
    mask = ~np.any(dev_profs < 0, 1)
    red_profs, reduced = _common.reduce_profiles(
        red_game, dev_red_players[mask], dev_profs[mask])
    devs = (np.eye(red_game.num_strats, dtype=int)[None]
            .repeat(num_profs, 0)
            .reshape((-1, red_game.num_strats))[mask][reduced])
    red_profs += devs
    red_profs, red_inds = np.unique(
        utils.axis_to_elem(red_profs), return_inverse=True)
    red_profs = utils.axis_from_elem(red_profs)
    if not return_contributions:
        return red_profs

    full_inds = np.arange(num_profs)[:, None].repeat(
        red_game.num_strats, 1).flat[mask][reduced]
    strat_inds = devs.nonzero()[1]
    return red_profs, red_inds, full_inds, strat_inds


def expand_deviation_profiles(
        full_game, rest, red_players, role_index=None):
    """Expand all deviation profiles from a restriction

    Parameters
    ----------
    full_game : Game
        The game the deviations profiles will be valid for.
    rest : [bool]
        The restriction to get deviations from.
    red_players : [int]
        The number of players in each role in the reduced game.
    role_index : int, optional
        If specified , only expand deviations for the role selected.
    """
    rest = np.asarray(rest, bool)
    rdev = np.eye(full_game.num_roles, dtype=int)
    red_players = np.broadcast_to(np.asarray(red_players, int),
                                  full_game.num_roles)
    support = np.add.reduceat(rest, full_game.role_starts)

    def dev_profs(red_players, full_players, mask, rst):
        """Deviation profiles for a particular role"""
        rgame = rsgame.empty(red_players, support)
        sub_profs = restrict.translate(rgame.all_profiles(), rest)
        game = rsgame.empty(full_players, full_game.num_role_strats)
        non_devs = hierarchical.expand_profiles(game, sub_profs)
        ndevs = np.sum(~mask)
        devs = np.zeros((ndevs, full_game.num_strats), int)
        devs[:, rst:rst + mask.size][:, ~mask] = np.eye(ndevs, dtype=int)
        profs = non_devs[:, None] + devs
        profs.shape = (-1, full_game.num_strats)
        return profs

    if role_index is None: # pylint: disable=no-else-return
        expanded_profs = [dev_profs(red_players, full_players, mask, rs)
                          for red_players, full_players, mask, rs
                          in zip(red_players - rdev,
                                 full_game.num_role_players - rdev,
                                 np.split(rest,
                                          full_game.role_starts[1:]),
                                 full_game.role_starts)]
        return np.concatenate(expanded_profs)

    else:
        full_players = full_game.num_role_players.copy()
        full_players[role_index] -= 1
        red_players = red_players.copy()
        red_players[role_index] -= 1
        mask = np.split(rest, full_game.role_starts[1:])[
            role_index]
        rstart = full_game.role_starts[role_index]
        return dev_profs(red_players, full_players, mask, rstart)
