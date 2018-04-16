"""Hierarchical reduction"""
import numpy as np

from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import _common


def reduce_game(full_game, red_players):
    """Reduce a game using hierarchical reduction

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
        np.all(red_game.num_role_players > 0),
        'all reduced players must be greater than zero')
    utils.check(
        np.all(full_game.num_role_players >= red_game.num_role_players),
        'all full counts must not be less than reduced counts')

    if full_game.is_empty():
        return red_game
    elif full_game.num_profiles < red_game.num_all_profiles:
        profiles = full_game.profiles()
        payoffs = full_game.payoffs()
    else:
        profiles = expand_profiles(
            full_game, red_game.all_profiles())
        payoffs = full_game.get_payoffs(profiles)
        valid = ~np.all(np.isnan(payoffs) | (profiles == 0), 1)
        profiles = profiles[valid]
        payoffs = payoffs[valid]

    red_profiles, mask = _common.reduce_profiles(
        full_game, red_game.num_role_players[None], profiles)
    return paygame.game_replace(red_game, red_profiles, payoffs[mask])


def reduce_profiles(red_game, profiles):
    """Reduce profiles hierarchically

    Parameters
    ----------
    red_game : Game
        Game that reduced profiles will be profiles for.
    profiles : ndarray-like
        The profiles to reduce.
    """
    profiles = np.asarray(profiles, int)
    utils.check(
        profiles.shape[-1] == red_game.num_strats,
        'profiles must be appropriate shape')
    return _common.reduce_profiles(
        red_game, red_game.num_role_players[None],
        profiles.reshape((-1, red_game.num_strats)))[0]


def expand_profiles(full_game, profiles):
    """Expand profiles hierarchically

    Parameters
    ----------
    full_game : Game
        Game that expanded profiles will be valid for.
    profiles : ndarray-like
        The profiles to expand
    """
    profiles = np.asarray(profiles, int)
    utils.check(
        profiles.shape[-1] == full_game.num_strats,
        'profiles must be appropriate shape')
    return _common.expand_profiles(
        full_game, full_game.num_role_players[None],
        profiles.reshape((-1, full_game.num_strats)))


def expand_deviation_profiles(
        full_game, rest, red_players, role_index=None):
    """Expand all deviation profiles from a restricted game

    Parameters
    ----------
    full_game : Game
        The game the deviations profiles will be valid for.
    rest : [bool]
        The restriction to get deviations from.
    red_players : ndarray-like
        The number of players in each role in the reduced game.
    role_index : int, optional
        If specified , only expand deviations for the role selected.
    """
    utils.check(
        full_game.is_restriction(rest), 'restriction must be valid')
    return expand_profiles(
        full_game, restrict.deviation_profiles(
            rsgame.empty(red_players, full_game.num_role_strats),
            rest, role_index))
