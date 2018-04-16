"""Identity reduction

This is the same as not reducing a game.
"""
import numpy as np

from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import utils


def reduce_game(full_game, red_players=None):
    """Return original game

    Parameters
    ----------
    full_game : Game
        The game to reduce.
    red_players : ndarray-like, optional
        If specified, this must match the number of players per role in
        full_game.
    """
    utils.check(
        red_players is None or np.all(
            full_game.num_role_players == red_players),
        'identity reduction must have same number of players')
    return paygame.game_copy(full_game)


def expand_profiles(full_game, profiles):
    """Return input profiles

    Parameters
    ----------
    full_game : Game
        Game that all profiles must be valid for.
    profiles : ndarray-like
        The profiles.
    axis : int, optional
        The axis the profiles lie on.
    """
    profiles = np.asarray(profiles, int)
    utils.check(
        full_game.is_profile(profiles).all(),
        'profiles must be valid')
    return profiles.reshape((-1, full_game.num_strats))


def reduce_profiles(red_game, profiles):
    """Return original profiles

    Parameters
    ----------
    red_game : Game
        Game that all profiles must be valid for.
    profiles : ndarray-like
        The profiles.
    axis : int, optional
        The axis the profiles are on.
    """
    profiles = np.asarray(profiles, int)
    utils.check(
        red_game.is_profile(profiles).all(),
        'profiles must be valid')
    return profiles.reshape((-1, red_game.num_strats))


def expand_deviation_profiles(
        full_game, rest, red_players=None, role_index=None):
    """Expand all deviation profiles from a restriction

    Parameters
    ----------
    full_game : Game
        The game the deviations profiles will be valid for.
    rest : [bool]
        The restriction to get deviations from.
    red_players : [int], optional
        The number of players in each role in the reduced game.IF
        specified, it must match the number for full_game.
    role_index : int, optional
        If specified , only expand deviations for the role selected.
    """
    utils.check(
        red_players is None or np.all(
            full_game.num_role_players == red_players),
        'identity reduction must have same number of players')
    return restrict.deviation_profiles(full_game, rest, role_index)
