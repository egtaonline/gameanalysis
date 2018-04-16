"""Twins reduction

This is the same as a deviation preserving reduction reduced to two for all
roles."""
import numpy as np

from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import deviation_preserving as dpr


def reduce_game(full_game, red_players=None):
    """Reduce a game using twins reduction

    Parameters
    ----------
    full_game : Game
        The game to reduce.
    red_players : ndarray-like, optional
        The reduced number of players for each role. This must be None or
        the reduced number of players for the twins reductions.
    """
    exp_red_players = np.minimum(full_game.num_role_players, 2)
    utils.check(
        red_players is None or np.all(exp_red_players == red_players),
        "twins reduction didn't get expected reduced players")
    return dpr.reduce_game(full_game, exp_red_players)


def expand_profiles(full_game, profiles):
    """Expand profiles using twins reduction

    Parameters
    ----------
    full_game : Game
        Game that expanded profiles will be valid for.
    profiles : ndarray-like
        The profiles to expand
    """
    red_players = np.minimum(full_game.num_role_players, 2)
    profiles = np.asarray(profiles, int)
    red_game = rsgame.empty(red_players, full_game.num_role_strats)
    utils.check(
        red_game.is_profile(profiles).all(), 'profiles must be valid')
    return dpr.expand_profiles(full_game, profiles)


def reduce_profiles(red_game, profiles):
    """Reduce profiles using twins

    Parameters
    ----------
    red_game : Game
        Game that reduced profiles will be profiles for. This game must
        have the valid twins reduction number of players.
    profiles : ndarray-like
        The profiles to reduce.
    """
    profiles = np.asarray(profiles, int)
    utils.check(
        np.all(red_game.num_role_players <= 2),
        'red game must be a twins game')
    return dpr.reduce_profiles(red_game, profiles)


def expand_deviation_profiles(full_game, rest, red_players=None,
                              role_index=None):
    """Expand all deviation profiles from a restriction

    Parameters
    ----------
    full_game : Game
        The game the deviations profiles will be valid for.
    rest : [bool]
        The restriction to get deviations from.
    red_players : [int], optional
        The number of players in each role in the reduced game.IF
        specified, it must match the expected number for twins reduction.
    role_index : int, optional
        If specified , only expand deviations for the role selected.
    """
    exp_red_players = np.minimum(full_game.num_role_players, 2)
    utils.check(
        red_players is None or np.all(exp_red_players == red_players),
        "twins reduction didn't get expected reduced players")
    return dpr.expand_deviation_profiles(
        full_game, rest, exp_red_players, role_index)
