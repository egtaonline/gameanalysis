"""Module for computing dominated strategies"""
import numpy as np

from gameanalysis import regret
from gameanalysis import rsgame


def _gains(game):
    """Returns the gains for deviating for every profile in the game

    Also returns the profile supports for indexing when the gains array should
    be zero because it's invalid versus having an actual zero gain."""
    return np.stack([
        regret.pure_strategy_deviation_gains(game, prof)
        for prof in game.profiles()])


def _weak_dominance(game, gains, supports, conditional):
    """Returns the strategies that are weakly dominated"""
    with np.errstate(invalid='ignore'):  # nans
        dominated = (gains >= 0) & supports.repeat(game.num_strat_devs, -1)
    not_dominates = dominated | np.repeat(~supports, game.num_strat_devs, -1)
    if not conditional:
        not_dominates |= np.isnan(gains)
    non_self = game.dev_from_indices != game.dev_to_indices
    return np.logical_or.reduceat(
        dominated.any(0) & not_dominates.all(0) & non_self,
        game.dev_strat_starts)


def weakly_dominated(game, *, conditional=True):
    """Return a mask of the strategies that are weakly dominated

    If conditional, then missing data will be treated as dominating."""
    return _weak_dominance(
        game, _gains(game), game.profiles() > 0, conditional)


def _strict_dominance(game, gains, supports, conditional):
    """Returns the strategies that are strictly dominated"""
    with np.errstate(invalid='ignore'):  # nans
        dominated = gains > 0
    not_dominates = dominated | np.repeat(~supports, game.num_strat_devs, -1)
    if not conditional:
        not_dominates |= np.isnan(gains)
    return np.logical_or.reduceat(
        dominated.any(0) & not_dominates.all(0), game.dev_strat_starts)


def strictly_dominated(game, *, conditional=True):
    """Return a mask of the strategies that are strictly dominated

    If conditional, then missing data will be treated as dominating."""
    return _strict_dominance(
        game, _gains(game), game.profiles() > 0, conditional)


def _never_best_response(game, gains, supports, conditional):
    """Returns the strategies that are never a best response"""
    best_gains = np.fmax.reduceat(gains, game.dev_strat_starts, 1).repeat(
        game.num_strat_devs, 1)
    best_resps = (best_gains == gains) & supports.repeat(
        game.num_strat_devs, 1)
    if conditional:
        best_resps |= np.isnan(best_gains)
    is_br = best_resps.any(0)
    return np.bincount(game.dev_to_indices, is_br, game.num_strats) == 0


def never_best_response(game, *, conditional=True):
    """Return a mask of the strategies that are never a best response

    If conditional, then missing data is treated as a best response. The
    counted best response will be the largest deviation that has data."""
    return _never_best_response(
        game, _gains(game), game.profiles() > 0, conditional)


_CRITERIA = {
    'weakdom': _weak_dominance,
    'strictdom': _strict_dominance,
    'neverbr': _never_best_response,
}


def iterated_elimination(game, criterion, *, conditional=True):
    """Return a restriction resulting from iterated elimination of strategies

    Parameters
    ----------
    game : Game
        The game to run iterated elimination on
    criterion : {'weakdom', 'strictdom', 'neverbr'}
        The criterion to use to eliminated strategies.
    conditional : bool
        Whether to use conditional criteria. In general, conditional set to
        true will assume that unobserved payoffs are large. See the other
        methods for a more detailed explanation
    """
    # There's a few recomputed things that could be passed to save computation
    # time, but they're minimal and probably not that important
    cfunc = _CRITERIA[criterion]

    egame = rsgame.empty_copy(game)
    gains = _gains(game)
    supports = game.profiles() > 0

    rest = np.ones(game.num_strats, bool)
    mask = ~cfunc(egame, gains, supports, conditional)
    while (not np.all(mask) and np.any(np.add.reduceat(
            mask, egame.role_starts) > 1)):
        rest[rest] = mask
        prof_mask = ~np.any(supports & ~mask, -1)
        to_in_mask = mask[egame.dev_to_indices]
        from_in_mask = mask[egame.dev_from_indices]

        egame = egame.restrict(mask)
        gains = gains[prof_mask][:, to_in_mask & from_in_mask]
        supports = supports[prof_mask][:, mask]
        mask = ~cfunc(egame, gains, supports, conditional)

    rest[rest] = mask
    return rest
