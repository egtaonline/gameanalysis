"""Module for computing dominated strategies"""
import numpy as np

from gameanalysis import regret


def _dev_inds(num_strats):
    """Returns the deviation strategy indices for a deviation array"""
    sizes = num_strats.repeat(num_strats)
    offsets = np.insert(sizes.cumsum(), 0, 0)
    strat_offs = offsets[:-1].repeat(sizes)
    role_offs = np.insert(num_strats[:-1].cumsum(), 0, 0).repeat(
        num_strats ** 2)
    return np.arange(offsets[-1]) - strat_offs + role_offs


def _gains(game):
    """Returns the gains for deviating for every profile in the game

    Also returns the profile supports for indexing when the gains array should
    be zero because it's invalid versus having an actual zero gain."""
    sizes = np.repeat(game.num_strategies - 1, game.num_strategies)
    offsets = np.insert(sizes.cumsum(), 0, 0)
    size = offsets[-1]
    offsets = offsets[:-1]
    gains = np.zeros((game.num_profiles, size))
    supports = game.profiles > 0

    for i, (prof, support) in enumerate(zip(game.profiles, supports)):
        regs = regret.pure_strategy_deviation_gains(game, prof)
        reps = game.num_strategies[game.role_indices[support]] - 1
        reg_offsets = np.insert(reps[:-1].cumsum(), 0, 0)
        inds = (np.repeat(offsets[support] - reg_offsets, reps) +
                np.arange(regs.size))
        gains[i, inds] = regs

    return gains, supports


def _reduceat(ufunc, a, indices, axis=0, dtype=None, out=None):
    """Fix for the way reduceat handles empty slices"""
    if dtype is None:
        dtype = a.dtype
    if out is None:
        new_shape = list(a.shape)
        new_shape[axis] = indices.size
        out = np.empty(new_shape, dtype)
    out.fill(ufunc.identity)
    valid = np.diff(np.insert(indices, indices.size, a.shape[axis])) > 0
    index = [slice(None)] * out.ndim
    index[axis] = valid
    out[index] = ufunc.reduceat(a, indices[valid], axis)
    return out


def _weak_dominance(gains, supports, num_strats, conditional):
    """Returns the strategies that are weakly dominated"""
    sizes = np.repeat(num_strats - 1, num_strats)
    offsets = np.insert(sizes[:-1].cumsum(), 0, 0)
    with np.errstate(invalid='ignore'):  # nans
        dominated = (gains >= 0) & np.repeat(supports, sizes, -1)
    not_dominates = dominated | np.repeat(~supports, sizes, -1)
    if not conditional:
        not_dominates |= np.isnan(gains)
    return _reduceat(np.logical_or, dominated.any(0) & not_dominates.all(0),
                     offsets)


def _strict_dominance(gains, supports, num_strats, conditional):
    """Returns the strategies that are strictly dominated"""
    sizes = np.repeat(num_strats - 1, num_strats)
    offsets = np.insert(sizes[:-1].cumsum(), 0, 0)
    with np.errstate(invalid='ignore'):  # nans
        dominated = gains > 0
    not_dominates = dominated | np.repeat(~supports, sizes, -1)
    if not conditional:
        not_dominates |= np.isnan(gains)
    return _reduceat(np.logical_or, dominated.any(0) & not_dominates.all(0),
                     offsets)


def _never_best_response(gains, supports, num_strats, conditional):
    """Returns the strategies that are never a best response"""
    # This way we include self (e.g. 0) in best response
    self_sizes = np.repeat(num_strats, num_strats)
    self_offsets = np.insert(self_sizes[:-1].cumsum(), 0, 0)
    # The final insert indicies of self deviations. However, because the array
    # gets expended at each insertion, we have to subtract arange(num_inserts),
    # which is why it's missing from the final term
    self_inds = (self_offsets -  # + np.arange(num_strats.sum())
                 np.insert(num_strats[:-1].cumsum(), 0, 0).repeat(num_strats))
    self_gains = np.insert(gains, self_inds, 0, -1)

    # fmax ignores nans when possible
    best_gains = np.fmax.reduceat(self_gains, self_offsets, -1)\
        .repeat(self_sizes, -1)
    best_resps = (best_gains == self_gains) & supports.repeat(self_sizes, -1)
    if conditional:
        best_resps |= np.isnan(best_gains)
    is_br = best_resps.any(0)

    # Now we need to map deviations back onto strategies
    inds = _dev_inds(num_strats)
    return np.bincount(inds, is_br, num_strats.sum()) == 0


def weakly_dominated(game, conditional=True):
    """Return a mask of the strategies that are weakly dominated

    If conditional, then missing data will be treated as dominating."""
    gains, supports = _gains(game)
    return _weak_dominance(gains, supports, game.num_strategies, conditional)


def strictly_dominated(game, conditional=True):
    """Return a mask of the strategies that are strictly dominated

    If conditional, then missing data will be treated as dominating."""
    gains, supports = _gains(game)
    return _strict_dominance(gains, supports, game.num_strategies, conditional)


def never_best_response(game, conditional=True):
    """Return a mask of the strategies that are never a best response

    If conditional, then missing data is treated as a best response. The
    counted best response will be the largest deviation that has data."""
    gains, supports = _gains(game)
    return _never_best_response(gains, supports, game.num_strategies,
                                conditional)


_CRITERIA = {
    'weakdom': _weak_dominance,
    'strictdom': _strict_dominance,
    'neverbr': _never_best_response,
}


def iterated_elimination(game, criterion, conditional=True):
    """Return a subgame mask resulting from iterated elimination of strategies

    Parameters
    ----------
    game : Game
        The game to run iterated elimination on
    criterion : str, {'weakdom', 'strictdom', 'neverbr'}
        The criterion to use to eliminated strategies.
    conditional : bool
        Whether to use conditional criteria. In general, conditional set to
        true will assume that unobserved payoffs are large. See the other
        methods for a more detailed explanation
    """
    # There's a few recomputed things that could be passed to save computation
    # time, but they're minimal and probably not that important
    cfunc = _CRITERIA[criterion]

    num_strats = game.num_strategies
    gains, supports = _gains(game)

    subgame_mask = np.ones(game.num_role_strats, bool)
    mask = ~cfunc(gains, supports, num_strats, conditional)
    while (~np.all(mask) and np.any(np.add.reduceat(
            mask, np.insert(num_strats[:-1].cumsum(), 0, 0)) > 1)):
        subgame_mask[subgame_mask] = mask
        prof_mask = ~np.any(supports & ~mask, -1)
        dev_inds = _dev_inds(num_strats)
        strat_inds = np.arange(num_strats.sum()).repeat(
            num_strats.repeat(num_strats))
        dev_inds = dev_inds[dev_inds != strat_inds]
        dev_in_supp = np.in1d(dev_inds, mask.nonzero()[0])
        strat_in_supp = mask.repeat(np.repeat(num_strats - 1, num_strats))
        supports = supports[prof_mask][:, mask]
        gains = gains[prof_mask][:, dev_in_supp & strat_in_supp]
        num_strats = np.add.reduceat(
            mask, np.insert(num_strats[:-1].cumsum(), 0, 0))
        mask = ~cfunc(gains, supports, num_strats, conditional)

    subgame_mask[subgame_mask] = mask
    return subgame_mask
