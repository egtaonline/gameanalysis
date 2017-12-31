import functools
import warnings

import numpy as np


# FIXME Change all from test import and from gameanalysis import to from . import

games = [
    ([1], [1]),
    ([1], [2]),
    ([2], [1]),
    ([1, 1], [1, 1]),
    ([1, 1], [2, 2]),
    ([2, 2], [1, 1]),
    ([1, 2], [2, 1]),
    ([2, 1], [1, 2]),
    ([3, 4], [2, 3]),
    ([2, 3, 4], [4, 3, 2]),
]


def warnings_filter(status='ignore'):
    """decorator to apply warning filter to a function

    Arguments
    ---------
    status : str, optional
        The status to treat all warnings as. Default ignores all wanrings."""
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return func(*args, **kwargs)
        return wrapped
    return decorator


# FIXME Use this everywhere we don't have an actual ground truth for the
# jacobian.


def mixture_jacobian_estimate(game, function, mix, step=1e-6):
    """Estimate a jacobian from deviations
    
    Make small perturbations to a function of a mixture to estimate it's
    mixture jacobian.

    Parameters
    ----------
    game : RsGame
        Game the mixtures come from.
    function : (mixture) -> array
        Function of a mixture that returns an array.
    mix : mixture
        The point to evaluate the jacobian at.
    step : float, optional
        The amount to perturb the mixture. Smaller values will tend to produce
        better estimates of the jacobian but run a higher risk of encountering
        numeric precision issues.
    """
    default = function(mix)
    zero = np.zeros(game.num_strats)

    num_base = game.num_role_strats - 1
    offsets = np.zeros((game.num_strats,) * 2)
    results = np.empty(offsets.shape)
    for off, res, start, num in zip(
            offsets, results, game.role_starts.repeat(num_base),
            game.num_role_strats.repeat(num_base)):
        offs = off[start:start + num]
        shift = zero
        while not game.is_mixture(shift):
            np.copyto(offs, np.random.normal(0, step, num))
            offs -= offs.sum() / num
            shift = mix + off
        np.copyto(res, function(mix + off) - default)
    num_ind = num_base.sum()
    offsets[num_ind:] = np.eye(game.num_roles).repeat(game.num_role_strats, 1)
    results[num_ind:].fill(0)
    return np.linalg.solve(offsets, results).T
