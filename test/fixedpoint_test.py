import pytest
import numpy as np

from gameanalysis import fixedpoint
from gameanalysis import utils


def simple_fixed_point(dim, rate):
    target = utils.simplex_project(np.random.rand(dim))

    def func(mix):
        img = mix.copy()
        img += rate * (target - img)
        return img

    return target, func


@pytest.mark.parametrize('dim', [2, 3, 4, 7, 10])
@pytest.mark.parametrize('rate', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('tol', [1e-3, 1e-4, 1e-6])
@pytest.mark.parametrize('disc_mult', [0, 1, 2, 3])
def test_fixed_point(dim, rate, tol, disc_mult):
    start = utils.simplex_project(np.random.rand(dim))
    target, func = simple_fixed_point(dim, rate)
    print(target)
    res = fixedpoint.fixed_point(func, start, tol=tol,
                                 init_disc=dim * disc_mult)
    print()
    assert np.all(np.abs(res - target) <= tol)
