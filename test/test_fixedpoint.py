"""Test fixed point calculation"""
import pytest
import numpy as np

from gameanalysis import fixedpoint
from gameanalysis import utils


@pytest.mark.parametrize('dim', [2, 3, 4, 7, 10])
def test_ndim_fixed_point(dim):
    """Test that it computes a fixed point for arbitrary dimensional cycles"""
    start = utils.simplex_project(np.random.rand(dim))
    res = fixedpoint.fixed_point(lambda x: np.roll(x, 1), start, disc=100)
    assert np.all(res >= 0)
    assert np.isclose(res.sum(), 1)
    assert np.all(np.abs(res - 1 / dim) <= 0.01)


def progressive_fixed_point(func, start, init_disc, final_disc, ratio=2):
    """Progressive fixed point calculation"""
    while init_disc <= final_disc * ratio:
        start = fixedpoint.fixed_point(func, start, disc=init_disc)
        init_disc *= ratio
    return start


@pytest.mark.parametrize('dim', [2, 3, 4, 7, 10])
@pytest.mark.parametrize('disc', [10**3, 10**4, 10**6])
def test_ndim_progressive_fixed_point(dim, disc):
    """Test that it computes a fixed point for arbitrary dimensional cycles"""
    start = utils.simplex_project(np.random.rand(dim))
    res = progressive_fixed_point(lambda x: np.roll(x, 1), start, 16, disc)
    assert np.all(res >= 0)
    assert np.isclose(res.sum(), 1)
    assert np.all(np.abs(res - 1 / dim) <= 1 / disc)


@pytest.mark.slow
@pytest.mark.parametrize('tol', [1e-3, 1e-4, 1e-6])
def test_rps_fixed_point(tol):
    """Test that it computes a fixed point for bad shapley triangles"""
    # pytest: disable-msg=invalid-name
    start = utils.simplex_project(np.random.rand(3))
    weights = 1 + 3 * np.random.random(3)
    vara, varb, varc = weights
    expected = np.linalg.solve(
        [[0, -vara, 1, 1], [1, 0, -varb, 1], [-varc, 1, 0, 1], [1, 1, 1, 0]],
        [0, 0, 0, 1])[:-1]

    def func(inp):
        """Example fixed point function"""
        inter = np.roll(inp, 1) - weights * np.roll(inp, -1)
        res = np.maximum(0, inter - inter.dot(inp)) + inp
        return res / res.sum()

    res = fixedpoint.fixed_point(func, start, tol=tol)
    assert np.all(np.abs(res - expected) <= tol)
