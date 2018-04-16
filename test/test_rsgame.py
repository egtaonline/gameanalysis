"""Test rsgame"""
# pylint: disable=too-many-lines
import json
import warnings

import numpy as np
import numpy.random as rand
import pytest
import scipy.special as sps

from gameanalysis import rsgame
from gameanalysis import utils
from test import utils as testutils # pylint: disable=wrong-import-order


TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps


# ----------
# StratArray
# ----------


def stratarray(num_strats):
    """Create a strat array"""
    return rsgame.empty(np.ones_like(num_strats), num_strats)


def test_stratarray_properties(): # pylint: disable=too-many-statements
    """Test properties"""
    sarr = stratarray([1])
    assert sarr.num_strats == 1
    assert np.all(sarr.num_role_strats == [1])
    assert sarr.num_roles == 1
    assert np.all(sarr.role_starts == [0])
    assert np.all(sarr.role_indices == [0])
    assert sarr.num_all_restrictions == 1
    assert sarr.num_pure_restrictions == 1
    assert np.all(sarr.num_strat_devs == [0])
    assert np.all(sarr.num_role_devs == [0])
    assert sarr.num_devs == 0
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_strat_starts == [0])
        assert len(warns) == 1
        assert issubclass(warns[0].category, UserWarning)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_role_starts == [0])
        assert len(warns) == 1
        assert issubclass(warns[0].category, UserWarning)
    assert np.all(sarr.dev_from_indices == [])
    assert np.all(sarr.dev_to_indices == [])

    sarr = stratarray([3])
    assert sarr.num_strats == 3
    assert np.all(sarr.num_role_strats == [3])
    assert sarr.num_roles == 1
    assert np.all(sarr.role_starts == [0])
    assert np.all(sarr.role_indices == [0, 0, 0])
    assert sarr.num_all_restrictions == 7
    assert sarr.num_pure_restrictions == 3
    assert np.all(sarr.num_strat_devs == [2, 2, 2])
    assert np.all(sarr.num_role_devs == [6])
    assert sarr.num_devs == 6
    assert np.all(sarr.dev_strat_starts == [0, 2, 4])
    assert np.all(sarr.dev_role_starts == [0])
    assert np.all(sarr.dev_from_indices == [0, 0, 1, 1, 2, 2])
    assert np.all(sarr.dev_to_indices == [1, 2, 0, 2, 0, 1])

    sarr = stratarray([1, 3])
    assert sarr.num_strats == 4
    assert np.all(sarr.num_role_strats == [1, 3])
    assert sarr.num_roles == 2
    assert np.all(sarr.role_starts == [0, 1])
    assert np.all(sarr.role_indices == [0, 1, 1, 1])
    assert sarr.num_all_restrictions == 7
    assert sarr.num_pure_restrictions == 3
    assert np.all(sarr.num_strat_devs == [0, 2, 2, 2])
    assert np.all(sarr.num_role_devs == [0, 6])
    assert sarr.num_devs == 6
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_strat_starts == [0, 0, 2, 4])
        assert len(warns) == 1
        assert issubclass(warns[0].category, UserWarning)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_role_starts == [0, 0])
        assert len(warns) == 1
        assert issubclass(warns[0].category, UserWarning)
    assert np.all(sarr.dev_from_indices == [1, 1, 2, 2, 3, 3])
    assert np.all(sarr.dev_to_indices == [2, 3, 1, 3, 1, 2])

    sarr = stratarray([3, 2, 1])
    assert sarr.num_strats == 6
    assert np.all(sarr.num_role_strats == [3, 2, 1])
    assert sarr.num_roles == 3
    assert np.all(sarr.role_starts == [0, 3, 5])
    assert np.all(sarr.role_indices == [0, 0, 0, 1, 1, 2])
    assert sarr.num_all_restrictions == 21
    assert sarr.num_pure_restrictions == 6
    assert np.all(sarr.num_strat_devs == [2, 2, 2, 1, 1, 0])
    assert np.all(sarr.num_role_devs == [6, 2, 0])
    assert sarr.num_devs == 8
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_strat_starts == [0, 2, 4, 6, 7, 8])
        assert len(warns) == 1
        assert issubclass(warns[0].category, UserWarning)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_role_starts == [0, 6, 8])
        assert len(warns) == 1
        assert issubclass(warns[0].category, UserWarning)
    assert np.all(sarr.dev_from_indices == [0, 0, 1, 1, 2, 2, 3, 4])
    assert np.all(sarr.dev_to_indices == [1, 2, 0, 2, 0, 1, 4, 3])


def test_restriction_enumeration():
    """Test enumerate restrictions"""
    sarr = stratarray([1])
    all_restrictions = [[True]]
    assert not np.setxor1d(utils.axis_to_elem(all_restrictions),
                           utils.axis_to_elem(sarr.all_restrictions())).size
    pure_restrictions = [[True]]
    assert not np.setxor1d(utils.axis_to_elem(pure_restrictions),
                           utils.axis_to_elem(sarr.pure_restrictions())).size

    sarr = stratarray([3])
    all_restrictions = [[True, False, False],
                        [False, True, False],
                        [True, True, False],
                        [False, False, True],
                        [True, False, True],
                        [False, True, True],
                        [True, True, True]]
    assert not np.setxor1d(utils.axis_to_elem(all_restrictions),
                           utils.axis_to_elem(sarr.all_restrictions())).size
    pure_restrictions = [[True, False, False],
                         [False, True, False],
                         [False, False, True]]
    assert not np.setxor1d(utils.axis_to_elem(pure_restrictions),
                           utils.axis_to_elem(sarr.pure_restrictions())).size

    sarr = stratarray([1, 3])
    all_restrictions = [[True, True, False, False],
                        [True, False, True, False],
                        [True, True, True, False],
                        [True, False, False, True],
                        [True, True, False, True],
                        [True, False, True, True],
                        [True, True, True, True]]
    assert not np.setxor1d(utils.axis_to_elem(all_restrictions),
                           utils.axis_to_elem(sarr.all_restrictions())).size
    pure_restrictions = [[True, True, False, False],
                         [True, False, True, False],
                         [True, False, False, True]]
    assert not np.setxor1d(utils.axis_to_elem(pure_restrictions),
                           utils.axis_to_elem(sarr.pure_restrictions())).size


def test_is_restriction():
    """Test is restriction"""
    sarr = stratarray([3, 2])
    assert sarr.is_restriction([True, False, True, False, True])
    assert not sarr.is_restriction([True, False, True, False, False])
    assert not sarr.is_restriction([False, False, False, True, False])
    assert not sarr.is_restriction([False, False, False, False, False])
    assert np.all([True] + [False] * 3 == sarr.is_restriction([
        [True, False, True, False, True],
        [True, False, True, False, False],
        [False, False, False, True, False],
        [False, False, False, False, False]]))
    assert sarr.is_restriction(
        [[True], [False], [True], [False], [True]], axis=0)

    with pytest.raises(ValueError):
        sarr.is_restriction([False, False, False, False])
    with pytest.raises(ValueError):
        sarr.is_restriction([False, False, False, False, False, False])
    with pytest.raises(ValueError):
        sarr.is_restriction([[False, False, False, False, False, False]])


def test_is_pure_restriction():
    """Test pure restrictions"""
    sarr = stratarray([3, 2])
    assert sarr.is_pure_restriction([True, False, False, False, True])
    assert not sarr.is_pure_restriction([True, False, True, False, False])
    assert not sarr.is_pure_restriction([False, True, True, True, False])
    assert not sarr.is_pure_restriction([False, False, True, True, True])

    with pytest.raises(ValueError):
        sarr.is_pure_restriction([False, False, False, False])


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_stratarray_restrictions(_, role_strats):
    """Test random restrictions"""
    sarr = stratarray(role_strats)
    all_restrictions = sarr.all_restrictions()
    assert sarr.is_restriction(all_restrictions).all()
    assert sarr.num_all_restrictions == all_restrictions.shape[0]
    pure_restrictions = sarr.pure_restrictions()
    assert sarr.is_restriction(pure_restrictions).all()
    assert sarr.num_pure_restrictions == pure_restrictions.shape[0]


def test_random_restrictions():
    """Test random restrictions"""
    # Technically some of these can fail, but it's extremely unlikely
    sarr = stratarray([3])
    rests = sarr.random_restrictions(1000)
    assert sarr.is_restriction(rests).all()
    assert not rests.all()

    rests = sarr.random_restrictions(1000, strat_prob=1)
    assert sarr.is_restriction(rests).all()
    assert rests.all()

    # Probability is raised to 1/3
    rests = sarr.random_restrictions(1000, strat_prob=0)
    assert sarr.is_restriction(rests).all()

    rests = sarr.random_restrictions(1000, strat_prob=0, normalize=False)
    assert sarr.is_restriction(rests).all()

    # This has a roughly .5^1000 probability of failure
    sarr = stratarray([3, 2])
    rests = sarr.random_restrictions(1000, strat_prob=[1, 1 / 2])
    assert sarr.is_restriction(rests).all()
    assert np.all([True, True, True, False, False] == rests.all(0))


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_random_restrictions(_, role_strats):
    """Test random random restrictions"""
    sarr = stratarray(role_strats)
    rest = sarr.random_restriction()
    assert len(rest.shape) == 1

    rests = sarr.random_restrictions(100)
    assert sarr.is_restriction(rests).all()


def test_trim_mixture_support():
    """Test trim mixture support"""
    sarr = stratarray([3])
    mix = np.array([0.7, 0.3, 0])
    not_trimmed = sarr.trim_mixture_support(mix, thresh=0.1)
    assert np.allclose(mix, not_trimmed)
    trimmed = sarr.trim_mixture_support(mix, thresh=0.4)
    assert np.allclose([1, 0, 0], trimmed)

    trimmed = sarr.trim_mixture_support(mix[:, None], thresh=0.4, axis=0)[:, 0]
    assert np.allclose([1, 0, 0], trimmed)


def test_is_mixture():
    """Test is mixture"""
    sarr = stratarray([3, 2])
    assert sarr.is_mixture([0.2, 0.3, 0.5, 0.6, 0.4])
    assert not sarr.is_mixture([0.2, 0.3, 0.4, 0.5, 0.6])
    assert not sarr.is_mixture([0.2, 0.3, 0.4, 0.5, 0.4])
    assert not sarr.is_mixture([0.2, 0.35, 0.4, 0.4, 0.6])
    assert not sarr.is_mixture([0.2, 0.25, 0.4, 0.4, 0.6])
    assert np.all(([True] + [False] * 4) == sarr.is_mixture([
        [0.2, 0.3, 0.5, 0.6, 0.4],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.4],
        [0.2, 0.35, 0.4, 0.4, 0.6],
        [0.2, 0.25, 0.4, 0.4, 0.6]]))
    assert sarr.is_mixture([[0.2], [0.3], [0.5], [0.6], [0.4]], axis=0)

    with pytest.raises(ValueError):
        sarr.is_mixture([0, 0, 0, 0])
    with pytest.raises(ValueError):
        sarr.is_mixture([0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        sarr.is_mixture([[0, 0, 0, 0, 0, 0]])


def test_is_pure_mixture():
    """Test is pure mixture"""
    sarr = stratarray([3, 2])
    assert sarr.is_pure_mixture([0, 1, 0, 0, 1])
    assert not sarr.is_pure_mixture([0.2, 0.3, 0.4, 0.5, 0.6])
    assert not sarr.is_pure_mixture([1, 0, 0, 0.5, 0.5])
    assert not sarr.is_pure_mixture([0.2, 0.8, 0, 1, 0])
    assert not sarr.is_pure_mixture([0.2, 0.4, 0.4, 0.4, 0.6])

    with pytest.raises(ValueError):
        sarr.is_pure_mixture([0, 0, 0, 0])


def test_mixture_project():
    """Test mixture project"""
    sarr = stratarray([1])
    mixtures = [[0],
                [1],
                [2],
                [-1]]
    expected = [[1]] * 4
    assert np.allclose(expected, sarr.mixture_project(mixtures))

    sarr = stratarray([3])
    mixtures = [[0, 0, 0],
                [1, 0, 0],
                [2, 1, 0],
                [1.2, 1.3, 1.5]]
    expected = [[1 / 3, 1 / 3, 1 / 3],
                [1, 0, 0],
                [1, 0, 0],
                [0.2, 0.3, 0.5]]
    assert np.allclose(expected, sarr.mixture_project(mixtures))

    sarr = stratarray([1, 3])
    mixtures = [[0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 2, 1, 0],
                [-1, 1.2, 1.3, 1.5]]
    expected = [[1, 1 / 3, 1 / 3, 1 / 3],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0.2, 0.3, 0.5]]
    assert np.allclose(expected, sarr.mixture_project(mixtures))

    sarr = stratarray([3, 2, 1])
    mixtures = [[0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [2, 1, 0, 1, 2, 2],
                [1.2, 1.3, 1.5, 0.4, 0.2, 0.3]]
    expected = [[1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 1],
                [0.2, 0.3, 0.5, 0.6, 0.4, 1]]
    assert np.allclose(expected, sarr.mixture_project(mixtures))


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_mixture_project(_, role_strats):
    """Test random mixture project"""
    sarr = stratarray(role_strats)
    for non_mixture in rand.uniform(-1, 1, (100, sarr.num_strats)):
        new_mix = sarr.mixture_project(non_mixture)
        assert sarr.is_mixture(new_mix), \
            'simplex project did not create a valid mixture'

    mixes = rand.uniform(-1, 1, (10, 12, sarr.num_strats))
    simps = sarr.mixture_project(mixes)
    assert simps.shape[:2] == (10, 12)
    assert sarr.is_mixture(simps).all()


def test_to_from_simplex():
    """Test to from simplex"""
    sarr = stratarray([2, 2])
    mixture = [1 / 5, 4 / 5, 1 / 5, 4 / 5]
    simplex = [2 / 15, 2 / 15, 11 / 15]
    assert np.allclose(simplex, sarr.mixture_to_simplex(mixture))
    assert np.allclose(mixture, sarr.mixture_from_simplex(simplex))

    mixture = [3 / 5, 2 / 5, 1 / 2, 1 / 2]
    simplex = [2 / 5, 1 / 3, 4 / 15]
    assert np.allclose(simplex, sarr.mixture_to_simplex(mixture))
    assert np.allclose(mixture, sarr.mixture_from_simplex(simplex))

    mixture = [3 / 4, 1 / 4, 1 / 4, 3 / 4]
    simplex = [1 / 2, 1 / 6, 1 / 3]
    assert np.allclose(simplex, sarr.mixture_to_simplex(mixture))
    assert np.allclose(mixture, sarr.mixture_from_simplex(simplex))

    mixture = [1, 0, 1 / 4, 3 / 4]
    simplex = [1, 0, 0]
    assert np.allclose(simplex, sarr.mixture_to_simplex(mixture))
    assert np.allclose(mixture, sarr.mixture_from_simplex(simplex))


@pytest.mark.parametrize('strats', [1, 2, 4])
def test_random_one_role_to_from_simplex(strats):
    """Test random from simplex"""
    sarr = stratarray([strats])
    inits = sarr.random_mixtures(100)
    simplicies = sarr.mixture_to_simplex(inits)
    assert np.allclose(inits, simplicies)
    mixtures = sarr.mixture_to_simplex(inits)
    assert np.allclose(inits, mixtures)


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_uniform_simplex_homotopy(_, role_strats):
    """Test homotopy"""
    sarr = stratarray(role_strats)
    uniform = sarr.uniform_mixture()
    simp = sarr.mixture_to_simplex(uniform)
    assert np.allclose(simp[0], simp[1:])
    assert np.allclose(uniform, sarr.mixture_from_simplex(simp))


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_simplex_homotopy(_, role_strats):
    """Test random homotopy"""
    sarr = stratarray(role_strats)
    mixes = sarr.random_mixtures(100)

    simp = sarr.mixture_to_simplex(mixes[0])
    assert np.all(simp >= 0)
    assert np.isclose(simp.sum(), 1)
    assert np.allclose(mixes[0], sarr.mixture_from_simplex(simp))

    simps = sarr.mixture_to_simplex(mixes)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(-1), 1)
    assert np.allclose(mixes, sarr.mixture_from_simplex(simps))

    mixes = mixes.reshape((4, 25, -1))
    simps = sarr.mixture_to_simplex(mixes)
    assert simps.shape[:2] == (4, 25)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(-1), 1)
    assert np.allclose(mixes, sarr.mixture_from_simplex(simps))


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_uniform_simplex_homotopy(_, role_strats):
    """Test uniform homotopy"""
    sarr = stratarray(role_strats)
    rand_mixes = sarr.random_mixtures(100)
    mask = np.repeat(rand.random((100, sarr.num_roles))
                     < 0.5, sarr.num_role_strats, 1)
    mixes = np.where(mask, rand_mixes, sarr.uniform_mixture())

    simp = sarr.mixture_to_simplex(mixes[0])
    assert np.all(simp >= 0)
    assert np.isclose(simp.sum(), 1)
    assert np.allclose(mixes[0], sarr.mixture_from_simplex(simp))

    simps = sarr.mixture_to_simplex(mixes)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(-1), 1)
    assert np.allclose(mixes, sarr.mixture_from_simplex(simps))

    mixes = mixes.reshape((4, 25, -1))
    simps = sarr.mixture_to_simplex(mixes)
    assert simps.shape[:2] == (4, 25)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(-1), 1)
    assert np.allclose(mixes, sarr.mixture_from_simplex(simps))


def test_uniform_mixture():
    """Test uniform mixture"""
    sarr = stratarray([1])
    assert np.allclose([1], sarr.uniform_mixture())

    sarr = stratarray([3])
    assert np.allclose([1 / 3] * 3, sarr.uniform_mixture())

    sarr = stratarray([1, 3])
    assert np.allclose([1] + [1 / 3] * 3, sarr.uniform_mixture())

    sarr = stratarray([3, 2, 1])
    assert np.allclose([1 / 3] * 3 + [1 / 2] * 2 + [1], sarr.uniform_mixture())


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_mixtures(_, role_strats):
    """Test random mixtures"""
    sarr = stratarray(role_strats)
    mix = sarr.random_mixture()
    assert len(mix.shape) == 1

    rand_mixes = sarr.random_mixtures(100)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_mixtures(100, alpha=0.1)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_mixtures(100, alpha=2)
    assert sarr.is_mixture(rand_mixes).all()


def test_random_sparse_mixtures():
    """Test sparse mixtures"""
    # Technically some of these can fail, but it's extremely unlikely
    sarr = stratarray([3])
    mixes = sarr.random_sparse_mixtures(1000)
    assert sarr.is_mixture(mixes).all()
    assert not np.all(mixes > 0)

    mixes = sarr.random_sparse_mixtures(1000, support_prob=1)
    assert sarr.is_mixture(mixes).all()
    assert np.all(mixes > 0)

    # Probability is raised to 1/3
    mixes = sarr.random_sparse_mixtures(1000, support_prob=0)
    assert sarr.is_mixture(mixes).all()

    mixes = sarr.random_sparse_mixtures(1000, support_prob=0, normalize=False)
    assert sarr.is_mixture(mixes).all()

    sarr = stratarray([3, 2])
    mixes = sarr.random_sparse_mixtures(1000, support_prob=[1, 1 / 2])
    assert sarr.is_mixture(mixes).all()
    assert np.all([True, True, True, False, False] == np.all(mixes > 0, 0))


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_random_sparse_mixtures(_, role_strats):
    """Test random sparse mixtures"""
    sarr = stratarray(role_strats)
    mix = sarr.random_sparse_mixture()
    assert len(mix.shape) == 1

    rand_mixes = sarr.random_sparse_mixtures(100)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_sparse_mixtures(100, alpha=0.1)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_mixtures(100, alpha=2)
    assert sarr.is_mixture(rand_mixes).all()


def test_biased_mixtures():
    """Test biased mixtures"""
    sarr = stratarray([1])
    expected = [[1]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3])
    expected = [[0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([1, 3])
    expected = [[1, 0.8, 0.1, 0.1],
                [1, 0.1, 0.8, 0.1],
                [1, 0.1, 0.1, 0.8]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3, 2, 1])
    expected = [[0.8, 0.1, 0.1, 0.8, 0.2, 1],
                [0.8, 0.1, 0.1, 0.2, 0.8, 1],
                [0.1, 0.8, 0.1, 0.8, 0.2, 1],
                [0.1, 0.8, 0.1, 0.2, 0.8, 1],
                [0.1, 0.1, 0.8, 0.8, 0.2, 1],
                [0.1, 0.1, 0.8, 0.2, 0.8, 1]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


def test_role_biased_mixtures():
    """Test random biased mixtures"""
    sarr = stratarray([1])
    expected = [[1]]
    actual = sarr.role_biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3])
    expected = [[0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]]
    actual = sarr.role_biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([1, 3])
    expected = [[1, 0.8, 0.1, 0.1],
                [1, 0.1, 0.8, 0.1],
                [1, 0.1, 0.1, 0.8]]
    actual = sarr.role_biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3, 2, 1])
    expected = [[0.8, 0.1, 0.1, 0.5, 0.5, 1],
                [0.1, 0.8, 0.1, 0.5, 0.5, 1],
                [0.1, 0.1, 0.8, 0.5, 0.5, 1],
                [1 / 3, 1 / 3, 1 / 3, 0.8, 0.2, 1],
                [1 / 3, 1 / 3, 1 / 3, 0.2, 0.8, 1]]
    actual = sarr.role_biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


@pytest.mark.parametrize('strats', [1, 2, 4])
@pytest.mark.parametrize('bias', [0.0, 0.2, 0.5, 0.8, 1.0])
def test_random_onerole_biased_equivelance(strats, bias):
    """Test single role biased mixtures"""
    sarr = stratarray([strats])
    amix = sarr.biased_mixtures(bias)
    bmix = sarr.role_biased_mixtures(bias)
    assert amix.shape == bmix.shape
    assert np.isclose(amix, bmix[:, None]).all(2).any(0).all()


def test_pure_mixtures():
    """Test pure mixtures"""
    sarr = stratarray([1])
    expected = [[1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3])
    expected = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([1, 3])
    expected = [[1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3, 2, 1])
    expected = [[1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 0, 1, 0, 1, 1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


def test_grid_mixtures_error():
    """Test grod mixtures"""
    sarr = stratarray([1])
    with pytest.raises(ValueError):
        sarr.grid_mixtures(1)


def test_grid_mixtures():
    """Test grid mixtures"""
    sarr = stratarray([1])
    expected = [[1]]
    actual = sarr.grid_mixtures(2)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()
    actual = sarr.grid_mixtures(3)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()
    actual = sarr.grid_mixtures(4)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3])
    expected = [[0, 0, 1],
                [0, 1 / 2, 1 / 2],
                [0, 1, 0],
                [1 / 2, 0, 1 / 2],
                [1 / 2, 1 / 2, 0],
                [1, 0, 0]]
    actual = sarr.grid_mixtures(3)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    expected = [[0, 0, 1],
                [0, 1 / 3, 2 / 3],
                [0, 2 / 3, 1 / 3],
                [0, 1, 0],
                [1 / 3, 0, 2 / 3],
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 2 / 3, 0],
                [2 / 3, 0, 1 / 3],
                [2 / 3, 1 / 3, 0],
                [1, 0, 0]]
    actual = sarr.grid_mixtures(4)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([1, 3])
    expected = [[1, 0, 0, 1],
                [1, 0, 1 / 2, 1 / 2],
                [1, 0, 1, 0],
                [1, 1 / 2, 0, 1 / 2],
                [1, 1 / 2, 1 / 2, 0],
                [1, 1, 0, 0]]
    actual = sarr.grid_mixtures(3)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = stratarray([3, 2, 1])
    expected = [[0, 0, 1, 0, 1, 1],
                [0, 0, 1, 1 / 2, 1 / 2, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 1 / 2, 1 / 2, 0, 1, 1],
                [0, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1],
                [0, 1 / 2, 1 / 2, 1, 0, 1],
                [0, 1, 0, 0, 1, 1],
                [0, 1, 0, 1 / 2, 1 / 2, 1],
                [0, 1, 0, 1, 0, 1],
                [1 / 2, 0, 1 / 2, 0, 1, 1],
                [1 / 2, 0, 1 / 2, 1 / 2, 1 / 2, 1],
                [1 / 2, 0, 1 / 2, 1, 0, 1],
                [1 / 2, 1 / 2, 0, 0, 1, 1],
                [1 / 2, 1 / 2, 0, 1 / 2, 1 / 2, 1],
                [1 / 2, 1 / 2, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 1 / 2, 1 / 2, 1],
                [1, 0, 0, 1, 0, 1]]
    actual = sarr.grid_mixtures(3)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_grid_pure_equivelance(_, role_strats):
    """Test random grid close"""
    sarr = stratarray(role_strats)
    expected = sarr.pure_mixtures()
    actual = sarr.grid_mixtures(2)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


@pytest.mark.parametrize('_,role_strats', testutils.GAMES)
def test_random_fixed_mixtures(_, role_strats):
    """Test random fixed mixtures"""
    sarr = stratarray(role_strats)
    assert sarr.is_mixture(sarr.biased_mixtures()).all()
    assert sarr.is_mixture(sarr.role_biased_mixtures()).all()
    assert sarr.is_mixture(sarr.pure_mixtures()).all()
    assert sarr.is_mixture(sarr.grid_mixtures(3)).all()
    assert sarr.is_mixture(sarr.grid_mixtures(4)).all()


def test_strat_name():
    """Test strat names"""
    sarr = stratarray([3, 2])
    for i, strat in enumerate(['s0', 's1', 's2', 's3', 's4']):
        assert strat == sarr.strat_name(i)


def test_indices():
    """Test indices"""
    sarr = stratarray([3, 2])
    assert sarr.role_index('r0') == 0
    assert sarr.role_index('r1') == 1
    assert sarr.role_strat_index('r0', 's0') == 0
    assert sarr.role_strat_index('r0', 's1') == 1
    assert sarr.role_strat_index('r0', 's2') == 2
    assert sarr.role_strat_index('r1', 's3') == 3
    assert sarr.role_strat_index('r1', 's4') == 4
    assert sarr.role_strat_dev_index('r0', 's0', 's1') == 0
    assert sarr.role_strat_dev_index('r0', 's0', 's2') == 1
    assert sarr.role_strat_dev_index('r0', 's1', 's0') == 2
    assert sarr.role_strat_dev_index('r0', 's1', 's2') == 3
    assert sarr.role_strat_dev_index('r0', 's2', 's0') == 4
    assert sarr.role_strat_dev_index('r0', 's2', 's1') == 5
    assert sarr.role_strat_dev_index('r1', 's3', 's4') == 6
    assert sarr.role_strat_dev_index('r1', 's4', 's3') == 7
    rs_names = (
        ('r0', 's0'), ('r0', 's1'), ('r0', 's2'), ('r1', 's3'), ('r1', 's4'))
    assert rs_names == sarr.role_strat_names


def test_to_mixture_from_json():
    """Test mixture to from json"""
    sarr = stratarray([2, 1])
    mix = [.6, .4, 1]
    json_mix = {'r0': {'s1': .4, 's0': .6}, 'r1': {'s2': 1}}
    assert sarr.mixture_to_json(mix) == json_mix
    new_mix = sarr.mixture_from_json(json_mix)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float
    new_mix.fill(0)
    sarr.mixture_from_json(json_mix, dest=new_mix)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_mixture_serialization(role_players, role_strats):
    """Test mixture serialization"""
    game = rsgame.empty(role_players, role_strats)
    mixes = game.random_mixtures(20)
    copies = np.empty(mixes.shape)
    for mix, copy in zip(mixes, copies):
        jmix = json.dumps(game.mixture_to_json(mix))
        game.mixture_from_json(json.loads(jmix), copy)
    assert np.allclose(copies, mixes)


def test_to_from_mix_repr():
    """Test to from repr"""
    sarr = stratarray([2, 1])
    mix = [.6, .4, 1]
    expected = 'r0: 60.00% s0, 40.00% s1; r1: 100.00% s2'
    assert sarr.mixture_to_repr(mix) == expected
    new_mix = sarr.mixture_from_repr(expected)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float
    new_mix.fill(0)
    sarr.mixture_from_repr(expected, dest=new_mix)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float


def test_to_from_mix_str():
    """test to from str"""
    sarr = stratarray([2, 1])
    mix = [0.3, 0.7, 1]
    expected = """
r0:
    s0:  30.00%
    s1:  70.00%
r1:
    s2: 100.00%
"""[1:-1]
    assert sarr.mixture_to_str(mix) == expected
    new_mix = sarr.mixture_from_str(expected)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float
    new_mix.fill(0)
    sarr.mixture_from_str(expected, dest=new_mix)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float


def test_to_from_restriction_json():
    """Test to from restriction json"""
    sarr = stratarray([2, 1])
    sub = [True, False, True]
    json_sub = {'r0': ['s0'], 'r1': ['s2']}
    assert sarr.restriction_to_json(sub) == json_sub
    new_sub = sarr.restriction_from_json(json_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool
    new_sub.fill(False)
    sarr.restriction_from_json(json_sub, dest=new_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_restriction_serialization(role_players, role_strats):
    """Test random restriction serialization"""
    game = rsgame.empty(role_players, role_strats)
    subs = game.random_restrictions(20)
    copies = np.empty(subs.shape, bool)
    for sub, copy in zip(subs, copies):
        jsub = json.dumps(game.restriction_to_json(sub))
        game.restriction_from_json(json.loads(jsub), copy)
    assert np.all(copies == subs)


def test_to_from_restriction_repr():
    """Test random restriction repr"""
    sarr = stratarray([2, 1])
    sub = [True, False, True]
    sub_repr = 'r0: s0; r1: s2'
    assert sarr.restriction_to_repr(sub) == sub_repr
    new_sub = sarr.restriction_from_repr(sub_repr)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool
    new_sub.fill(False)
    sarr.restriction_from_repr(sub_repr, dest=new_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool


def test_to_from_restriction_str():
    """Test to from restriction string"""
    sarr = stratarray([2, 1])
    sub = [True, False, True]
    sub_str = """
r0:
    s0
r1:
    s2
"""[1:-1]
    assert sarr.restriction_to_str(sub) == sub_str
    new_sub = sarr.restriction_from_str(sub_str)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool
    new_sub.fill(False)
    sarr.restriction_from_str(sub_str, dest=new_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool


def test_to_from_role_json():
    """Test to from role json"""
    sarr = stratarray([2, 1])
    role = [6, 3]
    json_role = {'r0': 6, 'r1': 3}
    assert sarr.role_to_json(role) == json_role
    arr = sarr.role_from_json(json_role)
    assert np.allclose(arr, role)
    assert arr.dtype == float
    arr = np.empty_like(arr)
    sarr.role_from_json(json_role, dest=arr)
    assert np.allclose(arr, role)
    assert arr.dtype == float


def test_to_from_role_repr():
    """Test to from role repr"""
    sarr = stratarray([2, 1])
    role = [6, 3]
    rep_role = 'r0: 6; r1: 3'
    assert sarr.role_to_repr(role) == rep_role
    arr = sarr.role_from_repr(rep_role)
    assert np.allclose(arr, role)
    assert arr.dtype == float
    arr = np.empty_like(arr)
    sarr.role_from_repr(rep_role, dest=arr)
    assert np.allclose(arr, role)
    assert arr.dtype == float
    assert np.allclose(sarr.role_from_repr('r0:6,r1:3'), role)


def test_trim_precision():
    """Test trim precision"""
    sarr = stratarray([3, 2])
    trimmed = sarr.trim_mixture_precision(
        [1 / 3, 1 / 3, 1 / 3, 0.62, 0.38], resolution=0.1)
    # Ties resolve as first strategies
    assert np.allclose(trimmed, [0.4, 0.3, 0.3, 0.6, 0.4])

    trimmed = sarr.trim_mixture_precision(
        [0.5, 0.25, 0.25, 0.5, 0.5], resolution=0.05)
    assert np.allclose(trimmed, [0.5, 0.25, 0.25, 0.5, 0.5])


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_role_serialization(role_players, role_strats):
    """Test role serialization"""
    game = rsgame.empty(role_players, role_strats)
    roles = np.random.random((20, game.num_roles))
    copies = np.empty(roles.shape)
    for role, copy in zip(roles, copies):
        jrole = json.dumps(game.role_to_json(role))
        game.role_from_json(json.loads(jrole), copy)
    assert np.allclose(copies, roles)


# ---------
# EmptyGame
# ---------

def test_emptygame_properties():
    """Test empty game"""
    game = rsgame.empty(1, 1)
    assert np.all(game.num_role_players == [1])
    assert game.num_players == 1
    assert game.zero_prob.shape == (1,)
    devs, jac = game.deviation_payoffs(game.random_mixture(), jacobian=True)
    assert devs.shape == (1,)
    assert np.isnan(devs).all()
    assert jac.shape == (1, 1)
    assert np.isnan(jac).all()

    game = rsgame.empty(3, 1)
    assert np.all(game.num_role_players == [3])
    assert game.num_players == 3
    assert game.zero_prob.shape == (1,)

    game = rsgame.empty([1, 3], 1)
    assert np.all(game.num_role_players == [1, 3])
    assert game.num_players == 4
    assert game.zero_prob.shape == (2,)

    game = rsgame.empty([3, 2, 1], 1)
    assert np.all(game.num_role_players == [3, 2, 1])
    assert game.num_players == 6
    assert game.zero_prob.shape == (3,)


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_emptygame_const_properties(role_players, role_strats):
    """Test empty game properties"""
    game = rsgame.empty(role_players, role_strats)

    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    assert np.isnan(game.min_strat_payoffs()).all()
    assert game.min_strat_payoffs().shape == (game.num_strats,)
    assert np.isnan(game.max_strat_payoffs()).all()
    assert game.max_strat_payoffs().shape == (game.num_strats,)
    assert np.isnan(game.min_role_payoffs()).all()
    assert game.min_role_payoffs().shape == (game.num_roles,)
    assert np.isnan(game.max_role_payoffs()).all()
    assert game.max_role_payoffs().shape == (game.num_roles,)

    prof = game.random_profile()
    pays = game.get_payoffs(prof)
    assert np.isnan(pays[prof > 0]).all()
    assert np.all(pays[prof == 0] == 0)
    assert pays.shape == (game.num_strats,)

    dprof = game.random_role_deviation_profile()
    assert dprof.shape == (game.num_roles, game.num_strats)
    dpays = game.get_dev_payoffs(dprof)
    assert dpays.shape == (game.num_strats,)
    assert np.isnan(dpays).all()

    dprof = game.random_role_deviation_profile(game.random_mixture())
    assert dprof.shape == (game.num_roles, game.num_strats)
    dpays = game.get_dev_payoffs(dprof)
    assert dpays.shape == (game.num_strats,)
    assert np.isnan(dpays).all()

    dprofs = game.random_role_deviation_profiles(20)
    assert dprofs.shape == (20, game.num_roles, game.num_strats)
    dpays = game.get_dev_payoffs(dprofs)
    assert dpays.shape == (20, game.num_strats)
    assert np.isnan(dpays).all()

    mix = game.random_mixture()
    dev_pays = game.deviation_payoffs(mix)
    assert np.isnan(dev_pays).all()
    assert dev_pays.shape == (game.num_strats,)

    exp_pays = game.expected_payoffs(mix)
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)

    bresp = game.best_response(mix)
    assert np.isnan(bresp).all()
    assert bresp.shape == (game.num_strats,)

    assert game.profiles().size == 0
    assert game.payoffs().size == 0

    assert game.is_empty()
    assert not game.is_complete()
    assert game.is_constant_sum()

    assert game.normalize() == game
    assert game.random_profile() not in game


def test_empty_restriction():
    """Test empty game restriction"""
    game = rsgame.empty(1, [2, 3])
    rgame = game.restrict([False, True, True, False, True])
    expected = rsgame.empty_names(('r0', 'r1'), 1, (('s1',), ('s2', 's4')))
    assert rgame == expected

    game = rsgame.empty([3, 4, 5], [4, 3, 2])
    rgame = game.restrict(
        [False, True, True, False, False, False, True, True, False])
    expected = rsgame.empty_names(
        ('r0', 'r1', 'r2'), [3, 4, 5], (('s1', 's2'), ('s6',), ('s7',)))
    assert rgame == expected

    game = rsgame.empty(1, [2, 3])
    with pytest.raises(ValueError):
        game.restrict([False, False, True, True, True])


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_empty_restriction(role_players, role_strats):
    """TEst random empty restriction"""
    game = rsgame.empty(role_players, role_strats)
    rest = game.random_restriction()
    rgame = game.restrict(rest)
    assert np.all(game.num_role_players == rgame.num_role_players)
    assert rgame.num_strats == rest.sum()


def test_num_all_profiles():
    """Test num all profiles"""
    game = rsgame.empty(1, 1)
    assert np.all(game.num_all_role_profiles == [1])
    assert game.num_all_profiles == 1

    game = rsgame.empty(3, 2)
    assert np.all(game.num_all_role_profiles == [4])
    assert game.num_all_profiles == 4

    game = rsgame.empty([1, 3], 2)
    assert np.all(game.num_all_role_profiles == [2, 4])
    assert game.num_all_profiles == 8

    game = rsgame.empty(1, [3, 1])
    assert np.all(game.num_all_role_profiles == [3, 1])
    assert game.num_all_profiles == 3

    game = rsgame.empty([3, 2, 1], 3)
    assert np.all(game.num_all_role_profiles == [10, 6, 3])
    assert game.num_all_profiles == 180

    game = rsgame.empty([3, 2, 1], [1, 2, 3])
    assert np.all(game.num_all_role_profiles == [1, 3, 3])
    assert game.num_all_profiles == 9

    game = rsgame.empty([20, 20], 20)
    assert np.all(game.num_all_role_profiles == [68923264410, 68923264410])
    assert game.num_all_profiles == 4750416376930772648100


def test_num_all_payoffs():
    """Test num all payoffs"""
    game = rsgame.empty(1, 1)
    assert game.num_all_payoffs == 1

    game = rsgame.empty(3, 2)
    assert game.num_all_payoffs == 6

    game = rsgame.empty([1, 3], 2)
    assert game.num_all_payoffs == 20

    game = rsgame.empty(1, [3, 1])
    assert game.num_all_payoffs == 6

    game = rsgame.empty([3, 2, 1], 3)
    assert game.num_all_payoffs == 774

    game = rsgame.empty([3, 2, 1], [1, 2, 3])
    assert game.num_all_payoffs == 30


def test_num_all_dpr_profiles():
    """Test num dpr profiles"""
    game = rsgame.empty(1, 1)
    assert game.num_all_dpr_profiles == 1

    game = rsgame.empty(3, 2)
    assert game.num_all_dpr_profiles == 6

    game = rsgame.empty([1, 3], 2)
    assert game.num_all_dpr_profiles == 16

    game = rsgame.empty(1, [3, 1])
    assert game.num_all_dpr_profiles == 3

    game = rsgame.empty([3, 2, 1], [1, 2, 3])
    assert game.num_all_dpr_profiles == 15


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_profile_counts(role_players, role_strats):
    """Test random profile counts"""
    game = rsgame.empty(role_players, role_strats)

    num_role_profiles = np.fromiter(  # pragma: no branch
        (rsgame.empty(p, s).all_profiles().shape[0] for p, s
         in zip(game.num_role_players, game.num_role_strats)),
        int, game.num_roles)
    assert np.all(num_role_profiles == game.num_all_role_profiles)

    num_profiles = game.all_profiles().shape[0]
    assert num_profiles == game.num_all_profiles

    num_payoffs = np.sum(game.all_profiles() > 0)
    assert num_payoffs == game.num_all_payoffs


def test_profile_id():
    """Test profile ids"""
    game = rsgame.empty(3, [2, 2])
    profs = [[[0, 3, 2, 1],
              [2, 1, 3, 0]],
             [[2, 1, 2, 1],
              [3, 0, 3, 0]],
             [[1, 2, 1, 2],
              [2, 1, 1, 2]]]
    ids = game.profile_to_id(profs)
    assert ids.shape == (3, 2)
    assert np.all((ids >= 0) & (ids < game.num_all_profiles))

    game = rsgame.empty(3, [1, 2])
    prof = [3, 1, 2]
    assert game.profile_to_id(prof) == 2
    assert np.all(game.profile_from_id(2) == prof)

    game = rsgame.empty([1, 1, 1], [2, 2, 2])
    ids = np.arange(game.num_all_profiles)
    profs = game.profile_from_id(ids)
    assert np.all(profs == game.all_profiles())


def test_profile_id_big():
    """Test large profile ids"""
    game = rsgame.empty([20, 20], 20)
    profile = np.zeros(40, int)
    profile[[19, 39]] = 20
    assert game.profile_to_id(profile) == 4750416376930772648099
    assert np.all(game.profile_from_id(4750416376930772648099) == profile)

    game = rsgame.empty(40, 40)
    profile = np.zeros(40, int)
    profile[39] = 40
    assert game.profile_to_id(profile) == 53753604366668088230809
    assert np.all(game.profile_from_id(53753604366668088230809) == profile)


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_profile_id(role_players, role_strats):
    """Test random profile ids"""
    # Here we have an expectation that all_profiles always returns profiles in
    # order of id
    game = rsgame.empty(role_players, role_strats)
    expected = np.arange(game.num_all_profiles)
    actual = game.profile_to_id(game.all_profiles())
    assert np.all(expected == actual)


def test_big_game_functions():
    """Test that everything works when game_size > int max"""
    game = rsgame.empty([100, 100], [30, 30])
    assert game.num_all_profiles > np.iinfo(int).max
    assert game.num_all_dpr_profiles > np.iinfo(int).max
    assert np.all(game.profile_to_id(game.random_profiles(1000)) >= 0)


def test_is_profile():
    """Test is profile"""
    game = rsgame.empty([2, 3], [3, 2])
    assert game.is_profile([1, 0, 1, 2, 1])
    assert not game.is_profile([1, 0, 2, 2, 1])
    assert not game.is_profile([1, -1, 2, 2, 1])
    assert not game.is_profile([1, 0, 0, 2, 1])
    assert not game.is_profile([1, 0, 1, 2, 2])
    assert not game.is_profile([1, 0, 1, 2, 0])
    assert np.all(([True] + [False] * 5) == game.is_profile([
        [1, 0, 1, 2, 1],
        [1, 0, 2, 2, 1],
        [1, -1, 2, 2, 1],
        [1, 0, 0, 2, 1],
        [1, 0, 1, 2, 2],
        [1, 0, 1, 2, 0]]))
    assert game.is_profile([[1], [0], [1], [2], [1]], axis=0)

    with pytest.raises(ValueError):
        game.is_profile([0, 0, 0, 0])
    with pytest.raises(ValueError):
        game.is_profile([0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        game.is_profile([[0, 0, 0, 0, 0, 0]])


def test_is_pure_profile():
    """Test is pure profile"""
    game = rsgame.empty([2, 3], [3, 2])
    assert game.is_pure_profile([2, 0, 0, 3, 0])
    assert not game.is_pure_profile([1, 0, 2, 2, 1])
    assert not game.is_pure_profile([1, 0, 1, 3, 0])
    assert not game.is_pure_profile([1, 1, 0, 2, 1])

    with pytest.raises(ValueError):
        game.is_pure_profile([0, 0, 0, 0])


def test_all_profiles():
    """Test all profiles"""
    game = rsgame.empty(1, 1)
    expected = [[1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.empty(3, 2)
    expected = [[3, 0],
                [2, 1],
                [1, 2],
                [0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.empty([1, 3], 2)
    expected = [[1, 0, 3, 0],
                [1, 0, 2, 1],
                [1, 0, 1, 2],
                [1, 0, 0, 3],
                [0, 1, 3, 0],
                [0, 1, 2, 1],
                [0, 1, 1, 2],
                [0, 1, 0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.empty(1, [3, 1])
    expected = [[1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.empty([3, 2, 1], [1, 2, 3])
    expected = [[3, 2, 0, 1, 0, 0],
                [3, 2, 0, 0, 1, 0],
                [3, 2, 0, 0, 0, 1],
                [3, 1, 1, 1, 0, 0],
                [3, 1, 1, 0, 1, 0],
                [3, 1, 1, 0, 0, 1],
                [3, 0, 2, 1, 0, 0],
                [3, 0, 2, 0, 1, 0],
                [3, 0, 2, 0, 0, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size


def test_pure_profiles():
    """Test pure profiles"""
    game = rsgame.empty(1, 1)
    expected = [[1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.empty(3, 2)
    expected = [[3, 0],
                [0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.empty([1, 3], 2)
    expected = [[1, 0, 3, 0],
                [1, 0, 0, 3],
                [0, 1, 3, 0],
                [0, 1, 0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.empty(1, [3, 1])
    expected = [[1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.empty([3, 2, 1], [1, 2, 3])
    expected = [[3, 2, 0, 1, 0, 0],
                [3, 2, 0, 0, 1, 0],
                [3, 2, 0, 0, 0, 1],
                [3, 0, 2, 1, 0, 0],
                [3, 0, 2, 0, 1, 0],
                [3, 0, 2, 0, 0, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size


def test_nearby_profiles_1():
    """This is essentially just testing single deviations"""
    game = rsgame.empty(1, 1)
    prof = [1]
    expected = [1]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.empty(3, 2)
    prof = [3, 0]
    expected = [[3, 0],
                [2, 1]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size
    prof = [2, 1]
    expected = [[3, 0],
                [2, 1],
                [1, 2]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.empty([1, 3], 2)
    prof = [1, 0, 0, 3]
    expected = [[0, 1, 0, 3],
                [1, 0, 0, 3],
                [1, 0, 1, 2]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size
    prof = [1, 0, 2, 1]
    expected = [[0, 1, 2, 1],
                [1, 0, 2, 1],
                [1, 0, 3, 0],
                [1, 0, 1, 2]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.empty(1, [3, 1])
    prof = [0, 0, 1, 1]
    expected = [[1, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.empty([3, 2, 1], [1, 2, 3])
    prof = [3, 2, 0, 1, 0, 0]
    expected = [[3, 1, 1, 1, 0, 0],
                [3, 2, 0, 1, 0, 0],
                [3, 2, 0, 0, 1, 0],
                [3, 2, 0, 0, 0, 1]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size
    prof = [3, 1, 1, 1, 0, 0]
    expected = [[3, 2, 0, 1, 0, 0],
                [3, 1, 1, 1, 0, 0],
                [3, 0, 2, 1, 0, 0],
                [3, 1, 1, 0, 1, 0],
                [3, 1, 1, 0, 0, 1]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
@pytest.mark.parametrize('num_devs', range(5))
def test_random_nearby_profiles(role_players, role_strats, num_devs):
    """Test random nearby profiles"""
    base = rsgame.empty(role_players, role_strats)
    prof = base.random_profile()
    nearby = base.nearby_profiles(prof, num_devs)
    diff = nearby - prof
    devs_from = np.add.reduceat((diff < 0) * -diff, base.role_starts, 1)
    devs_to = np.add.reduceat((diff > 0) * diff, base.role_starts, 1)
    assert np.all(devs_to.sum(1) <= num_devs)
    assert np.all(devs_from.sum(1) <= num_devs)
    assert np.all(devs_to == devs_from)
    assert np.all(base.is_profile(nearby))


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_fixed_profiles(role_players, role_strats):
    """Test fixed profiles"""
    game = rsgame.empty(role_players, role_strats)
    all_profiles = game.all_profiles()
    assert game.num_all_profiles == all_profiles.shape[0]
    assert game.is_profile(all_profiles).all()
    pure_profiles = game.pure_profiles()
    assert game.num_pure_restrictions == pure_profiles.shape[0]
    assert game.is_profile(pure_profiles).all()


def test_random_profiles():
    """Test random profiles"""
    game = rsgame.empty(3, 3)
    mixes = game.random_profiles(100, [0, 0.4, 0.6])
    assert np.all(mixes[:, 0] == 0)


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_random_profiles(role_players, role_strats):
    """Test random profiles"""
    game = rsgame.empty(role_players, role_strats)
    assert game.is_profile(game.random_profiles(100)).all()


def test_round_mixture_to_profile():
    """Test round mixture"""
    game = rsgame.empty(3, 3)

    prof = game.round_mixture_to_profile([1 / 3, 1 / 3, 1 / 3])
    assert np.all(prof == 1)

    prof = game.round_mixture_to_profile([2 / 3, 0, 1 / 3])
    assert np.all(prof == [2, 0, 1])

    prof = game.round_mixture_to_profile([.1, .2, .7])
    assert np.all(prof == [0, 1, 2])


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_round_mixture_to_profile(role_players, role_strats):
    """Test round mixture to profile"""
    game = rsgame.empty(role_players, role_strats)
    mixtures = np.concatenate([
        game.random_mixtures(100),
        game.random_sparse_mixtures(100),
    ])
    assert game.is_profile(game.round_mixture_to_profile(mixtures)).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_random_dev_profiles(role_players, role_strats):
    """Test random dev profiles"""
    game = rsgame.empty(role_players, role_strats)
    prof = game.random_role_deviation_profile()
    for role, dprof in enumerate(prof):
        role_players = game.num_role_players.copy()
        role_players[role] -= 1
        dgame = rsgame.empty(role_players, game.num_role_strats)
        assert dgame.is_profile(dprof).all()

    profs = game.random_role_deviation_profiles(100)
    assert profs.shape == (100, game.num_roles, game.num_strats)
    for role, dprofs in enumerate(np.rollaxis(profs, 1, 0)):
        role_players = game.num_role_players.copy()
        role_players[role] -= 1
        dgame = rsgame.empty(role_players, game.num_role_strats)
        assert dgame.is_profile(dprofs).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_random_deviator_profiles(role_players, role_strats):
    """Test random deviator profiles"""
    game = rsgame.empty(role_players, role_strats)
    profs = game.random_deviation_profile()
    assert profs.shape == (game.num_strats, game.num_strats)
    assert game.is_profile(profs).all()

    profs = game.random_deviation_profiles(100)
    assert profs.shape == (100, game.num_strats, game.num_strats)
    prof_mins = np.minimum.reduceat(
        profs, game.role_starts, 1).repeat(game.num_role_strats, 1)
    prof_devs = profs - prof_mins
    non_strat_roles = np.repeat(
        game.num_role_strats == 1, game.num_role_strats)
    assert np.all(np.any(prof_devs, 1) | non_strat_roles)
    assert np.all(np.any(prof_devs, 2) | non_strat_roles)
    assert game.is_profile(profs).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_max_prob_prof(role_players, role_strats):
    """Test max probability profile"""
    game = rsgame.empty(role_players, role_strats)
    profiles = game.all_profiles()
    log_prob = (np.sum(sps.gammaln(game.num_role_players + 1)) -
                np.sum(sps.gammaln(profiles + 1), 1))
    for mix in game.random_mixtures(100):
        probs = np.sum(np.log(mix + TINY) * profiles, 1) + log_prob
        mask = np.max(probs) - EPS < probs
        max_prob_profs = profiles[mask]
        actual = game.max_prob_prof(mix)
        assert np.all(np.in1d(utils.axis_to_elem(actual),
                              utils.axis_to_elem(max_prob_profs)))


def test_to_from_prof_json():
    """Test to from profile json"""
    game = rsgame.empty([11, 3], [2, 1])
    prof = [6, 5, 3]
    json_prof = {'r0': {'s1': 5, 's0': 6}, 'r1': {'s2': 3}}
    assert game.profile_to_json(prof) == json_prof
    new_prof = game.profile_from_json(json_prof)
    assert np.all(new_prof == prof)
    assert new_prof.dtype == int
    new_prof.fill(0)
    game.profile_from_json(json_prof, dest=new_prof)
    assert np.all(new_prof == prof)
    assert new_prof.dtype == int


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_profile_serialization(role_players, role_strats):
    """Test profile serialization"""
    game = rsgame.empty(role_players, role_strats)
    profs = game.random_profiles(20)
    copies = np.empty(profs.shape, int)
    for prof, copy in zip(profs, copies):
        jprof = json.dumps(game.profile_to_json(prof))
        game.profile_from_json(json.loads(jprof), copy)
    assert np.all(copies == profs)


def test_to_from_payoff_json():
    """Test to from payoff json"""
    game = rsgame.empty([11, 3], [2, 1])
    pay = [1, 2, 3]
    json_pay = {'r0': {'s1': 2, 's0': 1}, 'r1': {'s2': 3}}
    assert game.payoff_to_json(pay) == json_pay
    new_pay = game.payoff_from_json(json_pay)
    assert np.allclose(new_pay, pay)
    assert new_pay.dtype == float
    new_pay.fill(0)
    game.payoff_from_json(json_pay, dest=new_pay)
    assert np.allclose(new_pay, pay)
    assert new_pay.dtype == float

    pay = [np.nan, 2, 3]
    json_pay = {'r0': {'s1': [1, 3], 's0': []}, 'r1': {'s2': [2, 3, 4]}}
    new_pay = game.payoff_from_json(json_pay)
    assert np.allclose(new_pay, pay, equal_nan=True)
    assert new_pay.dtype == float

    pay = [1, 0, 3]
    json_pay = {'r0': {'s0': 1}, 'r1': {'s2': 3}}
    json_pay0 = {'r0': {'s1': 0, 's0': 1}, 'r1': {'s2': 3}}
    assert game.payoff_to_json(pay) == json_pay
    new_pay = game.payoff_from_json(json_pay)
    assert np.allclose(new_pay, pay)
    new_pay = game.payoff_from_json(json_pay0)
    assert np.allclose(new_pay, pay)


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_payoff_serialization(role_players, role_strats):
    """Test payoff serialization"""
    game = rsgame.empty(role_players, role_strats)
    pays = np.random.random((20, game.num_strats))
    pays *= pays < 0.8
    copies = np.empty(pays.shape)
    for pay, copy in zip(pays, copies):
        jpay = json.dumps(game.payoff_to_json(pay))
        game.payoff_from_json(json.loads(jpay), copy)
    assert np.allclose(copies, pays)


def test_to_from_prof_repr():
    """Test profile repr"""
    game = rsgame.empty([11, 3], [2, 1])
    prof = [6, 5, 3]
    prof_str = 'r0: 6 s0, 5 s1; r1: 3 s2'
    assert game.profile_to_repr(prof) == prof_str
    aprof = game.profile_from_repr(prof_str)
    assert np.all(aprof == prof)
    aprof = np.empty_like(aprof)
    game.profile_from_repr(prof_str, dest=aprof)
    assert np.all(aprof == prof)


def test_to_from_prof_str():
    """Test profile string"""
    game = rsgame.empty([11, 3], [2, 1])
    prof = [6, 5, 3]
    prof_str = """
r0:
    s0: 6
    s1: 5
r1:
    s2: 3
"""[1:-1]
    assert game.profile_to_str(prof) == prof_str
    assert np.all(game.profile_from_str(prof_str) == prof)


def test_dev_payoff_json():
    """Test dev payoff json"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'using reduceat with dev_role_starts will not produce correct '
            'results if any role only has one strategy. This might get fixed '
            'at some point, but currently extra care must be taken for these '
            'cases.',
            UserWarning)
        game = rsgame.empty([11, 3], [2, 1])
        devpay = [5, 0]
        json_devpay = {'r0': {'s0': {'s1': 5}}}
        json_devpay2 = {'r0': {'s0': {'s1': 5}, 's1': {'s0': 0}},
                        'r1': {'s2': {}}}
        assert game.devpay_to_json(devpay) == json_devpay
        dest = np.empty(game.num_devs)
        game.devpay_from_json(json_devpay, dest)
        assert np.allclose(dest, devpay)
        assert np.allclose(game.devpay_from_json(json_devpay), devpay)
        assert np.allclose(game.devpay_from_json(json_devpay2), devpay)

        devpay = [5, 4]
        json_devpay = {'r0': {'s0': {'s1': 5}, 's1': {'s0': 4}}}
        assert game.devpay_to_json(devpay) == json_devpay
        dest = np.empty(game.num_devs)
        game.devpay_from_json(json_devpay, dest)
        assert np.allclose(dest, devpay)
        assert np.allclose(game.devpay_from_json(json_devpay), devpay)


@pytest.mark.parametrize('role_players', [1, 2, 3, [3, 2, 1]])
@pytest.mark.parametrize('role_strats', [2, 4, [2, 3, 4]])
def test_random_devpay_serialization(role_players, role_strats):
    """Test dev payoff serialization"""
    game = rsgame.empty(role_players, role_strats)
    pays = np.random.random((20, game.num_devs))
    pays *= pays < 0.8
    copies = np.empty(pays.shape)
    for pay, copy in zip(pays, copies):
        jpay = json.dumps(game.devpay_to_json(pay))
        game.devpay_from_json(json.loads(jpay), copy)
    assert np.allclose(copies, pays)


def test_is_symmetric():
    """Test is symmetric"""
    assert rsgame.empty(3, 4).is_symmetric()
    assert not rsgame.empty([2, 2], 3).is_symmetric()


def test_is_asymmetric():
    """Test asymmetric"""
    assert rsgame.empty(1, 4).is_asymmetric()
    assert not rsgame.empty([1, 2], 3).is_asymmetric()


def test_to_from_json():
    """Test to from json"""
    game = rsgame.empty(4, 5)
    jgame = {'players': {'r0': 4},
             'strategies': {'r0': ['s0', 's1', 's2', 's3', 's4']},
             'type': 'empty.1'}
    old_jgame = {'roles': [{'name': 'r0',
                            'strategies': ['s0', 's1', 's2', 's3', 's4'],
                            'count': 4}]}
    assert game.to_json() == jgame
    assert rsgame.empty_json(jgame) == game
    assert rsgame.empty_json(old_jgame) == game
    json.dumps(game.to_json())  # serializable

    game = rsgame.empty([4, 3], [3, 4])
    jgame = {'players': {'r0': 4, 'r1': 3},
             'strategies': {'r0': ['s0', 's1', 's2'],
                            'r1': ['s3', 's4', 's5', 's6']},
             'type': 'empty.1'}
    old_jgame = {'roles': [{'name': 'r0',
                            'strategies': ['s0', 's1', 's2'],
                            'count': 4},
                           {'name': 'r1',
                            'strategies': ['s3', 's4', 's5', 's6'],
                            'count': 3}]}
    assert game.to_json() == jgame
    assert json.loads(json.dumps(game.to_json())) == jgame
    assert rsgame.empty_json(jgame) == game
    assert rsgame.empty_json(old_jgame) == game

    with pytest.raises(ValueError):
        rsgame.empty_json({})


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_json_serialization(role_players, role_strats):
    """Test random serialization"""
    game = rsgame.empty(role_players, role_strats)
    jgame = json.dumps(game.to_json())
    copy = rsgame.empty_json(json.loads(jgame))
    assert copy == game


def test_emptygame_hash_eq():
    """Test random hash and equality"""
    one = rsgame.empty(4, 5)
    two = rsgame.empty([4], [5])
    assert one == two and hash(one) == hash(two)

    one = rsgame.empty([1, 2], [3, 2])
    two = rsgame.empty([1, 2], [3, 2])
    assert one == two and hash(one) == hash(two)

    one = rsgame.empty([2], [3, 2])
    two = rsgame.empty([2, 2], [3, 2])
    assert one == two and hash(one) == hash(two)

    one = rsgame.empty([2, 3], [3])
    two = rsgame.empty([2, 3], [3, 3])
    assert one == two and hash(one) == hash(two)

    assert rsgame.empty(3, 4) != rsgame.empty(3, 5)
    assert rsgame.empty(3, 4) != rsgame.empty(2, 4)
    assert rsgame.empty([1, 2], 4) != rsgame.empty([2, 2], 4)
    assert rsgame.empty([1, 2], 4) != rsgame.empty([2, 1], 4)
    assert rsgame.empty(2, [2, 3]) != rsgame.empty(2, [2, 2])
    assert rsgame.empty(2, [2, 3]) != rsgame.empty(2, [3, 2])


def test_emptygame_repr():
    """Test emptygame repr"""
    game = rsgame.empty(3, 4)
    expected = 'EmptyGame([3], [4])'
    assert repr(game) == expected

    game = rsgame.empty(3, [4, 5])
    expected = 'EmptyGame([3 3], [4 5])'
    assert repr(game) == expected


def test_emptygame_str():
    """Test emptygame string"""
    game = rsgame.empty(3, 4)
    expected = """
EmptyGame:
    Roles: r0
    Players:
        3x r0
    Strategies:
        r0:
            s0
            s1
            s2
            s3
"""[1:-1]
    assert str(game) == expected

    game = rsgame.empty([3, 4], [4, 3])
    expected = """
EmptyGame:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
            s3
        r1:
            s4
            s5
            s6
"""[1:-1]
    assert str(game) == expected


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_random_emptygame_copy(role_players, role_strats):
    """Test emptygame copy"""
    game = rsgame.empty(role_players, role_strats)
    copy = rsgame.empty_copy(game)
    assert game == copy and hash(game) == hash(copy)


def test_empty_add_multiply():
    """Test emptygame add and multiply"""
    empty = rsgame.empty([1, 2], [3, 2])

    assert empty + 1 == empty
    assert empty - 1 == empty
    assert empty * 2 == empty
    assert empty / 2 == empty

    assert empty + [1, 2] == empty
    assert empty - [1, 2] == empty
    assert empty * [2, 3] == empty
    assert empty / [2, 3] == empty

    assert 1 + empty == empty
    assert 2 * empty == empty

    assert [1, 2] + empty == empty
    assert [2, 3] * empty == empty

    assert empty + empty == empty


@pytest.mark.parametrize('role_players,role_strats', testutils.GAMES)
def test_const_game(role_players, role_strats):
    """Test constant game"""
    game = rsgame.const(role_players, role_strats, 0)
    assert game.is_complete()
    assert game.is_constant_sum()
    assert not game.is_empty()
    assert game.num_profiles == game.num_all_profiles
    assert game.num_complete_profiles == game.num_all_profiles
    assert all(prof in game for prof in game.all_profiles())
    assert np.all(game.profiles() == game.all_profiles())

    assert np.allclose(game.deviation_payoffs(game.random_mixture()), 0)
    dev, jac = game.deviation_payoffs(game.random_mixture(), jacobian=True)
    assert np.allclose(dev, 0)
    assert dev.shape == (game.num_strats,)
    assert np.allclose(jac, 0)
    assert jac.shape == (game.num_strats,) * 2

    jstr = json.dumps(game.to_json())
    assert game == rsgame.const_json(json.loads(jstr))
    assert np.allclose(game.max_strat_payoffs(), 0)
    assert np.allclose(game.min_strat_payoffs(), 0)
    assert game == game.normalize()
    assert game == game.restrict(np.ones(game.num_strats, bool))


def test_repr():
    """Test repr"""
    const = rsgame.const([1, 2], [3, 2], 1)
    assert repr(const) == 'ConstantGame([1 2], [3 2], [1. 1.])'


def test_const_add_multiply():
    """Test const add multiple"""
    empty = rsgame.empty([1, 2], [3, 2])
    const1 = rsgame.const_replace(empty, 1)
    assert np.allclose(const1.payoffs(), const1.profiles() > 0)
    const2 = rsgame.const_replace(empty, [2, 3])
    assert np.allclose(
        const2.payoffs(),
        np.where(const2.profiles() > 0, [2, 2, 2, 3, 3], 0))

    assert const2 + 1 == rsgame.const_replace(empty, [3, 4])
    assert const2 - 1 == rsgame.const_replace(empty, [1, 2])
    assert const2 * 2 == rsgame.const_replace(empty, [4, 6])
    assert const2 / 2 == rsgame.const_replace(empty, [1, 3/2])

    assert const2 + [2, 1] == rsgame.const_replace(empty, 4)
    assert const2 - [1, 2] == rsgame.const_replace(empty, 1)
    assert const2 * [3, 2] == rsgame.const_replace(empty, 6)
    assert const2 / [2, 3] == rsgame.const_replace(empty, 1)

    assert 1 + const2 == rsgame.const_replace(empty, [3, 4])
    assert 2 * const2 == rsgame.const_replace(empty, [4, 6])

    assert const1 + const2 == rsgame.const_replace(empty, [3, 4])
    assert const2 + const1 == rsgame.const_replace(empty, [3, 4])


def test_const_names():
    """Test constant names"""
    game = rsgame.const_names(
        ['a', 'b'], [2, 3], [['1', '2'], ['3', '4', '5']], 1)
    assert game.role_names == ('a', 'b')
    assert np.all(game.num_role_players == [2, 3])
    assert game.strat_names == (('1', '2'), ('3', '4', '5'))


def test_mix():
    """Test mixture game"""
    empty = rsgame.empty([1, 2], [3, 2])
    const1 = rsgame.const_replace(empty, [1, 5])
    const2 = rsgame.const_replace(empty, [4, -1])
    mix = rsgame.mix(const1, const2, 1/3)
    assert mix == rsgame.const_replace(empty, [2, 3])
    assert rsgame.mix(const1, const2, 0) == const1
    assert rsgame.mix(const1, const2, 1) == const2


def test_add_const():
    """Test add constant"""
    empty = rsgame.empty([1, 2], [3, 2])
    const1 = rsgame.const_replace(empty, [1, 5])
    const2 = rsgame.const_replace(empty, [4, -1])
    const3 = rsgame.const_replace(empty, 1)
    add = rsgame.add(const1, const2, const3)
    assert add == rsgame.const_replace(empty, [6, 5])


def test_add_types():
    """Test add types"""
    empty = rsgame.empty([1, 2], [3, 2])
    const = UnAddC(empty, 1)

    assert empty + const == empty
    assert const + empty == empty
    assert rsgame.add(empty, const) == empty
    assert rsgame.add(const, empty) == empty

    with pytest.raises(TypeError):
        assert 'string' + empty
    with pytest.raises(TypeError):
        assert empty + 'string'
    with pytest.raises(TypeError):
        assert empty - 'string'
    with pytest.raises(TypeError):
        assert empty * 'string'
    with pytest.raises(TypeError):
        assert empty / 'string'


def test_add_game():
    """Test add games"""
    empty = rsgame.empty([1, 2], [3, 2])
    unadd1 = UnAddC(empty, 1)
    unadd2 = UnAddC(empty, [1, 2])
    add = unadd1 + unadd2

    assert add.is_complete()
    for prof in add.all_profiles():
        assert prof in add
    prof = add.random_profile()
    assert np.allclose(
        add.payoffs(),
        np.where(add.profiles() > 0, [2, 2, 2, 3, 3], 0))
    assert np.allclose(add.min_role_payoffs(), [2, 3])
    assert np.allclose(add.max_role_payoffs(), [2, 3])

    assert np.allclose(add.deviation_payoffs(add.random_mixture()),
                       [2, 2, 2, 3, 3])
    dev, jac = add.deviation_payoffs(add.random_mixture(), jacobian=True)
    assert np.allclose(dev, [2, 2, 2, 3, 3])
    assert dev.shape == (add.num_strats,)
    assert np.allclose(jac, 0)
    assert jac.shape == (add.num_strats,) * 2

    jstr = json.dumps(add.to_json())
    loaded = rsgame.add_json(json.loads(jstr))  # Won't load proper type
    assert np.allclose(add.payoffs(), loaded.get_payoffs(add.profiles()))
    assert repr(add) == 'AddGame([1 2], [3 2], 9 / 9)'

    add_comm = unadd2 + unadd1
    assert hash(add) == hash(add_comm)
    assert add == add_comm
    add3 = add + unadd1
    assert add3 == add_comm + unadd1
    assert np.allclose(add3.get_payoffs(prof), (prof > 0) * [3, 3, 3, 4, 4])
    add_add = add + [2, 5]
    assert np.allclose(add_add.get_payoffs(prof), (prof > 0) * [4, 4, 4, 8, 8])
    add_mul = add * 2
    assert np.allclose(add_mul.get_payoffs(prof), (prof > 0) * [4, 4, 4, 6, 6])

    rest = [True, True, False, True, True]
    add_rest = add.restrict(rest)
    assert rsgame.empty_copy(add_rest) == empty.restrict(rest)


class UnAddC(rsgame._ConstantGame): # pylint: disable=protected-access
    """A constant game that doesn't support adding"""
    def __init__(self, copy, const):
        super().__init__(
            copy.role_names, copy.strat_names, copy.num_role_players,
            np.asarray(const, float))

    def _add_game(self, _):
        return NotImplemented
