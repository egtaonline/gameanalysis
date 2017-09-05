import collections
import warnings

import numpy as np
import numpy.random as rand
import pytest
import scipy.special as sps

from gameanalysis import reduction
from gameanalysis import rsgame
from gameanalysis import utils
from test import testutils


TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps


# ----------
# StratArray
# ----------


def test_stratarray_properties():
    sarr = rsgame.StratArray(np.array([1]))
    assert sarr.num_strats == 1
    assert np.all(sarr.num_role_strats == [1])
    assert sarr.num_roles == 1
    assert np.all(sarr.role_starts == [0])
    assert np.all(sarr.role_indices == [0])
    assert sarr.num_all_subgames == 1
    assert sarr.num_pure_subgames == 1
    assert np.all(sarr.num_strat_devs == [0])
    assert np.all(sarr.num_role_devs == [0])
    assert sarr.num_devs == 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_strat_starts == [0])
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_role_starts == [0])
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    assert np.all(sarr.dev_from_indices == [])
    assert np.all(sarr.dev_to_indices == [])

    sarr = rsgame.StratArray(np.array([3]))
    assert sarr.num_strats == 3
    assert np.all(sarr.num_role_strats == [3])
    assert sarr.num_roles == 1
    assert np.all(sarr.role_starts == [0])
    assert np.all(sarr.role_indices == [0, 0, 0])
    assert sarr.num_all_subgames == 7
    assert sarr.num_pure_subgames == 3
    assert np.all(sarr.num_strat_devs == [2, 2, 2])
    assert np.all(sarr.num_role_devs == [6])
    assert sarr.num_devs == 6
    assert np.all(sarr.dev_strat_starts == [0, 2, 4])
    assert np.all(sarr.dev_role_starts == [0])
    assert np.all(sarr.dev_from_indices == [0, 0, 1, 1, 2, 2])
    assert np.all(sarr.dev_to_indices == [1, 2, 0, 2, 0, 1])

    sarr = rsgame.StratArray(np.array([1, 3]))
    assert sarr.num_strats == 4
    assert np.all(sarr.num_role_strats == [1, 3])
    assert sarr.num_roles == 2
    assert np.all(sarr.role_starts == [0, 1])
    assert np.all(sarr.role_indices == [0, 1, 1, 1])
    assert sarr.num_all_subgames == 7
    assert sarr.num_pure_subgames == 3
    assert np.all(sarr.num_strat_devs == [0, 2, 2, 2])
    assert np.all(sarr.num_role_devs == [0, 6])
    assert sarr.num_devs == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_strat_starts == [0, 0, 2, 4])
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_role_starts == [0, 0])
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    assert np.all(sarr.dev_from_indices == [1, 1, 2, 2, 3, 3])
    assert np.all(sarr.dev_to_indices == [2, 3, 1, 3, 1, 2])

    sarr = rsgame.StratArray(np.array([3, 2, 1]))
    assert sarr.num_strats == 6
    assert np.all(sarr.num_role_strats == [3, 2, 1])
    assert sarr.num_roles == 3
    assert np.all(sarr.role_starts == [0, 3, 5])
    assert np.all(sarr.role_indices == [0, 0, 0, 1, 1, 2])
    assert sarr.num_all_subgames == 21
    assert sarr.num_pure_subgames == 6
    assert np.all(sarr.num_strat_devs == [2, 2, 2, 1, 1, 0])
    assert np.all(sarr.num_role_devs == [6, 2, 0])
    assert sarr.num_devs == 8
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_strat_starts == [0, 2, 4, 6, 7, 8])
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert np.all(sarr.dev_role_starts == [0, 6, 8])
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    assert np.all(sarr.dev_from_indices == [0, 0, 1, 1, 2, 2, 3, 4])
    assert np.all(sarr.dev_to_indices == [1, 2, 0, 2, 0, 1, 4, 3])


def test_subgame_enumeration():
    sarr = rsgame.StratArray(np.array([1]))
    all_subgames = [[True]]
    assert not np.setxor1d(utils.axis_to_elem(all_subgames),
                           utils.axis_to_elem(sarr.all_subgames())).size
    pure_subgames = [[True]]
    assert not np.setxor1d(utils.axis_to_elem(pure_subgames),
                           utils.axis_to_elem(sarr.pure_subgames())).size

    sarr = rsgame.StratArray(np.array([3]))
    all_subgames = [[True, False, False],
                    [False, True, False],
                    [True, True, False],
                    [False, False, True],
                    [True, False, True],
                    [False, True, True],
                    [True, True, True]]
    assert not np.setxor1d(utils.axis_to_elem(all_subgames),
                           utils.axis_to_elem(sarr.all_subgames())).size
    pure_subgames = [[True, False, False],
                     [False, True, False],
                     [False, False, True]]
    assert not np.setxor1d(utils.axis_to_elem(pure_subgames),
                           utils.axis_to_elem(sarr.pure_subgames())).size

    sarr = rsgame.StratArray(np.array([1, 3]))
    all_subgames = [[True, True, False, False],
                    [True, False, True, False],
                    [True, True, True, False],
                    [True, False, False, True],
                    [True, True, False, True],
                    [True, False, True, True],
                    [True, True, True, True]]
    assert not np.setxor1d(utils.axis_to_elem(all_subgames),
                           utils.axis_to_elem(sarr.all_subgames())).size
    pure_subgames = [[True, True, False, False],
                     [True, False, True, False],
                     [True, False, False, True]]
    assert not np.setxor1d(utils.axis_to_elem(pure_subgames),
                           utils.axis_to_elem(sarr.pure_subgames())).size


def test_is_subgame():
    sarr = rsgame.StratArray(np.array([3, 2]))
    assert sarr.is_subgame([True, False, True, False, True])
    assert not sarr.is_subgame([True, False, True, False, False])
    assert not sarr.is_subgame([False, False, False, True, False])
    assert not sarr.is_subgame([False, False, False, False, False])
    assert np.all([True] + [False] * 3 == sarr.is_subgame([
        [True, False, True, False, True],
        [True, False, True, False, False],
        [False, False, False, True, False],
        [False, False, False, False, False]]))
    assert sarr.is_subgame([[True], [False], [True], [False], [True]], axis=0)

    with pytest.raises(AssertionError):
        sarr.is_subgame([False, False, False, False])
    with pytest.raises(AssertionError):
        sarr.is_subgame([False, False, False, False, False, False])
    with pytest.raises(AssertionError):
        sarr.is_subgame([[False, False, False, False, False, False]])


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_stratarray_subgames(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    all_subgames = sarr.all_subgames()
    assert sarr.is_subgame(all_subgames).all()
    assert sarr.num_all_subgames == all_subgames.shape[0]
    pure_subgames = sarr.pure_subgames()
    assert sarr.is_subgame(pure_subgames).all()
    assert sarr.num_pure_subgames == pure_subgames.shape[0]


def test_random_subgames():
    # Technically some of these can fail, but it's extremely unlikely
    sarr = rsgame.StratArray(np.array([3]))
    subgs = sarr.random_subgames(1000)
    assert sarr.is_subgame(subgs).all()
    assert not subgs.all()

    subgs = sarr.random_subgames(1000, 1)
    assert sarr.is_subgame(subgs).all()
    assert subgs.all()

    # Probability is raised to 1/3
    subgs = sarr.random_subgames(1000, 0)
    assert sarr.is_subgame(subgs).all()

    subgs = sarr.random_subgames(1000, 0, False)
    assert sarr.is_subgame(subgs).all()

    sarr = rsgame.StratArray(np.array([3, 2]))
    subgs = sarr.random_subgames(1000, [1, 1 / 2])
    assert sarr.is_subgame(subgs).all()
    assert np.all([True, True, True, False, False] == subgs.all(0))


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_random_subgames(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    subg = sarr.random_subgames()
    assert len(subg.shape) == 1

    subgs = sarr.random_subgames(100)
    assert sarr.is_subgame(subgs).all()


def test_trim_mixture_support():
    sarr = rsgame.StratArray(np.array([3]))
    mix = np.array([0.7, 0.3, 0])
    not_trimmed = sarr.trim_mixture_support(mix, 0.1)
    assert np.allclose(mix, not_trimmed)
    trimmed = sarr.trim_mixture_support(mix, 0.4)
    assert np.allclose([1, 0, 0], trimmed)

    trimmed = sarr.trim_mixture_support(mix[:, None], 0.4, 0)[:, 0]
    assert np.allclose([1, 0, 0], trimmed)


def test_is_mixture():
    sarr = rsgame.StratArray(np.array([3, 2]))
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

    with pytest.raises(AssertionError):
        sarr.is_mixture([0, 0, 0, 0])
    with pytest.raises(AssertionError):
        sarr.is_mixture([0, 0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        sarr.is_mixture([[0, 0, 0, 0, 0, 0]])


def test_mixture_project():
    sarr = rsgame.StratArray(np.array([1]))
    mixtures = [[0],
                [1],
                [2],
                [-1]]
    expected = [[1]] * 4
    assert np.allclose(expected, sarr.mixture_project(mixtures))

    sarr = rsgame.StratArray(np.array([3]))
    mixtures = [[0, 0, 0],
                [1, 0, 0],
                [2, 1, 0],
                [1.2, 1.3, 1.5]]
    expected = [[1 / 3, 1 / 3, 1 / 3],
                [1, 0, 0],
                [1, 0, 0],
                [0.2, 0.3, 0.5]]
    assert np.allclose(expected, sarr.mixture_project(mixtures))

    sarr = rsgame.StratArray(np.array([1, 3]))
    mixtures = [[0, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 2, 1, 0],
                [-1, 1.2, 1.3, 1.5]]
    expected = [[1, 1 / 3, 1 / 3, 1 / 3],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0.2, 0.3, 0.5]]
    assert np.allclose(expected, sarr.mixture_project(mixtures))

    sarr = rsgame.StratArray(np.array([3, 2, 1]))
    mixtures = [[0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 1],
                [2, 1, 0, 1, 2, 2],
                [1.2, 1.3, 1.5, 0.4, 0.2, 0.3]]
    expected = [[1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 1],
                [0.2, 0.3, 0.5, 0.6, 0.4, 1]]
    assert np.allclose(expected, sarr.mixture_project(mixtures))


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_mixture_project(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    for non_mixture in rand.uniform(-1, 1, (100, sarr.num_strats)):
        new_mix = sarr.mixture_project(non_mixture)
        assert sarr.is_mixture(new_mix), \
            "simplex project did not create a valid mixture"

    mixes = rand.uniform(-1, 1, (10, sarr.num_strats, 10))
    simps = sarr.mixture_project(mixes, 1)
    assert sarr.is_mixture(simps, 1).all()


def test_to_from_simplex():
    sarr = rsgame.StratArray(np.array([2, 2]))
    mixture = [1 / 5, 4 / 5, 1 / 5, 4 / 5]
    simplex = [2 / 15, 2 / 15, 11 / 15]
    assert np.allclose(simplex, sarr.to_simplex(mixture))
    assert np.allclose(mixture, sarr.from_simplex(simplex))

    mixture = [3 / 5, 2 / 5, 1 / 2, 1 / 2]
    simplex = [2 / 5, 1 / 3, 4 / 15]
    assert np.allclose(simplex, sarr.to_simplex(mixture))
    assert np.allclose(mixture, sarr.from_simplex(simplex))

    mixture = [3 / 4, 1 / 4, 1 / 4, 3 / 4]
    simplex = [1 / 2, 1 / 6, 1 / 3]
    assert np.allclose(simplex, sarr.to_simplex(mixture))
    assert np.allclose(mixture, sarr.from_simplex(simplex))

    mixture = [1, 0, 1 / 4, 3 / 4]
    simplex = [1, 0, 0]
    assert np.allclose(simplex, sarr.to_simplex(mixture))
    assert np.allclose(mixture, sarr.from_simplex(simplex))

    assert np.allclose(simplex, sarr.to_simplex(
        [[x] for x in mixture], 0)[:, 0])
    assert np.allclose(mixture, sarr.from_simplex(
        [[x] for x in simplex], 0)[:, 0])


@pytest.mark.parametrize('strats', [1, 2, 4])
def test_random_one_role_to_from_simplex(strats):
    sarr = rsgame.StratArray(np.array([strats]))
    inits = sarr.random_mixtures(100)
    simplicies = sarr.to_simplex(inits)
    assert np.allclose(inits, simplicies)
    mixtures = sarr.to_simplex(inits)
    assert np.allclose(inits, mixtures)


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_uniform_simplex_homotopy(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    uniform = sarr.uniform_mixture()
    simp = sarr.to_simplex(uniform)
    assert np.allclose(simp[0], simp[1:])
    assert np.allclose(uniform, sarr.from_simplex(simp))


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_simplex_homotopy(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    mixes = sarr.random_mixtures(100)

    simp = sarr.to_simplex(mixes[0])
    assert np.all(simp >= 0)
    assert np.isclose(simp.sum(), 1)
    assert np.allclose(mixes[0], sarr.from_simplex(simp))

    simps = sarr.to_simplex(mixes)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(-1), 1)
    assert np.allclose(mixes, sarr.from_simplex(simps))

    mixes = np.rollaxis(mixes, -1, 1)
    simps = sarr.to_simplex(mixes, 1)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(1), 1)
    assert np.allclose(mixes, sarr.from_simplex(simps, 1))


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_uniform_simplex_homotopy(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    rand_mixes = sarr.random_mixtures(100)
    mask = np.repeat(rand.random((100, sarr.num_roles))
                     < 0.5, sarr.num_role_strats, 1)
    mixes = np.where(mask, rand_mixes, sarr.uniform_mixture())

    simp = sarr.to_simplex(mixes[0])
    assert np.all(simp >= 0)
    assert np.isclose(simp.sum(), 1)
    assert np.allclose(mixes[0], sarr.from_simplex(simp))

    simps = sarr.to_simplex(mixes)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(-1), 1)
    assert np.allclose(mixes, sarr.from_simplex(simps))

    mixes = np.rollaxis(mixes, -1, 1)
    simps = sarr.to_simplex(mixes, 1)
    assert np.all(simps >= 0)
    assert np.allclose(simps.sum(1), 1)
    assert np.allclose(mixes, sarr.from_simplex(simps, 1))


def test_uniform_mixture():
    sarr = rsgame.StratArray(np.array([1]))
    assert np.allclose([1], sarr.uniform_mixture())

    sarr = rsgame.StratArray(np.array([3]))
    assert np.allclose([1 / 3] * 3, sarr.uniform_mixture())

    sarr = rsgame.StratArray(np.array([1, 3]))
    assert np.allclose([1] + [1 / 3] * 3, sarr.uniform_mixture())

    sarr = rsgame.StratArray(np.array([3, 2, 1]))
    assert np.allclose([1 / 3] * 3 + [1 / 2] * 2 + [1], sarr.uniform_mixture())


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_mixtures(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    mix = sarr.random_mixtures()
    assert len(mix.shape) == 1

    rand_mixes = sarr.random_mixtures(100)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_mixtures(100, 0.1)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_mixtures(100, 2)
    assert sarr.is_mixture(rand_mixes).all()


def test_random_sparse_mixtures():
    # Technically some of these can fail, but it's extremely unlikely
    sarr = rsgame.StratArray(np.array([3]))
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

    sarr = rsgame.StratArray(np.array([3, 2]))
    mixes = sarr.random_sparse_mixtures(1000, support_prob=[1, 1 / 2])
    assert sarr.is_mixture(mixes).all()
    assert np.all([True, True, True, False, False] == np.all(mixes > 0, 0))


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_random_sparse_mixtures(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    mix = sarr.random_sparse_mixtures()
    assert len(mix.shape) == 1

    rand_mixes = sarr.random_sparse_mixtures(100)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_sparse_mixtures(100, 0.1)
    assert sarr.is_mixture(rand_mixes).all()

    rand_mixes = sarr.random_mixtures(100, 2)
    assert sarr.is_mixture(rand_mixes).all()


def test_biased_mixtures():
    sarr = rsgame.StratArray(np.array([1]))
    expected = [[1]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3]))
    expected = [[0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([1, 3]))
    expected = [[1, 0.8, 0.1, 0.1],
                [1, 0.1, 0.8, 0.1],
                [1, 0.1, 0.1, 0.8]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3, 2, 1]))
    expected = [[0.8, 0.1, 0.1, 0.8, 0.2, 1],
                [0.8, 0.1, 0.1, 0.2, 0.8, 1],
                [0.1, 0.8, 0.1, 0.8, 0.2, 1],
                [0.1, 0.8, 0.1, 0.2, 0.8, 1],
                [0.1, 0.1, 0.8, 0.8, 0.2, 1],
                [0.1, 0.1, 0.8, 0.2, 0.8, 1]]
    actual = sarr.biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


def test_role_biased_mixtures():
    sarr = rsgame.StratArray(np.array([1]))
    expected = [[1]]
    actual = sarr.role_biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3]))
    expected = [[0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]]
    actual = sarr.role_biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([1, 3]))
    expected = [[1, 0.8, 0.1, 0.1],
                [1, 0.1, 0.8, 0.1],
                [1, 0.1, 0.1, 0.8]]
    actual = sarr.role_biased_mixtures(0.8)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3, 2, 1]))
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
    sarr = rsgame.StratArray(np.array([strats]))
    amix = sarr.biased_mixtures(bias)
    bmix = sarr.role_biased_mixtures(bias)
    assert amix.shape == bmix.shape
    assert np.isclose(amix, bmix[:, None]).all(2).any(0).all()


def test_pure_mixtures():
    sarr = rsgame.StratArray(np.array([1]))
    expected = [[1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3]))
    expected = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([1, 3]))
    expected = [[1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3, 2, 1]))
    expected = [[1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 0, 1, 0, 1, 1]]
    actual = sarr.pure_mixtures()
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


def test_grid_mixtures_error():
    sarr = rsgame.StratArray(np.array([1]))
    with pytest.raises(AssertionError):
        sarr.grid_mixtures(1)


def test_grid_mixtures():
    sarr = rsgame.StratArray(np.array([1]))
    expected = [[1]]
    actual = sarr.grid_mixtures(2)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()
    actual = sarr.grid_mixtures(3)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()
    actual = sarr.grid_mixtures(4)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3]))
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

    sarr = rsgame.StratArray(np.array([1, 3]))
    expected = [[1, 0, 0, 1],
                [1, 0, 1 / 2, 1 / 2],
                [1, 0, 1, 0],
                [1, 1 / 2, 0, 1 / 2],
                [1, 1 / 2, 1 / 2, 0],
                [1, 1, 0, 0]]
    actual = sarr.grid_mixtures(3)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()

    sarr = rsgame.StratArray(np.array([3, 2, 1]))
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


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_grid_pure_equivelance(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    expected = sarr.pure_mixtures()
    actual = sarr.grid_mixtures(2)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_fixed_mixtures(_, role_strats):
    sarr = rsgame.StratArray(np.array(role_strats))
    assert sarr.is_mixture(sarr.biased_mixtures()).all()
    assert sarr.is_mixture(sarr.role_biased_mixtures()).all()
    assert sarr.is_mixture(sarr.pure_mixtures()).all()
    assert sarr.is_mixture(sarr.grid_mixtures(3)).all()
    assert sarr.is_mixture(sarr.grid_mixtures(4)).all()


# --------
# BaseGame
# --------


def test_basegame_properties():
    game = rsgame.basegame(1, 1)
    assert np.all(game.num_role_players == [1])
    assert game.num_players == 1
    assert game.zero_prob.shape == (1,)

    game = rsgame.basegame(3, 1)
    assert np.all(game.num_role_players == [3])
    assert game.num_players == 3
    assert game.zero_prob.shape == (1,)

    game = rsgame.basegame([1, 3], 1)
    assert np.all(game.num_role_players == [1, 3])
    assert game.num_players == 4
    assert game.zero_prob.shape == (2,)

    game = rsgame.basegame([3, 2, 1], 1)
    assert np.all(game.num_role_players == [3, 2, 1])
    assert game.num_players == 6
    assert game.zero_prob.shape == (3,)


def test_basegame_min_payoffs():
    with pytest.raises(NotImplementedError):
        rsgame.basegame(1, 1).min_strat_payoffs()
    with pytest.raises(NotImplementedError):
        rsgame.basegame(1, 1).min_role_payoffs()


def test_basegame_max_payoffs():
    with pytest.raises(NotImplementedError):
        rsgame.basegame(1, 1).max_strat_payoffs()
    with pytest.raises(NotImplementedError):
        rsgame.basegame(1, 1).max_role_payoffs()


def test_basegame_deviation_payoffs():
    base = rsgame.basegame(1, 1)
    mix = base.uniform_mixture()
    with pytest.raises(NotImplementedError):
        base.deviation_payoffs(mix)


def test_num_all_profiles():
    game = rsgame.basegame(1, 1)
    assert np.all(game.num_all_role_profiles == [1])
    assert game.num_all_profiles == 1

    game = rsgame.basegame(3, 2)
    assert np.all(game.num_all_role_profiles == [4])
    assert game.num_all_profiles == 4

    game = rsgame.basegame([1, 3], 2)
    assert np.all(game.num_all_role_profiles == [2, 4])
    assert game.num_all_profiles == 8

    game = rsgame.basegame(1, [3, 1])
    assert np.all(game.num_all_role_profiles == [3, 1])
    assert game.num_all_profiles == 3

    game = rsgame.basegame([3, 2, 1], 3)
    assert np.all(game.num_all_role_profiles == [10, 6, 3])
    assert game.num_all_profiles == 180

    game = rsgame.basegame([3, 2, 1], [1, 2, 3])
    assert np.all(game.num_all_role_profiles == [1, 3, 3])
    assert game.num_all_profiles == 9


def test_num_all_payoffs():
    game = rsgame.basegame(1, 1)
    assert game.num_all_payoffs == 1

    game = rsgame.basegame(3, 2)
    assert game.num_all_payoffs == 6

    game = rsgame.basegame([1, 3], 2)
    assert game.num_all_payoffs == 20

    game = rsgame.basegame(1, [3, 1])
    assert game.num_all_payoffs == 6

    game = rsgame.basegame([3, 2, 1], 3)
    assert game.num_all_payoffs == 774

    game = rsgame.basegame([3, 2, 1], [1, 2, 3])
    assert game.num_all_payoffs == 30


def test_num_all_dpr_profiles():
    game = rsgame.basegame(1, 1)
    assert game.num_all_dpr_profiles == 1

    game = rsgame.basegame(3, 2)
    assert game.num_all_dpr_profiles == 6

    game = rsgame.basegame([1, 3], 2)
    assert game.num_all_dpr_profiles == 16

    game = rsgame.basegame(1, [3, 1])
    assert game.num_all_dpr_profiles == 3

    game = rsgame.basegame([3, 2, 1], [1, 2, 3])
    assert game.num_all_dpr_profiles == 15


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_profile_counts(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)

    num_role_profiles = np.fromiter(
        (rsgame.basegame(p, s).all_profiles().shape[0] for p, s
         in zip(game.num_role_players, game.num_role_strats)),
        int, game.num_roles)
    assert np.all(num_role_profiles == game.num_all_role_profiles)

    num_profiles = game.all_profiles().shape[0]
    assert num_profiles == game.num_all_profiles

    num_payoffs = np.sum(game.all_profiles() > 0)
    assert num_payoffs == game.num_all_payoffs

    red = reduction.DeviationPreserving(
        game.num_role_strats, game.num_role_players ** 2,
        game.num_role_players)
    num_dpr_profiles = red.expand_profiles(game.all_profiles()).shape[0]
    assert num_dpr_profiles == game.num_all_dpr_profiles


def test_profile_id():
    game = rsgame.basegame(3, [2, 2])
    profs = [[0, 2],
             [3, 1],
             [2, 3],
             [1, 0]]
    res = game.profile_id(profs, 0)
    assert res.shape == (2,)
    assert np.all((0 <= res) & (res < game.num_all_profiles))


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_profile_id(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)
    expected = np.arange(game.num_all_profiles)
    actual = game.profile_id(game.all_profiles())
    assert not np.setxor1d(expected, actual).size


def test_big_game_functions():
    """Test that everything works when game_size > int max"""
    game = rsgame.basegame([100, 100], [30, 30])
    assert game.num_all_profiles > np.iinfo(int).max
    assert game.num_all_dpr_profiles > np.iinfo(int).max
    assert np.all(game.profile_id(game.random_profiles(1000)) >= 0)


def test_is_profile():
    game = rsgame.basegame([2, 3], [3, 2])
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

    with pytest.raises(AssertionError):
        game.is_profile([0, 0, 0, 0])
    with pytest.raises(AssertionError):
        game.is_profile([0, 0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        game.is_profile([[0, 0, 0, 0, 0, 0]])


def test_all_profiles():
    game = rsgame.basegame(1, 1)
    expected = [[1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.basegame(3, 2)
    expected = [[3, 0],
                [2, 1],
                [1, 2],
                [0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.basegame([1, 3], 2)
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

    game = rsgame.basegame(1, [3, 1])
    expected = [[1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.basegame([3, 2, 1], [1, 2, 3])
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
    game = rsgame.basegame(1, 1)
    expected = [[1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.basegame(3, 2)
    expected = [[3, 0],
                [0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.basegame([1, 3], 2)
    expected = [[1, 0, 3, 0],
                [1, 0, 0, 3],
                [0, 1, 3, 0],
                [0, 1, 0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.basegame(1, [3, 1])
    expected = [[1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.basegame([3, 2, 1], [1, 2, 3])
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
    game = rsgame.basegame(1, 1)
    prof = [1]
    expected = [1]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.basegame(3, 2)
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

    game = rsgame.basegame([1, 3], 2)
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

    game = rsgame.basegame(1, [3, 1])
    prof = [0, 0, 1, 1]
    expected = [[1, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.basegame([3, 2, 1], [1, 2, 3])
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


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
@pytest.mark.parametrize('num_devs', range(5))
def test_random_nearby_profiles(role_players, role_strats, num_devs):
    base = rsgame.game(role_players, role_strats)
    prof = base.random_profiles()
    nearby = base.nearby_profiles(prof, num_devs)
    diff = nearby - prof
    devs_from = np.add.reduceat((diff < 0) * -diff, base.role_starts, 1)
    devs_to = np.add.reduceat((diff > 0) * diff, base.role_starts, 1)
    assert np.all(devs_to.sum(1) <= num_devs)
    assert np.all(devs_from.sum(1) <= num_devs)
    assert np.all(devs_to == devs_from)
    assert np.all(base.is_profile(nearby))


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_fixed_profiles(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)
    all_profiles = game.all_profiles()
    assert game.num_all_profiles == all_profiles.shape[0]
    assert game.is_profile(all_profiles).all()
    pure_profiles = game.pure_profiles()
    assert game.num_pure_subgames == pure_profiles.shape[0]
    assert game.is_profile(pure_profiles).all()


def test_random_profiles():
    game = rsgame.basegame(3, 3)
    mixes = game.random_profiles(100, [0, 0.4, 0.6])
    assert np.all(mixes[:, 0] == 0)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_random_profiles(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)
    assert game.is_profile(game.random_profiles(100)).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_random_dev_profiles(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)
    prof = game.random_dev_profiles(game.uniform_mixture())
    for r, dprof in enumerate(prof):
        role_players = game.num_role_players.copy()
        role_players[r] -= 1
        dgame = rsgame.basegame(role_players, game.num_role_strats)
        assert dgame.is_profile(dprof).all()

    profs = game.random_dev_profiles(game.uniform_mixture(), 100)
    assert profs.shape == (100, game.num_roles, game.num_strats)
    for r, dprofs in enumerate(np.rollaxis(profs, 1, 0)):
        role_players = game.num_role_players.copy()
        role_players[r] -= 1
        dgame = rsgame.basegame(role_players, game.num_role_strats)
        assert dgame.is_profile(dprofs).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_random_deviator_profiles(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)
    profs = game.random_deviator_profiles(game.uniform_mixture(), 100)
    assert profs.shape == (100, game.num_strats, game.num_strats)
    prof_mins = np.minimum.reduceat(
        profs, game.role_starts, 1).repeat(game.num_role_strats, 1)
    prof_devs = profs - prof_mins
    non_strat_roles = np.repeat(
        game.num_role_strats == 1, game.num_role_strats)
    assert np.all(np.any(prof_devs, 1) | non_strat_roles)
    assert np.all(np.any(prof_devs, 2) | non_strat_roles)
    assert game.is_profile(profs).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_max_prob_prof(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)
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


def test_is_symmetric():
    assert rsgame.basegame(3, 4).is_symmetric()
    assert not rsgame.basegame([2, 2], 3).is_symmetric()


def test_is_asymmetric():
    assert rsgame.basegame(1, 4).is_asymmetric()
    assert not rsgame.basegame([1, 2], 3).is_asymmetric()


def test_basegame_hash_eq():
    a = rsgame.basegame(4, 5)
    b = rsgame.basegame([4], [5])
    assert a == b and hash(a) == hash(b)

    a = rsgame.basegame([1, 2], [3, 2])
    b = rsgame.basegame([1, 2], [3, 2])
    assert a == b and hash(a) == hash(b)

    a = rsgame.basegame([2], [3, 2])
    b = rsgame.basegame([2, 2], [3, 2])
    assert a == b and hash(a) == hash(b)

    a = rsgame.basegame([2, 3], [3])
    b = rsgame.basegame([2, 3], [3, 3])
    assert a == b and hash(a) == hash(b)

    assert rsgame.basegame(3, 4) != rsgame.basegame(3, 5)
    assert rsgame.basegame(3, 4) != rsgame.basegame(2, 4)
    assert rsgame.basegame([1, 2], 4) != rsgame.basegame([2, 2], 4)
    assert rsgame.basegame([1, 2], 4) != rsgame.basegame([2, 1], 4)
    assert rsgame.basegame(2, [2, 3]) != rsgame.basegame(2, [2, 2])
    assert rsgame.basegame(2, [2, 3]) != rsgame.basegame(2, [3, 2])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_basegame_copy(role_players, role_strats):
    game = rsgame.basegame(role_players, role_strats)
    copy = rsgame.basegame_copy(game)
    assert game == copy and hash(game) == hash(copy)


def test_basegame_repr():
    game = rsgame.basegame(3, 4)
    expected = 'BaseGame([3], [4])'
    assert repr(game) == expected

    game = rsgame.basegame(3, [4, 5])
    expected = 'BaseGame([3 3], [4 5])'
    assert repr(game) == expected


# ----
# Game
# ----


def test_game_properties():
    game = rsgame.game(1, 1)
    assert np.all(game.profiles == np.empty((0, 1), int))
    assert np.all(game.payoffs == np.empty((0, 1), float))
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    game = rsgame.game(3, 2, [[3, 0]], [[0, 0]])
    assert np.all(game.profiles == [[3, 0]])
    assert np.all(game.payoffs == [[0, 0]])
    assert game.num_profiles == 1
    assert game.num_complete_profiles == 1

    profs = [[1, 0, 3, 0],
             [1, 0, 2, 1]]
    pays = [[0, 0, 0, 0],
            [np.nan, 0, 0, 0]]
    game = rsgame.game([1, 3], 2, profs, pays)
    assert game.profiles.shape == (2, 4)
    assert game.payoffs.shape == (2, 4)
    assert game.num_profiles == 2
    assert game.num_complete_profiles == 1

    game = rsgame.game(1, [3, 1])
    assert game.profiles.shape == (0, 4)
    assert game.payoffs.shape == (0, 4)
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    game = rsgame.game([3, 2, 1], 3)
    assert game.profiles.shape == (0, 9)
    assert game.payoffs.shape == (0, 9)
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    game = rsgame.game([3, 2, 1], [1, 2, 3])
    assert game.profiles.shape == (0, 6)
    assert game.payoffs.shape == (0, 6)
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    with pytest.raises(AssertionError):
        rsgame.game(1, 1, [[1]], [])
    with pytest.raises(AssertionError):
        rsgame.game(1, 1, [[2]], [[0]])
    with pytest.raises(AssertionError):
        rsgame.game(1, 2, [[1]], [[0]])
    with pytest.raises(AssertionError):
        rsgame.game(1, 2, [[2, -1]], [[0, 0]])
    with pytest.raises(AssertionError):
        rsgame.game(1, 2, [[1, 0]], [[0, 1]])
    with pytest.raises(AssertionError):
        rsgame.game(1, 2, [[1, 0], [1, 0]], [[0, 0], [0, 0]])


def test_dev_reps_on_large_games():
    profiles = [[1000, 0], [500, 500]]
    game = rsgame.game(1000, 2, profiles, np.zeros_like(profiles))
    expected = [[0, -np.inf], [688.77411439] * 2]
    assert np.allclose(expected, game._dev_reps)

    profiles = [[12] + [0] * 11, [1] * 12]
    game = rsgame.game(12, 12, profiles, np.zeros_like(profiles))
    expected = [[0] + [-np.inf] * 11, [17.50230785] * 12]
    assert np.allclose(expected, game._dev_reps)

    profiles = [[5] + [0] * 39, ([1] + [0] * 7) * 5]
    game = rsgame.game(5, 40, profiles, np.zeros_like(profiles))
    expected = [[0] + [-np.inf] * 39, ([3.17805383] + [-np.inf] * 7) * 5]
    assert np.allclose(expected, game._dev_reps)

    profiles = [([2] + [0] * 39) * 2,
                [2] + [0] * 39 + ([1] + [0] * 19) * 2,
                ([1] + [0] * 19) * 4]
    game = rsgame.game([2, 2], 40, profiles, np.zeros_like(profiles))
    expected = [([0] + [-np.inf] * 39) * 2,
                [0.69314718] + [-np.inf] * 39 + ([0] + [-np.inf] * 19) * 2,
                ([0.69314718] + [-np.inf] * 19) * 4]
    assert np.allclose(expected, game._dev_reps)


def test_min_max_payoffs():
    game = rsgame.game([2, 2], 2)
    mins = game.min_strat_payoffs()
    assert np.allclose([np.nan] * 4, mins, equal_nan=True)
    mins = game.min_role_payoffs()
    assert np.allclose([np.nan] * 2, mins, equal_nan=True)
    maxs = game.max_strat_payoffs()
    assert np.allclose([np.nan] * 4, maxs, equal_nan=True)
    maxs = game.max_role_payoffs()
    assert np.allclose([np.nan] * 2, maxs, equal_nan=True)

    profs = [[1, 1, 1, 1, 2, 0, 2, 0],
             [2, 0, 2, 0, 2, 0, 2, 0]]
    pays = [[np.nan, 1, 2, np.nan, 3, 0, np.nan, 0],
            [4, 0, 5, 0, 6, 0, np.nan, 0]]
    game = rsgame.game([2] * 4, 2, profs, pays)
    mins = game.min_strat_payoffs()
    assert np.allclose([4, 1, 2, np.nan, 3, np.nan, np.nan, np.nan], mins,
                       equal_nan=True)
    mins = game.min_role_payoffs()
    assert np.allclose([1, 2, 3, np.nan], mins, equal_nan=True)
    maxs = game.max_strat_payoffs()
    assert np.allclose([4, 1, 5, np.nan, 6, np.nan, np.nan, np.nan], maxs,
                       equal_nan=True)
    maxs = game.max_role_payoffs()
    assert np.allclose([4, 5, 6, np.nan], maxs, equal_nan=True)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_min_max_payoffs(role_players, role_strats):
    base = rsgame.basegame(role_players, role_strats)
    profiles = base.all_profiles()
    payoffs = rand.random(profiles.shape)
    mask = profiles > 0
    payoffs *= mask
    game = rsgame.game_copy(base, profiles, payoffs)

    assert (game.payoffs >= game.min_strat_payoffs())[mask].all()
    assert (game.payoffs >= game.min_role_payoffs().repeat(
        game.num_role_strats))[mask].all()
    assert (game.payoffs <= game.max_strat_payoffs())[mask].all()
    assert (game.payoffs <= game.max_role_payoffs().repeat(
        game.num_role_strats))[mask].all()


def test_get_payoffs():
    profs = [[2, 0, 0, 2, 1],
             [1, 1, 0, 0, 3]]
    pays = [[1, 0, 0, 2, 3],
            [4, 5, 0, 0, np.nan]]
    game = rsgame.game([2, 3], [3, 2], profs, pays)

    pay = game.get_payoffs([2, 0, 0, 2, 1])
    assert np.allclose([1, 0, 0, 2, 3], pay)
    pay = game.get_payoffs([1, 1, 0, 0, 3])
    assert np.allclose([4, 5, 0, 0, np.nan], pay, equal_nan=True)
    pay = game.get_payoffs([2, 0, 0, 3, 0])
    assert np.allclose([np.nan, 0, 0, np.nan, 0], pay, equal_nan=True)

    with pytest.raises(AssertionError):
        game.get_payoffs([1, 0, 0, 2, 1])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_get_payoffs(role_players, role_strats):
    base = rsgame.basegame(role_players, role_strats)
    profiles = base.all_profiles()
    payoffs = rand.random(profiles.shape)
    payoffs *= profiles > 0
    game = rsgame.game_copy(base, profiles, payoffs)

    for prof, pay in zip(profiles, payoffs):
        assert np.allclose(pay, game.get_payoffs(prof))


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_empty_get_payoffs(role_players, role_strats):
    game = rsgame.game(role_players, role_strats)

    for prof in game.all_profiles():
        supp = prof > 0
        pay = game.get_payoffs(prof)
        assert np.isnan(pay[supp]).all()
        assert np.all(pay[~supp] == 0)


def test_deviation_mixture_support():
    base = rsgame.basegame([2, 2], 3)
    profiles1 = [
        [2, 0, 0, 2, 0, 0],
        [1, 1, 0, 2, 0, 0],
        [0, 2, 0, 2, 0, 0],
    ]
    payoffs1 = [
        [1, 0, 0, 2, 0, 0],
        [3, 4, 0, 5, 0, 0],
        [0, 6, 0, 7, 0, 0],
    ]
    profiles2 = [
        [2, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 2, 0, 1, 1, 0],
    ]
    payoffs2 = [
        [8, 0, 0, 9, 10, 0],
        [11, 12, 0, 13, 14, 0],
        [0, 15, 0, 16, 17, 0],
    ]
    game1 = rsgame.game_copy(base, profiles1, payoffs1)
    game2 = rsgame.game_copy(base, profiles2, payoffs2)
    game3 = rsgame.game_copy(base, profiles1 + profiles2, payoffs1 + payoffs2)
    mix1 = [0.5, 0.5, 0, 0.3, 0.7, 0]
    mix2 = [0.5, 0.5, 0, 1, 0, 0]

    dev_payoffs = game1.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game1.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs, [2, 5, np.nan, 4.75, np.nan, np.nan],
                       equal_nan=True)
    dev_payoffs = game2.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game2.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs,
                       [np.nan, np.nan, np.nan, np.nan, 13.75, np.nan],
                       equal_nan=True)
    dev_payoffs = game3.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game3.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs, [2, 5, np.nan, 4.75, 13.75, np.nan],
                       equal_nan=True)


# Test sample game with different number of samples
def test_different_samples():
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6], [0, 0], [2, 3]],
        ],
    ]
    game = rsgame.samplegame(1, [1, 2], profiles, payoffs)

    assert np.all([1, 2] == game.num_samples), \
        "didn't get both sample sizes"
    assert repr(game) is not None


def test_deviation_payoffs_jacobian():
    game = rsgame.game(2, 3)
    eqm = np.ones(3) / 3
    dp, dpj = game.deviation_payoffs(eqm, jacobian=True)
    assert np.isnan(dp).all()
    assert np.isnan(dpj).all()

    profiles = [[2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]]
    payoffs = [[0, 0, 0],
               [-1, 1, 0],
               [1, 0, -1],
               [0, 0, 0],
               [0, -1, 1],
               [0, 0, 0]]
    game = rsgame.game(2, 3, profiles, payoffs)
    eqm = np.ones(3) / 3
    dp, dpj = game.deviation_payoffs(eqm, jacobian=True)
    assert np.allclose(dp, 0)
    expected_jac = np.array([[0, -1, 1],
                             [1, 0, -1],
                             [-1, 1, 0]])
    assert np.allclose(dpj, expected_jac)


def test_flat_profile_payoffs():
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6], [0, 0], [2, 3]],
        ],
    ]
    game = rsgame.samplegame(1, [1, 2], profiles, payoffs)

    expected_profs = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
    ])
    expected_pays = np.array([
        [5, 2, 0],
        [5, 0, 2],
        [6, 0, 3],
    ], float)

    assert np.all(game.flat_profiles[np.lexsort(game.flat_profiles.T)] ==
                  expected_profs[np.lexsort(expected_profs.T)])
    assert np.allclose(game.flat_payoffs[np.lexsort(game.flat_payoffs.T)],
                       expected_pays[np.lexsort(expected_pays.T)])


def test_nan_mask_for_dev_payoffs():
    profiles = [[3, 0, 0, 0],
                [2, 1, 0, 0],
                [2, 0, 1, 0]]
    payoffs = [[1, 0, 0, 0],
               [np.nan, 2, 0, 0],
               [5, 0, np.nan, 0]]
    game = rsgame.game([3], [4], profiles, payoffs)
    devs = game.deviation_payoffs([1, 0, 0, 0])
    assert np.allclose(devs, [1, 2, np.nan, np.nan], equal_nan=True)

    devs = game.deviation_payoffs([1, 0, 0, 0], assume_complete=True)
    assert np.allclose(devs, [1, 2, np.nan, 0], equal_nan=True)


def test_nan_payoffs_for_dev_payoffs():
    profiles = [[3, 0, 3, 0],
                [2, 1, 3, 0],
                [3, 0, 2, 1]]
    payoffs = [[1, 0, 2, 0],
               [np.nan, 3, np.nan, 0],
               [np.nan, 0, np.nan, 4]]
    game = rsgame.game([3, 3], [2, 2], profiles, payoffs)
    devs = game.deviation_payoffs([1, 0, 1, 0])
    assert np.allclose(devs, [1, 3, 2, 4])


@pytest.mark.parametrize('p', [2, 5, 10, 100])
def test_deviation_nans(p):
    profiles = [[p,     0, 0, 0, 1],
                [p - 1, 1, 0, 0, 1],
                [p - 1, 0, 1, 0, 1],
                [p - 1, 0, 0, 1, 1]]
    payoffs = [[1,      0, 0, 0, 2],
               [np.nan, 3, 0, 0, np.nan],
               [np.nan, 0, 4, 0, np.nan],
               [np.nan, 0, 0, 5, np.nan]]
    game = rsgame.game([p, 1], [4, 1], profiles, payoffs)
    mix = np.array([1, 0, 0, 0, 1])
    pays = game.deviation_payoffs(mix)
    assert not np.isnan(pays).any()


@pytest.mark.parametrize('p', [2, 5, 10, 100])
@pytest.mark.parametrize('q', [2, 5, 10, 100])
def test_deviation_nans_2(p, q):
    profiles = [[p,     0, 0, 0, q,     0],
                [p - 1, 1, 0, 0, q,     0],
                [p - 1, 0, 1, 0, q,     0],
                [p - 1, 0, 0, 1, q,     0],
                [p,     0, 0, 0, q - 1, 1]]
    payoffs = [[1,      0, 0, 0, 2,      0],
               [np.nan, 3, 0, 0, np.nan, 0],
               [np.nan, 0, 4, 0, np.nan, 0],
               [np.nan, 0, 0, 5, np.nan, 0],
               [6,      0, 0, 0, np.nan, 7]]
    game = rsgame.game([p, q], [4, 2], profiles, payoffs)
    mix = np.array([1, 0, 0, 0, 1, 0])
    pays = game.deviation_payoffs(mix)
    assert not np.isnan(pays).any()


def test_expected_payoffs():
    game = rsgame.game(2, [2, 2])
    pays = game.get_expected_payoffs([0.2, 0.8, 0.4, 0.6])
    assert np.allclose([np.nan, np.nan], pays, equal_nan=True)

    profs = [[2, 0],
             [1, 1],
             [0, 2]]
    pays = [[1, 0],
            [2, 3],
            [0, 4]]
    game = rsgame.game(2, 2, profs, pays)
    pays = game.get_expected_payoffs([0.2, 0.8])
    assert np.allclose(3.4, pays)

    profs = [[2, 0, 2, 0],
             [2, 0, 1, 1],
             [2, 0, 0, 2],
             [1, 1, 2, 0],
             [1, 1, 1, 1],
             [1, 1, 0, 2],
             [0, 2, 2, 0],
             [0, 2, 1, 1],
             [0, 2, 0, 2]]
    pays = [[1, 0, 2, 0],
            [3, 0, 4, 5],
            [6, 0, 0, 7],
            [8, 9, 10, 0],
            [11, 12, 13, 14],
            [15, 16, 0, 17],
            [0, 18, 19, 0],
            [0, 20, 21, 22],
            [0, 23, 0, 24]]
    game = rsgame.game([2, 2], [2, 2], profs, pays)
    pays = game.get_expected_payoffs([0.2, 0.8, 0.4, 0.6])
    assert np.allclose([17.424, 18.824], pays)

    profs = [[2, 0, 2, 0],
             [2, 0, 1, 1],
             [2, 0, 0, 2]]
    pays = [[1, 0, 2, 0],
            [3, 0, 4, 5],
            [6, 0, 0, 7]]
    game = rsgame.game([2, 2], [2, 2], profs, pays)
    pays = game.get_expected_payoffs([0.2, 0.8, 0.4, 0.6])
    assert np.allclose([np.nan, np.nan], pays, equal_nan=True)
    pays = game.get_expected_payoffs([1, 0, 0.4, 0.6])
    assert np.allclose([3.76, 5], pays)

    profs = [[2, 0, 2, 0],
             [2, 0, 1, 1],
             [2, 0, 0, 2]]
    pays = [[1, 0, 2, 0],
            [3, 0, 4, 5],
            [np.nan, 0, 0, 7]]
    game = rsgame.game([2, 2], [2, 2], profs, pays)
    pays = game.get_expected_payoffs([1, 0, 0.4, 0.6])
    assert np.allclose([np.nan, 5], pays, equal_nan=True)


def test_expected_payoffs_jac():
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[1, 0],
               [3, 3],
               [0, 1]]
    game = rsgame.game(2, 2, profiles, payoffs)
    ep, ep_jac = game.get_expected_payoffs([.5, .5], jacobian=True)
    ep_jac -= ep_jac.sum() / 2  # project on simplex
    assert np.allclose(ep, 2)
    assert np.allclose(ep_jac, 0), \
        "maximum surplus should have 0 jacobian"

    dev_data = game.deviation_payoffs([0.5, 0.5], jacobian=True)
    ep, ep_jac = game.get_expected_payoffs([.5, .5], jacobian=True,
                                           deviations=dev_data)
    ep_jac -= ep_jac.sum() / 2  # project on simplex
    assert np.allclose(ep, 2)
    assert np.allclose(ep_jac, 0), \
        "maximum surplus should have 0 jacobian"


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_is_empty(role_players, role_strats):
    game = rsgame.game(role_players, role_strats)
    assert game.is_empty()

    game = rsgame.game_copy(rsgame.basegame(role_players, role_strats))
    assert game.is_empty()

    game = rsgame.game_copy(game, np.empty((0, game.num_strats), int),
                            np.empty((0, game.num_strats)))
    assert game.is_empty()

    game = rsgame.game_copy(game, game.random_profiles()[None],
                            np.zeros((1, game.num_strats)))
    assert not game.is_empty()


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_is_complete(role_players, role_strats):
    game = rsgame.game(role_players, role_strats)
    assert not game.is_complete()

    game = rsgame.game_copy(game, game.all_profiles(),
                            np.zeros((game.num_all_profiles, game.num_strats)))
    assert game.is_complete()

    game = rsgame.game_copy(game, game.profiles[1:], game.payoffs[1:])
    assert not game.is_complete()


def test_is_constant_sum():
    game = rsgame.game(2, 3)
    assert game.is_constant_sum()

    profiles = [
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
    ]
    payoffs = [
        [2, 0, -2, 0],
        [3, 0, 0, -3],
        [0, 5, -5, 0],
        [0, 1, 0, -1],
    ]
    game = rsgame.game(1, [2, 2], profiles, payoffs)
    assert game.is_constant_sum()

    payoffs = game.payoffs.copy()
    payoffs[game.profiles > 0] += 1
    game = rsgame.game_copy(game, game.profiles, payoffs)
    assert game.is_constant_sum()

    profiles = [
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
    ]
    payoffs = [
        [1, 0, 2, 0],
        [3, 0, 0, 4],
        [0, 5, 6, 0],
        [0, 7, 0, 8],
    ]
    game = rsgame.game_copy(game, profiles, payoffs)
    assert not game.is_constant_sum()


def test_contains():
    profs = [[2, 0, 0, 2, 1],
             [1, 1, 0, 0, 3]]
    pays = [[1, 0, 0, 2, 3],
            [4, 5, 0, 0, np.nan]]
    game = rsgame.game([2, 3], [3, 2], profs, pays)
    assert [2, 0, 0, 2, 1] in game
    assert [1, 1, 0, 0, 3] not in game
    assert [1, 1, 0, 2, 1] not in game


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_contains(role_players, role_strats):
    game = rsgame.game(role_players, role_strats)
    for prof in game.all_profiles():
        assert prof not in game

    game = rsgame.game_copy(game, game.all_profiles(),
                            np.zeros((game.num_all_profiles, game.num_strats)))
    for prof in game.all_profiles():
        assert prof in game


def test_game_hash_eq():
    a = rsgame.game(4, 5)
    b = rsgame.game([4], [5])
    assert a == b and hash(a) == hash(b)

    a = rsgame.game(4, 2, [[3, 1], [2, 2]], [[1, 2], [3, 4]])
    b = rsgame.game([4], [2], [[2, 2], [3, 1]], [[3, 4], [1, 2]])
    assert a == b and hash(a) == hash(b)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_game_copy(role_players, role_strats):
    base = rsgame.basegame(role_players, role_strats)
    profs = base.all_profiles()
    rand.shuffle(profs)
    num_profs = rand.randint(0, base.num_all_profiles + 1)
    profs = profs[:num_profs].copy()
    pays = rand.random(profs.shape)
    pays *= profs > 0
    game = rsgame.game_copy(base, profs, pays)

    copy = rsgame.game_copy(game)
    assert game == copy and hash(game) == hash(copy)

    perm = rand.permutation(num_profs)
    copy = rsgame.game_copy(game, game.profiles[perm], game.payoffs[perm])
    assert game == copy and hash(game) == hash(copy)


def test_game_repr():
    game = rsgame.game(3, 4)
    expected = 'Game([3], [4], 0 / 20)'
    assert repr(game) == expected

    game = rsgame.basegame(3, [4, 5])
    game = rsgame.game_copy(game, game.all_profiles()[:21],
                            np.zeros((21, game.num_strats)))
    expected = 'Game([3 3], [4 5], 21 / 700)'
    assert repr(game) == expected


# ----------
# SampleGame
# ----------


def test_samplegame_properties():
    game = rsgame.samplegame(2, 3)
    assert np.all([] == game.num_sample_profs)
    assert np.all([] == game.sample_starts)
    assert np.all([] == game.num_samples)

    base = rsgame.basegame(1, [4, 3])
    game = rsgame.samplegame_copy(
        base, base.all_profiles(), [np.zeros((12, 7, 2))])
    assert np.all([12] == game.num_sample_profs)
    assert np.all([0] == game.sample_starts)
    assert np.all([2] == game.num_samples)

    game = rsgame.basegame([3, 4], [4, 3])
    profiles = game.all_profiles()[:30]
    spays = [np.zeros((9, game.num_strats, 4)),
             np.zeros((11, game.num_strats, 1)),
             np.zeros((10, game.num_strats, 2))]
    game = rsgame.samplegame_copy(game, profiles, spays)
    assert np.all([9, 11, 10] == game.num_sample_profs)
    assert np.all([0, 9, 20] == game.sample_starts)
    assert np.all([4, 1, 2] == game.num_samples)


def test_empty_samplegame_resample():
    sgame = rsgame.samplegame([2, 3], [3, 2])
    assert rsgame.game_copy(sgame) == sgame.resample()
    assert rsgame.game_copy(sgame) == sgame.resample(1)

    sgame = rsgame.samplegame_copy(rsgame.basegame([2, 3], [3, 2]))
    assert rsgame.game_copy(sgame) == sgame.resample()
    assert rsgame.game_copy(sgame) == sgame.resample(1)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_samplegame_singlesample_resample(role_players, role_strats):
    base = rsgame.basegame(role_players, role_strats)
    profs = base.all_profiles()
    pays = rand.random(profs.shape)
    pays *= profs > 0
    sgame = rsgame.samplegame_copy(rsgame.game_copy(base, profs, pays))
    copy = rsgame.game_copy(sgame)

    game = sgame.resample()
    assert game == copy

    game = sgame.resample(100)
    assert game == copy

    game = sgame.resample(independent_role=True)
    assert game == copy
    game = sgame.resample(independent_strategy=True)
    assert game == copy
    game = sgame.resample(independent_profile=True)
    assert game == copy
    game = sgame.resample(independent_role=True, independent_strategy=True)
    assert game == copy
    game = sgame.resample(independent_role=True, independent_profile=True)
    assert game == copy
    game = sgame.resample(independent_strategy=True, independent_profile=True)
    assert game == copy
    game = sgame.resample(independent_role=True, independent_strategy=True,
                          independent_profile=True)
    assert game == copy


def test_samplegame_resample_changes():
    base = rsgame.basegame(1, [3, 2])
    profiles = base.all_profiles()
    payoffs = rand.random(profiles.shape + (1000,))
    view = payoffs.view()
    view.shape = (-1, 1000)
    view[profiles.ravel() == 0] = 0
    sgame = rsgame.samplegame_copy(base, profiles, [payoffs])
    copy = rsgame.game_copy(sgame)

    # These aren't guaranteed to be true, but they're highly unlikely
    game = sgame.resample()
    assert game != copy

    game = sgame.resample(100)
    assert game != copy

    game = sgame.resample(independent_role=True)
    assert game != copy
    game = sgame.resample(independent_strategy=True)
    assert game != copy
    game = sgame.resample(independent_profile=True)
    assert game != copy
    game = sgame.resample(independent_role=True, independent_strategy=True)
    assert game != copy
    game = sgame.resample(independent_role=True, independent_profile=True)
    assert game != copy
    game = sgame.resample(independent_strategy=True, independent_profile=True)
    assert game != copy
    game = sgame.resample(independent_role=True, independent_strategy=True,
                          independent_profile=True)
    assert game != copy


def test_get_sample_payoffs():
    base = rsgame.basegame(2, [1, 2])
    profiles = [
        [2, 2, 0],
        [2, 0, 2],
    ]
    spayoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6], [0, 0], [2, 3]],
        ],
    ]
    game = rsgame.samplegame_copy(base, profiles, spayoffs)
    pay = game.get_sample_payoffs([2, 1, 1])
    assert np.allclose(np.empty((0, 3)), pay)
    pay = game.get_sample_payoffs([2, 2, 0])
    assert np.allclose([[5, 2, 0]], pay)
    pay = game.get_sample_payoffs([2, 0, 2])
    assert np.allclose([[5, 0, 2], [6, 0, 3]], pay)

    with pytest.raises(AssertionError):
        game.get_sample_payoffs([2, 1, 2])
    with pytest.raises(AssertionError):
        game.get_sample_payoffs([2, 0, 2, 0])


def test_samplegame_hash_eq():
    a = rsgame.samplegame(4, 5)
    b = rsgame.samplegame([4], [5])
    assert a == b and hash(a) == hash(b)

    a = rsgame.samplegame(
        4, 2,
        [[3, 1], [2, 2]],
        [[[[1], [2]]], [[[3, 5], [4, 6]]]])
    b = rsgame.samplegame(
        [4], [2],
        [[2, 2], [3, 1]],
        [[[[5, 3], [6, 4]]], [[[1], [2]]]])
    assert a == b and hash(a) == hash(b)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_samplegame_copy(role_players, role_strats):
    base = rsgame.basegame(role_players, role_strats)
    profs = base.all_profiles()
    rand.shuffle(profs)
    spays = []
    start = 0
    for n, c in collections.Counter(
            rand.geometric(.5, base.num_all_profiles) - 1).items():
        if n == 0:
            profs = np.delete(profs, slice(start, start + c), 0)
        else:
            sprofs = profs[start:start + c]
            start += c
            pays = rand.random((c * base.num_strats, n))
            pays[sprofs.ravel() == 0] = 0
            spays.append(pays.reshape((c, base.num_strats, n)))

    game = rsgame.samplegame_copy(base, profs, spays)

    copy = rsgame.samplegame_copy(game)
    assert game == copy and hash(game) == hash(copy)

    samp_perm = rand.permutation(game.num_samples.size)
    prof_list = np.split(game.profiles, game.sample_starts[1:], 0)
    sprofs = []
    spays = []
    for i in samp_perm:
        perm = rand.permutation(game.num_sample_profs[i])
        sprofs.append(prof_list[i][perm])
        spays.append(game.sample_payoffs[i][perm])

    profiles = np.concatenate(
        sprofs) if sprofs else np.empty((0, game.num_strats))
    copy = rsgame.samplegame_copy(game, profiles, spays)
    assert game == copy and hash(game) == hash(copy)


# Test sample game with different number of samples
def test_samplegame_different_samples():
    base = rsgame.basegame(1, [1, 2])
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6], [0, 0], [2, 3]],
        ],
    ]
    sgame = rsgame.samplegame_copy(base, profiles, payoffs)
    game = rsgame.game_copy(sgame)

    assert not np.setxor1d([1, 2], sgame.num_samples).size
    # This could technically fail, but it's extremely unlikely
    assert any(game != sgame.resample() for _ in range(1000))


def test_samplegame_repr():
    game = rsgame.samplegame(2, 3)
    expected = 'SampleGame([2], [3], 0 / 6, 0)'
    assert repr(game) == expected

    base = rsgame.basegame(1, [4, 3])
    game = rsgame.samplegame_copy(
        base, base.all_profiles(), [np.zeros((12, 7, 2))])
    expected = 'SampleGame([1 1], [4 3], 12 / 12, 2)'
    assert repr(game) == expected

    base = rsgame.basegame(1, [1, 2])
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6], [0, 0], [2, 3]],
        ],
    ]
    game = rsgame.samplegame_copy(base, profiles, payoffs)
    expected = 'SampleGame([1 1], [1 2], 2 / 2, 1 - 2)'
    assert repr(game) == expected

    payoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6, 7], [0, 0, 0], [2, 3, 4]],
        ],
    ]
    game = rsgame.samplegame_copy(base, profiles, payoffs)
    expected = 'SampleGame([1 1], [1 2], 2 / 2, 1 - 3)'
    assert repr(game) == expected
