import itertools
import json
import warnings

import numpy as np
import numpy.random as rand
import pytest
import scipy.special as sps

from gameanalysis import rsgame
from gameanalysis import utils
from test import utils as testutils


TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps


# ----------
# StratArray
# ----------


def stratarray(num_strats):
    roles = tuple(chr(97 + i) for i in range(len(num_strats)))
    sit = (chr(97 + i) for i in range(sum(num_strats)))  # pragma: no branch
    strats = tuple(tuple(itertools.islice(sit, s)) for s in num_strats)
    return rsgame.StratArray(roles, strats)


def test_stratarray_properties():
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


def test_restriction_enumeration():
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

    with pytest.raises(AssertionError):
        sarr.is_restriction([False, False, False, False])
    with pytest.raises(AssertionError):
        sarr.is_restriction([False, False, False, False, False, False])
    with pytest.raises(AssertionError):
        sarr.is_restriction([[False, False, False, False, False, False]])


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_stratarray_restrictions(_, role_strats):
    sarr = stratarray(role_strats)
    all_restrictions = sarr.all_restrictions()
    assert sarr.is_restriction(all_restrictions).all()
    assert sarr.num_all_restrictions == all_restrictions.shape[0]
    pure_restrictions = sarr.pure_restrictions()
    assert sarr.is_restriction(pure_restrictions).all()
    assert sarr.num_pure_restrictions == pure_restrictions.shape[0]


def test_random_restrictions():
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


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_random_restrictions(_, role_strats):
    sarr = stratarray(role_strats)
    rest = sarr.random_restriction()
    assert len(rest.shape) == 1

    rests = sarr.random_restrictions(100)
    assert sarr.is_restriction(rests).all()


def test_trim_mixture_support():
    sarr = stratarray([3])
    mix = np.array([0.7, 0.3, 0])
    not_trimmed = sarr.trim_mixture_support(mix, thresh=0.1)
    assert np.allclose(mix, not_trimmed)
    trimmed = sarr.trim_mixture_support(mix, thresh=0.4)
    assert np.allclose([1, 0, 0], trimmed)

    trimmed = sarr.trim_mixture_support(mix[:, None], thresh=0.4, axis=0)[:, 0]
    assert np.allclose([1, 0, 0], trimmed)


def test_is_mixture():
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

    with pytest.raises(AssertionError):
        sarr.is_mixture([0, 0, 0, 0])
    with pytest.raises(AssertionError):
        sarr.is_mixture([0, 0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        sarr.is_mixture([[0, 0, 0, 0, 0, 0]])


def test_mixture_project():
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


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_mixture_project(_, role_strats):
    sarr = stratarray(role_strats)
    for non_mixture in rand.uniform(-1, 1, (100, sarr.num_strats)):
        new_mix = sarr.mixture_project(non_mixture)
        assert sarr.is_mixture(new_mix), \
            "simplex project did not create a valid mixture"

    mixes = rand.uniform(-1, 1, (10, 12, sarr.num_strats))
    simps = sarr.mixture_project(mixes)
    assert simps.shape[:2] == (10, 12)
    assert sarr.is_mixture(simps).all()


def test_to_from_simplex():
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
    sarr = stratarray([strats])
    inits = sarr.random_mixtures(100)
    simplicies = sarr.mixture_to_simplex(inits)
    assert np.allclose(inits, simplicies)
    mixtures = sarr.mixture_to_simplex(inits)
    assert np.allclose(inits, mixtures)


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_uniform_simplex_homotopy(_, role_strats):
    sarr = stratarray(role_strats)
    uniform = sarr.uniform_mixture()
    simp = sarr.mixture_to_simplex(uniform)
    assert np.allclose(simp[0], simp[1:])
    assert np.allclose(uniform, sarr.mixture_from_simplex(simp))


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_simplex_homotopy(_, role_strats):
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


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_uniform_simplex_homotopy(_, role_strats):
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
    sarr = stratarray([1])
    assert np.allclose([1], sarr.uniform_mixture())

    sarr = stratarray([3])
    assert np.allclose([1 / 3] * 3, sarr.uniform_mixture())

    sarr = stratarray([1, 3])
    assert np.allclose([1] + [1 / 3] * 3, sarr.uniform_mixture())

    sarr = stratarray([3, 2, 1])
    assert np.allclose([1 / 3] * 3 + [1 / 2] * 2 + [1], sarr.uniform_mixture())


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_mixtures(_, role_strats):
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


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_random_sparse_mixtures(_, role_strats):
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
    sarr = stratarray([strats])
    amix = sarr.biased_mixtures(bias)
    bmix = sarr.role_biased_mixtures(bias)
    assert amix.shape == bmix.shape
    assert np.isclose(amix, bmix[:, None]).all(2).any(0).all()


def test_pure_mixtures():
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
    sarr = stratarray([1])
    with pytest.raises(AssertionError):
        sarr.grid_mixtures(1)


def test_grid_mixtures():
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


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_grid_pure_equivelance(_, role_strats):
    sarr = stratarray(role_strats)
    expected = sarr.pure_mixtures()
    actual = sarr.grid_mixtures(2)
    assert np.isclose(expected, actual[:, None]).all(2).any(0).all()


@pytest.mark.parametrize('_,role_strats', testutils.games)
def test_random_fixed_mixtures(_, role_strats):
    sarr = stratarray(role_strats)
    assert sarr.is_mixture(sarr.biased_mixtures()).all()
    assert sarr.is_mixture(sarr.role_biased_mixtures()).all()
    assert sarr.is_mixture(sarr.pure_mixtures()).all()
    assert sarr.is_mixture(sarr.grid_mixtures(3)).all()
    assert sarr.is_mixture(sarr.grid_mixtures(4)).all()


def test_strat_name():
    sarr = stratarray([3, 2])
    for i, s in enumerate(['a', 'b', 'c', 'd', 'e']):
        assert s == sarr.strat_name(i)


def test_indices():
    sarr = stratarray([3, 2])
    assert 0 == sarr.role_index('a')
    assert 1 == sarr.role_index('b')
    assert 0 == sarr.role_strat_index('a', 'a')
    assert 1 == sarr.role_strat_index('a', 'b')
    assert 2 == sarr.role_strat_index('a', 'c')
    assert 3 == sarr.role_strat_index('b', 'd')
    assert 4 == sarr.role_strat_index('b', 'e')
    assert 0 == sarr.role_strat_dev_index('a', 'a', 'b')
    assert 1 == sarr.role_strat_dev_index('a', 'a', 'c')
    assert 2 == sarr.role_strat_dev_index('a', 'b', 'a')
    assert 3 == sarr.role_strat_dev_index('a', 'b', 'c')
    assert 4 == sarr.role_strat_dev_index('a', 'c', 'a')
    assert 5 == sarr.role_strat_dev_index('a', 'c', 'b')
    assert 6 == sarr.role_strat_dev_index('b', 'd', 'e')
    assert 7 == sarr.role_strat_dev_index('b', 'e', 'd')
    rs_names = (('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'd'), ('b', 'e'))
    assert rs_names == sarr.role_strat_names


def test_to_mixture_from_json():
    sarr = stratarray([2, 1])
    mix = [.6, .4, 1]
    json_mix = {'a': {'b': .4, 'a': .6}, 'b': {'c': 1}}
    assert sarr.mixture_to_json(mix) == json_mix
    new_mix = sarr.mixture_from_json(json_mix)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float
    new_mix.fill(0)
    sarr.mixture_from_json(json_mix, dest=new_mix)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_mixture_serialization(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    mixes = game.random_mixtures(20)
    copies = np.empty(mixes.shape)
    for mix, copy in zip(mixes, copies):
        jmix = json.dumps(game.mixture_to_json(mix))
        game.mixture_from_json(json.loads(jmix), copy)
    # 1e-4 is the tolerance for serialization
    assert np.allclose(copies, mixes, atol=1e-4)


def test_to_from_mix_repr():
    sarr = stratarray([2, 1])
    mix = [.6, .4, 1]
    expected = "a: 60.00% a, 40.00% b; b: 100.00% c"
    assert sarr.mixture_to_repr(mix) == expected
    new_mix = sarr.mixture_from_repr(expected)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float
    new_mix.fill(0)
    sarr.mixture_from_repr(expected, dest=new_mix)
    assert np.allclose(new_mix, mix)
    assert new_mix.dtype == float


def test_to_from_mix_str():
    sarr = stratarray([2, 1])
    mix = [0.3, 0.7, 1]
    expected = """
a:
    a:  30.00%
    b:  70.00%
b:
    c: 100.00%
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
    sarr = stratarray([2, 1])
    sub = [True, False, True]
    json_sub = {'a': ['a'], 'b': ['c']}
    assert sarr.restriction_to_json(sub) == json_sub
    new_sub = sarr.restriction_from_json(json_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool
    new_sub.fill(False)
    sarr.restriction_from_json(json_sub, dest=new_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_restriction_serialization(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    subs = game.random_restrictions(20)
    copies = np.empty(subs.shape, bool)
    for sub, copy in zip(subs, copies):
        jsub = json.dumps(game.restriction_to_json(sub))
        game.restriction_from_json(json.loads(jsub), copy)
    assert np.all(copies == subs)


def test_to_from_restriction_repr():
    sarr = stratarray([2, 1])
    sub = [True, False, True]
    sub_repr = "a: a; b: c"
    assert sarr.restriction_to_repr(sub) == sub_repr
    new_sub = sarr.restriction_from_repr(sub_repr)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool
    new_sub.fill(False)
    sarr.restriction_from_repr(sub_repr, dest=new_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool


def test_to_from_restriction_str():
    sarr = stratarray([2, 1])
    sub = [True, False, True]
    sub_str = """
a:
    a
b:
    c
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
    sarr = stratarray([2, 1])
    role = [6, 3]
    json_role = {'a': 6, 'b': 3}
    assert sarr.role_to_json(role) == json_role
    arr = sarr.role_from_json(json_role)
    assert np.allclose(arr, role)
    assert arr.dtype == float
    arr = np.empty_like(arr)
    sarr.role_from_json(json_role, dest=arr)
    assert np.allclose(arr, role)
    assert arr.dtype == float


def test_to_from_role_repr():
    sarr = stratarray([2, 1])
    role = [6, 3]
    rep_role = 'a: 6; b: 3'
    assert sarr.role_to_repr(role) == rep_role
    arr = sarr.role_from_repr(rep_role)
    assert np.allclose(arr, role)
    assert arr.dtype == float
    arr = np.empty_like(arr)
    sarr.role_from_repr(rep_role, dest=arr)
    assert np.allclose(arr, role)
    assert arr.dtype == float


def test_trim_precision():
    sarr = stratarray([3, 2])
    trimmed = sarr.trim_mixture_precision(
        [1/3, 1/3, 1/3, 0.62, 0.38], resolution=0.1)
    # Ties resolve as first strategies
    assert np.allclose(trimmed, [0.4, 0.3, 0.3, 0.6, 0.4])

    trimmed = sarr.trim_mixture_precision(
        [0.5, 0.25, 0.25, 0.5, 0.5], resolution=0.05)
    assert np.allclose(trimmed, [0.5, 0.25, 0.25, 0.5, 0.5])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_role_serialization(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
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
    game = rsgame.emptygame(1, 1)
    assert np.all(game.num_role_players == [1])
    assert game.num_players == 1
    assert game.zero_prob.shape == (1,)

    game = rsgame.emptygame(3, 1)
    assert np.all(game.num_role_players == [3])
    assert game.num_players == 3
    assert game.zero_prob.shape == (1,)

    game = rsgame.emptygame([1, 3], 1)
    assert np.all(game.num_role_players == [1, 3])
    assert game.num_players == 4
    assert game.zero_prob.shape == (2,)

    game = rsgame.emptygame([3, 2, 1], 1)
    assert np.all(game.num_role_players == [3, 2, 1])
    assert game.num_players == 6
    assert game.zero_prob.shape == (3,)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_emptygame_const_properties(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)

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
    assert np.isnan(pays[0 < prof]).all()
    assert np.all(pays[0 == prof] == 0)
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

    exp_pays = game.expected_payoffs(mix, deviations=dev_pays)
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)

    dev_pays, dev_jac = game.deviation_payoffs(mix, jacobian=True)
    assert np.isnan(dev_pays).all()
    assert dev_pays.shape == (game.num_strats,)
    assert np.isnan(dev_jac).all()
    assert dev_jac.shape == (game.num_strats, game.num_strats)

    exp_pays, exp_jac = game.expected_payoffs(mix, jacobian=True)
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)
    assert np.isnan(exp_jac).all()
    assert exp_jac.shape == (game.num_roles, game.num_strats)

    exp_pays, exp_jac = game.expected_payoffs(
        mix, jacobian=True, deviations=(dev_pays, dev_jac))
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)
    assert np.isnan(exp_jac).all()
    assert exp_jac.shape == (game.num_roles, game.num_strats)

    br = game.best_response(mix)
    assert np.isnan(br).all()
    assert br.shape == (game.num_strats,)

    assert game.profiles().size == 0
    assert game.payoffs().size == 0

    assert game.is_empty()
    assert not game.is_complete()
    assert game.is_constant_sum()

    assert game.normalize() == game
    assert game.random_profile() not in game


def test_empty_restriction():
    game = rsgame.emptygame(1, [2, 3])
    rgame = game.restrict([False, True, True, False, True])
    expected = rsgame.emptygame_names(('r0', 'r1'), 1, (('s1',), ('s2', 's4')))
    assert rgame == expected

    game = rsgame.emptygame([3, 4, 5], [4, 3, 2])
    rgame = game.restrict(
        [False, True, True, False, False, False, True, True, False])
    expected = rsgame.emptygame_names(
        ('r0', 'r1', 'r2'), [3, 4, 5], (('s1', 's2'), ('s6',), ('s7',)))
    assert rgame == expected

    game = rsgame.emptygame(1, [2, 3])
    with pytest.raises(AssertionError):
        game.restrict([False, False, True, True, True])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_empty_restriction(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    rest = game.random_restriction()
    rgame = game.restrict(rest)
    assert np.all(game.num_role_players == rgame.num_role_players)
    assert rgame.num_strats == rest.sum()


def test_num_all_profiles():
    game = rsgame.emptygame(1, 1)
    assert np.all(game.num_all_role_profiles == [1])
    assert game.num_all_profiles == 1

    game = rsgame.emptygame(3, 2)
    assert np.all(game.num_all_role_profiles == [4])
    assert game.num_all_profiles == 4

    game = rsgame.emptygame([1, 3], 2)
    assert np.all(game.num_all_role_profiles == [2, 4])
    assert game.num_all_profiles == 8

    game = rsgame.emptygame(1, [3, 1])
    assert np.all(game.num_all_role_profiles == [3, 1])
    assert game.num_all_profiles == 3

    game = rsgame.emptygame([3, 2, 1], 3)
    assert np.all(game.num_all_role_profiles == [10, 6, 3])
    assert game.num_all_profiles == 180

    game = rsgame.emptygame([3, 2, 1], [1, 2, 3])
    assert np.all(game.num_all_role_profiles == [1, 3, 3])
    assert game.num_all_profiles == 9

    game = rsgame.emptygame([20, 20], 20)
    assert np.all(game.num_all_role_profiles == [68923264410, 68923264410])
    assert game.num_all_profiles == 4750416376930772648100


def test_num_all_payoffs():
    game = rsgame.emptygame(1, 1)
    assert game.num_all_payoffs == 1

    game = rsgame.emptygame(3, 2)
    assert game.num_all_payoffs == 6

    game = rsgame.emptygame([1, 3], 2)
    assert game.num_all_payoffs == 20

    game = rsgame.emptygame(1, [3, 1])
    assert game.num_all_payoffs == 6

    game = rsgame.emptygame([3, 2, 1], 3)
    assert game.num_all_payoffs == 774

    game = rsgame.emptygame([3, 2, 1], [1, 2, 3])
    assert game.num_all_payoffs == 30


def test_num_all_dpr_profiles():
    game = rsgame.emptygame(1, 1)
    assert game.num_all_dpr_profiles == 1

    game = rsgame.emptygame(3, 2)
    assert game.num_all_dpr_profiles == 6

    game = rsgame.emptygame([1, 3], 2)
    assert game.num_all_dpr_profiles == 16

    game = rsgame.emptygame(1, [3, 1])
    assert game.num_all_dpr_profiles == 3

    game = rsgame.emptygame([3, 2, 1], [1, 2, 3])
    assert game.num_all_dpr_profiles == 15


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_profile_counts(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)

    num_role_profiles = np.fromiter(  # pragma: no branch
        (rsgame.emptygame(p, s).all_profiles().shape[0] for p, s
         in zip(game.num_role_players, game.num_role_strats)),
        int, game.num_roles)
    assert np.all(num_role_profiles == game.num_all_role_profiles)

    num_profiles = game.all_profiles().shape[0]
    assert num_profiles == game.num_all_profiles

    num_payoffs = np.sum(game.all_profiles() > 0)
    assert num_payoffs == game.num_all_payoffs


def test_profile_id():
    game = rsgame.emptygame(3, [2, 2])
    profs = [[[0, 3, 2, 1],
              [2, 1, 3, 0]],
             [[2, 1, 2, 1],
              [3, 0, 3, 0]],
             [[1, 2, 1, 2],
              [2, 1, 1, 2]]]
    ids = game.profile_to_id(profs)
    assert ids.shape == (3, 2)
    assert np.all((0 <= ids) & (ids < game.num_all_profiles))

    game = rsgame.emptygame(3, [1, 2])
    prof = [3, 1, 2]
    assert game.profile_to_id(prof) == 2
    assert np.all(game.profile_from_id(2) == prof)

    game = rsgame.emptygame([1, 1, 1], [2, 2, 2])
    ids = np.arange(game.num_all_profiles)
    profs = game.profile_from_id(ids)
    assert np.all(profs == game.all_profiles())


def test_profile_id_big():
    game = rsgame.emptygame([20, 20], 20)
    profile = np.zeros(40, int)
    profile[[19, 39]] = 20
    assert game.profile_to_id(profile) == 4750416376930772648099
    assert np.all(game.profile_from_id(4750416376930772648099) == profile)

    game = rsgame.emptygame(40, 40)
    profile = np.zeros(40, int)
    profile[39] = 40
    assert game.profile_to_id(profile) == 53753604366668088230809
    assert np.all(game.profile_from_id(53753604366668088230809) == profile)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_profile_id(role_players, role_strats):
    # Here we have an expectation that all_profiles always returns profiles in
    # order of id
    game = rsgame.emptygame(role_players, role_strats)
    expected = np.arange(game.num_all_profiles)
    actual = game.profile_to_id(game.all_profiles())
    assert np.all(expected == actual)


def test_big_game_functions():
    """Test that everything works when game_size > int max"""
    game = rsgame.emptygame([100, 100], [30, 30])
    assert game.num_all_profiles > np.iinfo(int).max
    assert game.num_all_dpr_profiles > np.iinfo(int).max
    assert np.all(game.profile_to_id(game.random_profiles(1000)) >= 0)


def test_is_profile():
    game = rsgame.emptygame([2, 3], [3, 2])
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
    game = rsgame.emptygame(1, 1)
    expected = [[1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.emptygame(3, 2)
    expected = [[3, 0],
                [2, 1],
                [1, 2],
                [0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.emptygame([1, 3], 2)
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

    game = rsgame.emptygame(1, [3, 1])
    expected = [[1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.all_profiles())).size

    game = rsgame.emptygame([3, 2, 1], [1, 2, 3])
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
    game = rsgame.emptygame(1, 1)
    expected = [[1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.emptygame(3, 2)
    expected = [[3, 0],
                [0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.emptygame([1, 3], 2)
    expected = [[1, 0, 3, 0],
                [1, 0, 0, 3],
                [0, 1, 3, 0],
                [0, 1, 0, 3]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.emptygame(1, [3, 1])
    expected = [[1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1]]
    assert not np.setxor1d(utils.axis_to_elem(expected),
                           utils.axis_to_elem(game.pure_profiles())).size

    game = rsgame.emptygame([3, 2, 1], [1, 2, 3])
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
    game = rsgame.emptygame(1, 1)
    prof = [1]
    expected = [1]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.emptygame(3, 2)
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

    game = rsgame.emptygame([1, 3], 2)
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

    game = rsgame.emptygame(1, [3, 1])
    prof = [0, 0, 1, 1]
    expected = [[1, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1]]
    actual = game.nearby_profiles(prof, 1)
    assert not np.setxor1d(
        utils.axis_to_elem(expected),
        utils.axis_to_elem(actual)).size

    game = rsgame.emptygame([3, 2, 1], [1, 2, 3])
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
    base = rsgame.emptygame(role_players, role_strats)
    prof = base.random_profile()
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
    game = rsgame.emptygame(role_players, role_strats)
    all_profiles = game.all_profiles()
    assert game.num_all_profiles == all_profiles.shape[0]
    assert game.is_profile(all_profiles).all()
    pure_profiles = game.pure_profiles()
    assert game.num_pure_restrictions == pure_profiles.shape[0]
    assert game.is_profile(pure_profiles).all()


def test_random_profiles():
    game = rsgame.emptygame(3, 3)
    mixes = game.random_profiles(100, [0, 0.4, 0.6])
    assert np.all(mixes[:, 0] == 0)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_random_profiles(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    assert game.is_profile(game.random_profiles(100)).all()


def test_round_mixture_to_profile():
    game = rsgame.emptygame(3, 3)

    prof = game.round_mixture_to_profile([1/3, 1/3, 1/3])
    assert np.all(prof == 1)

    prof = game.round_mixture_to_profile([2/3, 0, 1/3])
    assert np.all(prof == [2, 0, 1])

    prof = game.round_mixture_to_profile([.1, .2, .7])
    assert np.all(prof == [0, 1, 2])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_round_mixture_to_profile(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    mixtures = np.concatenate([
        game.random_mixtures(100),
        game.random_sparse_mixtures(100),
    ])
    assert game.is_profile(game.round_mixture_to_profile(mixtures)).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_random_dev_profiles(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    prof = game.random_role_deviation_profile()
    for r, dprof in enumerate(prof):
        role_players = game.num_role_players.copy()
        role_players[r] -= 1
        dgame = rsgame.emptygame(role_players, game.num_role_strats)
        assert dgame.is_profile(dprof).all()

    profs = game.random_role_deviation_profiles(100)
    assert profs.shape == (100, game.num_roles, game.num_strats)
    for r, dprofs in enumerate(np.rollaxis(profs, 1, 0)):
        role_players = game.num_role_players.copy()
        role_players[r] -= 1
        dgame = rsgame.emptygame(role_players, game.num_role_strats)
        assert dgame.is_profile(dprofs).all()


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_random_deviator_profiles(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
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


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_max_prob_prof(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
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
    game = rsgame.emptygame([11, 3], [2, 1])
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


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_profile_serialization(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    profs = game.random_profiles(20)
    copies = np.empty(profs.shape, int)
    for prof, copy in zip(profs, copies):
        jprof = json.dumps(game.profile_to_json(prof))
        game.profile_from_json(json.loads(jprof), copy)
    assert np.all(copies == profs)


def test_to_from_payoff_json():
    game = rsgame.emptygame([11, 3], [2, 1])
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


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_payoff_serialization(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    pays = np.random.random((20, game.num_strats))
    pays *= pays < 0.8
    copies = np.empty(pays.shape)
    for pay, copy in zip(pays, copies):
        jpay = json.dumps(game.payoff_to_json(pay))
        game.payoff_from_json(json.loads(jpay), copy)
    assert np.allclose(copies, pays)


def test_to_from_prof_repr():
    game = rsgame.emptygame([11, 3], [2, 1])
    prof = [6, 5, 3]
    prof_str = 'r0: 6 s0, 5 s1; r1: 3 s2'
    assert game.profile_to_repr(prof) == prof_str
    aprof = game.profile_from_repr(prof_str)
    assert np.all(aprof == prof)
    aprof = np.empty_like(aprof)
    game.profile_from_repr(prof_str, dest=aprof)
    assert np.all(aprof == prof)


def test_to_from_prof_str():
    game = rsgame.emptygame([11, 3], [2, 1])
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
    game = rsgame.emptygame([11, 3], [2, 1])
    devpay = [5, 0]
    json_devpay = {'r0': {'s0': {'s1': 5}}}
    json_devpay2 = {'r0': {'s0': {'s1': 5}, 's1': {'s0': 0}}, 'r1': {'s2': {}}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert game.devpay_to_json(devpay) == json_devpay
        dest = np.empty(game.num_devs)
        game.devpay_from_json(json_devpay, dest)
        assert np.allclose(dest, devpay)
        assert np.allclose(game.devpay_from_json(json_devpay), devpay)
        assert np.allclose(game.devpay_from_json(json_devpay2), devpay)

    devpay = [5, 4]
    json_devpay = {'r0': {'s0': {'s1': 5}, 's1': {'s0': 4}}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert game.devpay_to_json(devpay) == json_devpay
        dest = np.empty(game.num_devs)
        game.devpay_from_json(json_devpay, dest)
        assert np.allclose(dest, devpay)
        assert np.allclose(game.devpay_from_json(json_devpay), devpay)


@pytest.mark.parametrize('role_players', [1, 2, 3, [3, 2, 1]])
@pytest.mark.parametrize('role_strats', [2, 4, [2, 3, 4]])
def test_random_devpay_serialization(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    pays = np.random.random((20, game.num_devs))
    pays *= pays < 0.8
    copies = np.empty(pays.shape)
    for pay, copy in zip(pays, copies):
        jpay = json.dumps(game.devpay_to_json(pay))
        game.devpay_from_json(json.loads(jpay), copy)
    assert np.allclose(copies, pays)


def test_is_symmetric():
    assert rsgame.emptygame(3, 4).is_symmetric()
    assert not rsgame.emptygame([2, 2], 3).is_symmetric()


def test_is_asymmetric():
    assert rsgame.emptygame(1, 4).is_asymmetric()
    assert not rsgame.emptygame([1, 2], 3).is_asymmetric()


def test_to_from_json():
    game = rsgame.emptygame(4, 5)
    jgame = {'players': {'r0': 4},
             'strategies': {'r0': ['s0', 's1', 's2', 's3', 's4']},
             'type': 'emptygame.1'}
    old_jgame = {'roles': [{'name': 'r0',
                            'strategies': ['s0', 's1', 's2', 's3', 's4'],
                            'count': 4}]}
    assert game.to_json() == jgame
    assert rsgame.emptygame_json(jgame) == game
    assert rsgame.emptygame_json(old_jgame) == game
    json.dumps(game.to_json())  # serializable

    game = rsgame.emptygame([4, 3], [3, 4])
    jgame = {'players': {'r0': 4, 'r1': 3},
             'strategies': {'r0': ['s0', 's1', 's2'],
                            'r1': ['s3', 's4', 's5', 's6']},
             'type': 'emptygame.1'}
    old_jgame = {'roles': [{'name': 'r0',
                            'strategies': ['s0', 's1', 's2'],
                            'count': 4},
                           {'name': 'r1',
                            'strategies': ['s3', 's4', 's5', 's6'],
                            'count': 3}]}
    assert game.to_json() == jgame
    assert json.loads(json.dumps(game.to_json())) == jgame
    assert rsgame.emptygame_json(jgame) == game
    assert rsgame.emptygame_json(old_jgame) == game

    with pytest.raises(ValueError):
        rsgame.emptygame_json({})


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_json_serialization(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    jgame = json.dumps(game.to_json())
    copy = rsgame.emptygame_json(json.loads(jgame))
    assert copy == game


def test_emptygame_hash_eq():
    a = rsgame.emptygame(4, 5)
    b = rsgame.emptygame([4], [5])
    assert a == b and hash(a) == hash(b)

    a = rsgame.emptygame([1, 2], [3, 2])
    b = rsgame.emptygame([1, 2], [3, 2])
    assert a == b and hash(a) == hash(b)

    a = rsgame.emptygame([2], [3, 2])
    b = rsgame.emptygame([2, 2], [3, 2])
    assert a == b and hash(a) == hash(b)

    a = rsgame.emptygame([2, 3], [3])
    b = rsgame.emptygame([2, 3], [3, 3])
    assert a == b and hash(a) == hash(b)

    assert rsgame.emptygame(3, 4) != rsgame.emptygame(3, 5)
    assert rsgame.emptygame(3, 4) != rsgame.emptygame(2, 4)
    assert rsgame.emptygame([1, 2], 4) != rsgame.emptygame([2, 2], 4)
    assert rsgame.emptygame([1, 2], 4) != rsgame.emptygame([2, 1], 4)
    assert rsgame.emptygame(2, [2, 3]) != rsgame.emptygame(2, [2, 2])
    assert rsgame.emptygame(2, [2, 3]) != rsgame.emptygame(2, [3, 2])


def test_emptygame_repr():
    game = rsgame.emptygame(3, 4)
    expected = 'EmptyGame([3], [4])'
    assert repr(game) == expected

    game = rsgame.emptygame(3, [4, 5])
    expected = 'EmptyGame([3 3], [4 5])'
    assert repr(game) == expected


def test_emptygame_str():
    game = rsgame.emptygame(3, 4)
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

    game = rsgame.emptygame([3, 4], [4, 3])
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


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_emptygame_copy(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    copy = rsgame.emptygame_copy(game)
    assert game == copy and hash(game) == hash(copy)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_complete_game(role_players, role_strats):
    base = rsgame.emptygame(role_players, role_strats)
    game = ZeroGame(base.role_names, base.strat_names, base.num_role_players)
    assert game.is_complete()
    assert game.is_constant_sum()
    assert not game.is_empty()
    assert game.num_profiles == game.num_all_profiles
    assert game.num_complete_profiles == game.num_all_profiles
    assert all(prof in game for prof in game.all_profiles())
    assert np.all(game.profiles() == game.all_profiles())

    assert np.allclose(game.deviation_payoffs(game.random_mixture()), 0)
    assert np.allclose(game.max_strat_payoffs(), 0)
    assert np.allclose(game.min_strat_payoffs(), 0)
    assert game == game.normalize()
    assert game == game.restrict(np.ones(game.num_strats, bool))


class ZeroGame(rsgame.CompleteGame):
    def deviation_payoffs(self, mix, *, jacobian=False):
        return np.zeros(self.num_strats)

    def get_payoffs(self, profs):
        return np.zeros(profs.shape)

    def max_strat_payoffs(self):
        return np.zeros(self.num_strats)

    def min_strat_payoffs(self):
        return np.zeros(self.num_strats)

    def normalize(self):
        return self

    def restrict(self, rest):
        base = rsgame.emptygame_copy(self).restrict(rest)
        return ZeroGame(base.role_names, base.strat_names,
                        base.num_role_players)
