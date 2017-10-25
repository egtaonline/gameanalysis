import itertools
import random
import warnings

import numpy as np
import numpy.random as rand
import pytest

from gameanalysis import utils


def array_set_equals(a, b):
    """Returns true if the unique last dimensions are the same set"""
    return not np.setxor1d(utils.axis_to_elem(a), utils.axis_to_elem(b)).size


def test_comb():
    assert utils.comb(100, 10) == 17310309456440
    assert utils.comb(100, 20) == 535983370403809682970
    assert utils.comb([100], 20)[0] == 535983370403809682970


def test_only():
    assert utils.only([None]) is None, \
        "only didn't return only element"
    with pytest.raises(ValueError):
        utils.only([])
    with pytest.raises(ValueError):
        utils.only([None, None])
    with pytest.raises(ValueError):
        utils.only(5)


def test_game_size():
    assert utils.game_size(2000, 10) == 1442989326579174917694151


def test_one_line():
    short = "This is a short string, so it won't get truncated"
    assert utils.one_line(short, 100) == short, \
        "short string still got truncated"
    long_str = "This is relatively long"
    expected = "This is rela...g"
    assert utils.one_line(long_str, 16) == expected, \
        "one_line didn't truncate as expected"


def test_acomb():
    actual = utils.acomb(5, 0)
    assert actual.shape == (1, 5)
    assert not actual.any()

    actual = utils.acomb(5, 5)
    assert actual.shape == (1, 5)
    assert actual.all()

    actual = utils.acomb(6, 4)
    expected = np.zeros_like(actual)
    for i, inds in enumerate(itertools.combinations(range(6), 4)):
        expected[i, inds] = True
    assert array_set_equals(actual, expected)

    actual = utils.acomb(6, 4, True)
    expected = np.zeros_like(actual)
    for i, inds in enumerate(map(list, itertools.combinations_with_replacement(
            range(6), 4))):
        np.add.at(expected[i], inds, 1)
    assert array_set_equals(actual, expected)


def test_acartesian2():
    a = np.array([[1, 2, 3],
                  [4, 5, 6]], int)
    b = np.array([[7, 8],
                  [9, 10],
                  [11, 12]], int)
    c = np.array([[13]], int)

    assert array_set_equals(a, utils.acartesian2(a))
    assert array_set_equals(b, utils.acartesian2(b))
    assert array_set_equals(c, utils.acartesian2(c))

    expected = np.array([[1, 2, 3,  7,  8, 13],
                         [1, 2, 3,  9, 10, 13],
                         [1, 2, 3, 11, 12, 13],
                         [4, 5, 6,  7,  8, 13],
                         [4, 5, 6,  9, 10, 13],
                         [4, 5, 6, 11, 12, 13]], int)
    assert array_set_equals(expected[:, :-1], utils.acartesian2(a, b))
    assert array_set_equals(expected, utils.acartesian2(a, b, c))


def test_simplex_project():
    res = utils.simplex_project(np.array([0, 0, 0]))
    assert np.allclose(res, [1 / 3] * 3), \
        "projecting [0, 0, 0] didn't result in uniform"

    res = utils.simplex_project(np.array([1.2, 1.4]))
    assert np.allclose(res, [.4, .6]), \
        "simplex project didn't return correct result"

    res = utils.simplex_project(np.array([-0.1, 0.8]))
    assert np.allclose(res, [0.05, 0.95]), \
        "simplex project didn't return correct result"


@pytest.mark.parametrize('array', [
    rand.random(6),
    rand.random((4, 5)),
    rand.random((2, 3, 4)),
    rand.random((3, 4, 5)),
    rand.random((5, 4)),
])
def test_simplex_project_random(array):
    simp = utils.simplex_project(array)
    assert simp.shape == array.shape
    assert np.all(simp >= 0)
    assert np.allclose(simp.sum(-1), 1)


def test_multinomial_mode():
    actual = utils.multinomial_mode([1], 4)
    expected = [4]
    assert np.all(actual == expected)

    actual = utils.multinomial_mode([0, 1], 4)
    expected = [0, 4]
    assert np.all(actual == expected)

    actual = utils.multinomial_mode([0.5, 0.5], 4)
    expected = [2, 2]
    assert np.all(actual == expected)

    actual = utils.multinomial_mode([0.5, 0.5], 3)
    expected = [[1, 2], [2, 1]]
    assert np.all(actual == expected, 1).any()

    actual = utils.multinomial_mode([0.45, 0.45, 0.1], 5)
    expected = [[3, 2, 0], [2, 3, 0]]
    assert np.all(actual == expected, 1).any()


def test_elem_axis():
    x = np.array([[5.4, 2.2],
                  [5.7, 2.8],
                  [9.6, 1.2]], float)
    assert np.all(x == utils.elem_to_axis(utils.axis_to_elem(x), float))
    assert np.all(x.astype(int) ==
                  utils.elem_to_axis(utils.axis_to_elem(x.astype(int)), int))
    assert utils.unique_axis(x).shape == (3, 2)
    array, counts = utils.unique_axis(x.astype(int), return_counts=True)
    assert array.shape == (2, 2)
    assert not np.setxor1d(counts, [2, 1]).size


def test_empty_elem_axis():
    x = np.empty((0, 2), float)
    assert np.all(x.shape == utils.elem_to_axis(
        utils.axis_to_elem(x), float).shape)
    assert np.all(
        x.astype(int).shape ==
        utils.elem_to_axis(utils.axis_to_elem(x.astype(int)), int).shape)
    assert utils.unique_axis(x).shape == (0, 2)
    array, counts = utils.unique_axis(x.astype(int), return_counts=True)
    assert array.shape == (0, 2)
    assert counts.shape == (0,)


def test_hash_array():
    arrayset = {utils.hash_array([3, 4, 5]), utils.hash_array([6, 7])}

    assert utils.hash_array([3, 4, 5]) in arrayset
    assert utils.hash_array([6, 7]) in arrayset
    assert utils.hash_array([3, 4, 6]) not in arrayset


@pytest.mark.parametrize('_', range(100))
def test_random_bipartite(_):
    shape = (random.randint(1, 5), random.randint(1, 5))
    mins = (random.randint(1, shape[1]), random.randint(1, shape[0]))
    mask = utils.random_con_bitmask(.2, shape, mins)
    assert np.all(mask.sum(0) >= mins[1])
    assert np.all(mask.sum(1) >= mins[0])


@pytest.mark.parametrize('_', range(100))
def test_random_con_bitmask(_):
    ndim = random.randint(2, 4)
    shape = tuple(random.randint(1, 5) for _ in range(ndim))
    total = utils.prod(shape)
    mins = tuple(random.randint(1, total // d) for d in shape)
    mask = utils.random_con_bitmask(.2, shape, mins)

    for dim, min_in in enumerate(mins):
        assert np.all(np.rollaxis(mask, dim).reshape(
            mask.shape[dim], -1).sum(1) >= min_in)


@pytest.mark.parametrize('_', range(100))
def test_random_con_bitmask_default(_):
    ndim = random.randint(2, 4)
    shape = tuple(random.randint(1, 5) for _ in range(ndim))
    mask = utils.random_con_bitmask(.2, shape)

    for dim in range(ndim):
        assert np.rollaxis(mask, dim).reshape(
            mask.shape[dim], -1).any(1).all()


def test_iunique():
    items = ['a', 'd', 'b', 'd', 'c', 'c', 'a']
    expected = ['a', 'd', 'b', 'c']
    actual = list(utils.iunique(items))
    assert expected == actual


def test_random_strings():
    assert next(utils.random_strings(5, digits='a')) == 'aaaaa'
    assert len(next(utils.random_strings(6))) == 6
    assert all(5 <= len(s) <= 10 for s
               in itertools.islice(utils.random_strings(5, 10), 40))


def test_prefix_strings():
    assert utils.is_sorted(utils.prefix_strings('', 13))


def test_is_sorted():
    assert utils.is_sorted([])
    assert utils.is_sorted([0])
    assert utils.is_sorted([[0], [1], [2]], key=lambda x: x[0])
    assert utils.is_sorted([3, 4])
    assert utils.is_sorted([3, 4], strict=True)
    assert not utils.is_sorted([3, 4], reverse=True)
    assert not utils.is_sorted([3, 4], reverse=True, strict=True)
    assert utils.is_sorted([4, 3], reverse=True)
    assert utils.is_sorted([4, 3], reverse=True, strict=True)
    assert utils.is_sorted([3, 4, 4, 5])
    assert not utils.is_sorted([3, 4, 4, 5], strict=True)
    assert utils.is_sorted([5, 4, 4, 3], reverse=True)
    assert not utils.is_sorted([5, 4, 4, 3], reverse=True, strict=True)


def test_memoization():
    # We need an object we can memoize
    class Obj(object):
        pass

    called = [0]

    @utils.memoize
    def func(obj):
        called[0] += 1

    obj = Obj()
    assert called[0] == 0
    assert func(obj) is None
    assert called[0] == 1
    assert func(obj) is None
    assert called[0] == 1


def test_deprecation():

    @utils.deprecated
    def func(a, b):
        return a, b

    try:
        func(None, None)
        raise ValueError("This should never be reached")  # pragma: no cover
    except DeprecationWarning:
        pass

    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always', DeprecationWarning)
        assert func(3, b=4) == (3, 4)

    assert len(warns) == 1
    assert issubclass(warns[0].category, DeprecationWarning)
