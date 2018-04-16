"""Test utils"""
import itertools
import warnings

import numpy as np
import numpy.random as rand
import pytest

from gameanalysis import utils


def array_set_equals(one, two):
    """Returns true if the unique last dimensions are the same set"""
    return not np.setxor1d(utils.axis_to_elem(one),
                           utils.axis_to_elem(two)).size


def test_comb():
    """Test comb"""
    assert utils.comb(100, 10) == 17310309456440
    assert utils.comb(100, 20) == 535983370403809682970
    assert utils.comb([100], 20)[0] == 535983370403809682970
    assert np.all(utils.comb(100, [10, 20]) ==
                  [17310309456440, 535983370403809682970])
    assert utils.comb_inv(535983370403809682970, 20) == 100
    combs = utils.comb(100, [0, 1, 2, 10])
    # k of 0 is unrecoverable, so it must return 0
    assert np.all(utils.comb_inv(combs, [0, 1, 2, 10]) == [0, 100, 100, 100])
    combs = utils.comb([0, 1, 3, 5, 3, 2], [0, 0, 2, 2, 3, 1])
    assert np.all(utils.comb_inv(combs, [0, 0, 2, 2, 3, 1]) ==
                  [0, 0, 3, 5, 3, 2])
    assert np.all(utils.comb_inv(100, [1, 2, 5, 10]) == [100, 14, 8, 12])


def test_check():
    """Test that check works"""
    utils.check(True, '')
    with pytest.raises(ValueError):
        utils.check(False, '')


def test_game_size():
    """Test game size"""
    assert utils.game_size(2000, 10) == 1442989326579174917694151
    assert np.all(utils.game_size([10, 20, 100], 10) ==
                  [92378, 10015005, 4263421511271])
    assert np.all(utils.game_size(10, [10, 20, 100]) ==
                  [92378, 20030010, 42634215112710])
    assert np.all(utils.game_size([100, 20, 10], [10, 20, 100]) ==
                  [4263421511271, 68923264410, 42634215112710])
    assert utils.game_size_inv(1442989326579174917694151, 2000) == 10
    assert np.all(utils.game_size_inv(
        [92378, 10015005, 4263421511271], [10, 20, 100]) == 10)
    assert np.all(utils.game_size_inv(
        [92378, 20030010, 42634215112710], 10) == [10, 20, 100])
    assert np.all(utils.game_size_inv(
        [4263421511271, 68923264410, 42634215112710],
        [100, 20, 10]) == [10, 20, 100])
    assert np.all(utils.game_size_inv(100, [1, 5, 20]) == [100, 4, 2])


def test_repeat():
    """test repeat"""
    assert [1, 1, 2, 2, 2] == list(utils.repeat([1, 2], [2, 3]))
    assert [1, 1, 2, 2, 2] == list(utils.repeat([1, 2, 2], [2, 1, 2]))
    assert [1, 2] == list(utils.repeat([1, 3, 2], [1, 0, 1]))


def test_acomb():
    """Test acomb"""
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
    """Test acartesian2"""
    one = np.array([[1, 2, 3],
                    [4, 5, 6]], int)
    two = np.array([[7, 8],
                    [9, 10],
                    [11, 12]], int)
    three = np.array([[13]], int)

    assert array_set_equals(one, utils.acartesian2(one))
    assert array_set_equals(two, utils.acartesian2(two))
    assert array_set_equals(three, utils.acartesian2(three))

    expected = np.array([[1, 2, 3, 7, 8, 13],
                         [1, 2, 3, 9, 10, 13],
                         [1, 2, 3, 11, 12, 13],
                         [4, 5, 6, 7, 8, 13],
                         [4, 5, 6, 9, 10, 13],
                         [4, 5, 6, 11, 12, 13]], int)
    assert array_set_equals(expected[:, :-1], utils.acartesian2(one, two))
    assert array_set_equals(expected, utils.acartesian2(one, two, three))


def test_simplex_project():
    """Test simplex project"""
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
    """Test simplex project on random arrays"""
    simp = utils.simplex_project(array)
    assert simp.shape == array.shape
    assert np.all(simp >= 0)
    assert np.allclose(simp.sum(-1), 1)


def test_multinomial_mode():
    """Test multinomial mode"""
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


def test_geometric_histogram():
    """test geometric histogram"""
    res = utils.geometric_histogram(1, .2)
    assert res.sum() == 1
    assert np.all(res >= 0)
    assert res[-1] > 0

    res = utils.geometric_histogram(10000, .5)
    assert res[0] > 0  # Fails with prob .5 ^ 10000
    assert res.sum() == 10000
    assert np.all(res >= 0)
    assert res[-1] > 0


def test_elem_axis():
    """Test identity of axis_to_elem o axis_from_elem"""
    arr = np.array([[5.4, 2.2],
                    [5.7, 2.8],
                    [9.6, 1.2]], float)
    assert np.all(arr == utils.axis_from_elem(utils.axis_to_elem(arr)))
    assert np.all(arr.astype(int) ==
                  utils.axis_from_elem(utils.axis_to_elem(arr.astype(int))))


def test_empty_elem_axis():
    """Test axis_to_elem works for empty arrays"""
    arr = np.empty((0, 2), float)
    assert np.all(arr.shape == utils.axis_from_elem(
        utils.axis_to_elem(arr)).shape)
    assert np.all(
        arr.astype(int).shape ==
        utils.axis_from_elem(utils.axis_to_elem(arr.astype(int))).shape)


def test_hash_array():
    """Test that hash array works"""
    arrayset = {utils.hash_array([3, 4, 5]), utils.hash_array([6, 7])}

    assert utils.hash_array([3, 4, 5]) in arrayset
    assert utils.hash_array([6, 7]) in arrayset
    assert utils.hash_array([3, 4, 6]) not in arrayset


def test_iunique():
    """Test iter unique"""
    items = ['a', 'd', 'b', 'd', 'c', 'c', 'a']
    expected = ['a', 'd', 'b', 'c']
    actual = list(utils.iunique(items))
    assert expected == actual


def test_random_strings():
    """Test random strings returns intended results"""
    assert next(utils.random_strings(5, digits='a')) == 'aaaaa'
    assert len(next(utils.random_strings(6))) == 6
    assert all(5 <= len(s) <= 10 for s
               in itertools.islice(utils.random_strings(5, 10), 40))


def test_prefix_strings():
    """Test prefix strings returns sorted lists"""
    assert utils.is_sorted(utils.prefix_strings('', 13))


def test_is_sorted():
    """Test is_sorted"""
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
    """Test that memoization works"""
    # We need an object we can memoize
    class Obj(object): # pylint: disable=too-few-public-methods
        """Object to memoize"""
        pass

    called = [0]

    @utils.memoize
    def func(_):
        """Memoized"""
        called[0] += 1

    obj = Obj()
    assert called[0] == 0
    assert func(obj) is None
    assert called[0] == 1
    assert func(obj) is None
    assert called[0] == 1


@pytest.mark.parametrize('_', range(20))
def test_allclose_perm(_):
    """Test allclose_perm accurately detected permutation"""
    one, two = np.random.random((2, 10, 4))
    three = one.copy()
    np.random.shuffle(three)
    assert utils.allclose_perm(one, three)
    assert not utils.allclose_perm(one, two)


def test_deprecation():
    """Test that deprecation works"""
    @utils.deprecated
    def func(aarg, barg):
        """Test function"""
        return aarg, barg

    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always', DeprecationWarning)
        assert func(3, barg=4) == (3, 4)

    assert len(warns) == 1
    assert issubclass(warns[0].category, DeprecationWarning)


def test_subsequences():
    """Test that sub-sequences work"""
    expected = [(i, i+1) for i in range(4)]
    assert list(utils.subsequences(range(5))) == expected

    expected = [(i, i+1, i+2) for i in range(3)]
    assert list(utils.subsequences(range(5), 3)) == expected


def test_asubsequences_1d():
    """Test that 1d sub-sequences work"""
    array = np.arange(5)
    expected = np.stack([np.arange(4), np.arange(1, 5)])
    assert np.all(utils.asubsequences(array) == expected)

    expected = np.stack([np.arange(3), np.arange(1, 4), np.arange(2, 5)])
    assert np.all(utils.asubsequences(array, 3) == expected)


def test_asubsequences_2d():
    """Test that 1d sub-sequences work"""
    array = np.random.random((5, 3))
    expected = np.stack([array[:-1], array[1:]])
    assert np.all(utils.asubsequences(array) == expected)
