import numpy as np
import pytest

from gameanalysis import collect
from gameanalysis import utils
from test import utils as tu


def test_connected_component():
    simset = collect.mcces(1)
    assert simset.add([0], 0)
    assert simset.add([1.5], 1)
    assert not simset.add([0.75], .5)
    assert len(simset) == 1
    assert [((0,), 0)] == list(simset)
    assert (repr(simset) ==
            'MinimumConnectedComponentElementSet(1, [((0,), 0)])')

    simset.clear()
    assert simset.add([0.75], 1)
    assert not simset.add([1.5], 0.5)
    assert not simset.add([0], 0)
    assert [((0,), 0)] == list(simset)

    simset.clear()
    assert simset.add([0], 0)
    assert not simset.add([0.75], 1)
    assert not simset.add([1.5], 0.5)
    assert simset.add([3], 2)
    assert [((0,), 0), ((3,), 2)] == list(simset)


@tu.warnings_filter(DeprecationWarning)
def test_weighted_similarity():
    simset = collect.WeightedSimilaritySet(lambda a, b: abs(a - b) <= 1)
    simset.add(1, 0)
    simset.add(2, 1)
    simset.add(0, .5)
    assert len(simset) == 1
    assert len(simset) == 1  # computed not
    assert [(1, 0)] == list(simset)
    repr(simset)

    simset.clear()
    simset.add(0, .5)
    simset.add(2, 1)
    simset.add(1, 0)
    assert [(1, 0)] == list(simset)


def test_dynamic_array():
    darray = collect.DynamicArray(3, dtype=int)
    str(darray)
    repr(darray)
    darray.ensure_capacity(100)
    darray.append([0, 1, 2])
    darray.append([[3, 4, 5], [6, 7, 8]])
    assert len(darray) == 3
    assert np.all(darray.data == [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert all(np.all(x == y) for x, y
               in zip(darray, [[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    assert np.all(darray.pop() == [6, 7, 8])
    darray.compact()
    assert np.all(darray.pop(2) == [[0, 1, 2], [3, 4, 5]])

    darray = collect.DynamicArray([3], dtype=int)
    darray.append([0, 1, 2])
    assert len(darray) == 1


def test_dynamic_array_failure():
    with pytest.raises(ValueError):
        collect.DynamicArray(3, dtype=int).append([0, 0])


def test_bit_set():
    bitset = collect.bitset()
    assert not bitset
    a = np.array([0, 1, 1, 1, 0], bool)
    b = np.array([1, 0, 0, 1, 1], bool)
    c = np.array([0, 0, 0, 1, 0], bool)
    assert bitset.add(a)
    assert bitset
    assert not bitset.add(c)
    assert bitset.add(b)
    assert not bitset.add(c)
    assert not bitset.add(a)
    assert not bitset.add(b)

    expected = frozenset(map(utils.hash_array, [a, b]))
    actual = frozenset(map(utils.hash_array, bitset))
    assert expected == actual
    assert bitset

    bitset.clear()
    assert not bitset
    assert repr(bitset) == 'BitSet([])'
    assert list(bitset) == []
    assert bitset.add(a)


@tu.warnings_filter(DeprecationWarning)
def test_mixture_set():
    mixset = collect.MixtureSet(0.1)
    assert not mixset
    assert mixset.add(np.array([0.2, 0.1, 0.7]))
    assert not mixset.add(np.array([0.25, 0.05, 0.7]))
    assert mixset.add(np.array([0.3, 0, 0.7]))
    assert len(mixset) == 2
    mixset.clear()
    assert repr(mixset) == 'MixtureSet([])'
    assert mixset.add(np.array([0.2, 0.1, 0.7]))
