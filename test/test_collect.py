import numpy as np

from gameanalysis import collect
from gameanalysis import utils


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


def test_bitset():
    bitset = collect.bitset(5)
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
    assert bitset == collect.bitset(5, [a, b])

    bitset.clear()
    assert not bitset
    assert repr(bitset) == 'BitSet([0])'
    assert list(map(list, bitset)) == [[False] * 5]
    assert bitset.add(a)
