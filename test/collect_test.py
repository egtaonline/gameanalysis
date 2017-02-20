import numpy as np

from gameanalysis import collect


def test_bit_set():
    bitset = collect.BitSet()
    assert not bitset
    assert bitset.add(np.array([0, 1, 1, 0, 1], bool))
    assert not bitset.add(np.array([0, 1, 0, 0, 1], bool))
    assert bitset.add(np.array([0, 1, 0, 1, 1], bool))
    bitset.clear()
    assert repr(bitset) == 'BitSet([])'
    assert bitset.add(np.array([0, 1, 1, 0, 1], bool))


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
