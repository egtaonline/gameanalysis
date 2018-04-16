"""Module with useful collections for game analysis"""
import bisect

import numpy as np

from gameanalysis import utils


def mcces(thresh):
    """Create a new minimum connected component set"""
    return _MinimumConnectedComponentElementSet(thresh)


class _MinimumConnectedComponentElementSet(object):
    """A class for returning vectors with the minimum weight

    Vectors are only returned if they have the minimum weight in their
    connected component, where two vectors are connected if they're closer than
    `thresh` distance apart.

    Inserts can take up to `O(n)` where `n` is the number of elements inserted.
    If this is problematic, a better data structure will probably be
    necessary."""

    def __init__(self, thresh):
        self._thresh = thresh ** 2
        self._set = []

    def _similar(self, ait, bit):
        """Test if elements are similar"""
        return sum((ai - bi) ** 2 for ai, bi in zip(ait, bit)) <= self._thresh

    def add(self, vector, weight):
        """Add a vector with a weight

        Returns true if the element is distinct from every element in the
        container"""
        vector = tuple(vector)
        mins = (weight, vector)
        vecs = [vector]
        new_set = []
        for set_tup in self._set:
            smin, svecs = set_tup
            if any(self._similar(vector, v) for v in svecs):
                mins = min(smin, mins)
                vecs.extend(svecs)
            else:
                new_set.append(set_tup)

        bisect.insort(new_set, (mins, vecs))
        self._set = new_set
        return len(vecs) == 1

    def clear(self):
        """Remove all vectors added to the set"""
        self._set.clear()

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter((v, w) for (w, v), _ in self._set)

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__[1:], self._thresh, list(self))


def bitset(dim, iterable=()):
    """Create a new bitset"""
    bits = _BitSet(dim)
    for bit in iterable:
        bits.add(bit)
    return bits


class _BitSet(object):
    """Set of bitmasks

    A bitmask is in the set if all of the true bits have been added
    together. When iterating, all maximal bitsets are returned. An empty bitset
    still contains 0."""
    # This compresses all bitmasks down to the number they are
    # implicitly, and uses bitwise math to replicate the same functions.

    def __init__(self, dim):
        self._masks = [0]
        self._mask = 2 ** np.arange(dim)

    def add(self, bitmask):
        """Add a mask to the bit set"""
        bitmask = np.asarray(bitmask, bool)
        if bitmask not in self: # pylint: disable=no-else-return
            num = bitmask.dot(self._mask)
            self._masks[:] = [m for m in self._masks if m & ~num]
            self._masks.append(num)
            return True
        else:
            return False

    def clear(self):
        """Clear all bitmasks that were added"""
        self._masks.clear()
        self._masks.append(0)

    def __contains__(self, bitmask):
        utils.check(
            bitmask.size == self._mask.size,
            "can't add bitmasks of different sizes")
        num = bitmask.dot(self._mask)
        return not all(num & ~m for m in self._masks)

    def __iter__(self):
        return ((m // self._mask % 2).astype(bool) for m in self._masks)

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return (type(self) is type(othr) and
                self._mask.size == othr._mask.size and
                frozenset(self._masks) == frozenset(othr._masks))

    def __bool__(self):
        return len(self._masks) > 1 or bool(self._masks[0] != 0)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__[1:], self._masks)
