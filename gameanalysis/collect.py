from collections import abc

import numpy as np


class WeightedSimilaritySet(object):
    """A set of non-similar elements prioritized by weight

    Allows adding a bunch of weighted elements, and when iterated, only
    iterates over dissimilar elements with the lowest weights. Adding new
    elements that are similar to the existing set, but with higher weights
    won't change the set returned."""
    def __init__(self, is_similar):
        self._is_similar = is_similar
        self._i = 0  # Tie breaking
        self._items = []
        self._set = []
        self._computed = True

    def add(self, item, weight):
        self._computed = False
        self._set.clear()
        self._items.append((weight, self._i, item))
        self._i += 1
        return self

    def _satisfy(self):
        if not self._computed:
            self._items.sort()
            for w, _, i in self._items:
                if all(not self._is_similar(i, j) for j, _ in self._set):
                    self._set.append((i, w))

    def __len__(self):
        self._satisfy()
        return len(self._set)

    def __iter__(self):
        self._satisfy()
        return iter(self._set)

    def __repr__(self):
        self._satisfy()
        suffix = ('.add(' + ').add('.join('{}, {}'.format(i, w)
                                          for w, _, i in self._items) + ')'
                  if self._items else '')
        return '{}({}){}'.format(self.__class__.__name__,
                                 self._is_similar.__name__, suffix)


class DynamicArray(object):
    """A object with a backed array that also allows adding data"""
    def __init__(self, item_shape, dtype=None, initial_room=8,
                 grow_fraction=2):
        assert grow_fraction > 1
        if not isinstance(item_shape, abc.Sized):
            item_shape = (item_shape,)
        self._data = np.empty((initial_room,) + item_shape, dtype)
        self._length = 0
        self._grow_fraction = 2

    def append(self, array):
        """Append an array"""
        array = np.asarray(array, self._data.dtype)
        if array.shape[1:] == self._data.shape[1:]:
            # Adding multiple
            num = array.shape[0]
            self.ensure_capacity(self._length + num)
            self._data[self._length:self._length+num] = array
            self._length += num

        elif array.shape == self._data.shape[1:]:
            # Adding one
            self.ensure_capacity(self._length + 1)
            self._data[self._length] = array
            self._length += 1

        else:
            raise ValueError("Invalid shape for append")

    def pop(self, num=None):
        """Pop one or several arrays"""
        if num is None:
            assert self._length > 0, "can't pop from an empty array"
            self._length -= 1
            return self._data[self._length].copy()

        else:
            assert num >= 0 and self._length >= num
            self._length -= num
            return self._data[self._length:self._length+num].copy()

    @property
    def data(self):
        """A view of all of the data"""
        return self._data[:self._length]

    def ensure_capacity(self, new_capacity):
        """Make sure the array has a least new_capacity"""
        if new_capacity > self._data.shape[0]:
            growth = round(self._data.shape[0] * self._grow_fraction) + 1
            new_size = max(growth, new_capacity)
            new_data = np.empty((new_size,) + self._data.shape[1:],
                                self._data.dtype)
            new_data[:self._length] = self._data[:self._length]
            self._data = new_data

    def compact(self):
        """Trim underlying storage to hold only valid data"""
        self._data = self.data.copy()
        self._length = self._data.shape[0]

    def __len__(self):
        return self._length

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)
