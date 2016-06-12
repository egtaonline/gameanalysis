import functools
from collections import abc

import numpy as np


# To avoid computing the hash of the same memory over and over again, we need
# to cache the hash for the same arrays. However, the cache lookup using normal
# array equalities would be just as expensive and recomputing the hash each
# time, so instead we have the QuickHash object, which only hashes on location,
# view, and type of the data. This will not always cache the "same data" but if
# the same object is passed in multiple times, it will be cached.
class _QuickHash(object):
    def __init__(self, array):
        self.array = array

    def _key(self):
        return (np.byte_bounds(self.array) + self.array.strides +
                (self.array.dtype,))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return self._key() == other._key()


def _lru_quick_hash(func):

    @functools.wraps(func)
    def wrapper(array):
        return func(_QuickHash(array))

    return wrapper


@_lru_quick_hash
@functools.lru_cache()
class ArrayHash(object):
    """A subclass of ndarray that can be put in a dict or set

    Everything passed in is proxied to the underlying array, but hash and eq
    which behave as standard python would expect. The passed in array should
    not be writable.
    """
    def __init__(self, wrapped_array):
        self._array = wrapped_array.array
        self._array.setflags(write=False)
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._array.data.tobytes())
        return self._hash

    def __eq__(self, other):
        return bool(np.all(self._array == other._array))

    def __str__(self):
        return str(self._array)

    def __repr__(self):
        return repr(self._array)


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
