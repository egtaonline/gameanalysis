from collections import abc

import numpy as np


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
