import numpy as np

from collections import OrderedDict

from BasicFunctions import *


def _blocked_attribute(*args, **kwargs):
	raise TypeError("hashing requires immutability")


class h_dict(OrderedDict):
	"""
	An sorted and immutable (and therefore hashable) subclass of OrderedDict.
	"""
	__setitem__ = update = setdefault = _blocked_attribute
	__delitem__ = clear = pop = popitem = _blocked_attribute

	def __init__(self, *args, **kwargs):
		OrderedDict.__init__(self)
		d = dict(*args, **kwargs)
		for k,v in sorted(d.items()):
			OrderedDict.__setitem__(self, k, v)

	def __hash__(self):
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(tuple(self.items()))
			return self._hash

	def __repr__(self):
		return '{' + OrderedDict.__repr__(self)[8:-2] + '}'


class h_array(np.ndarray):
	"""
	An immutable (and therefore hashable and sortable) subclass of ndarray.

	Useful for profile data of strategy counts, payoffs, and mixed strategied
	"""
	byteswap = fill = itemset = resize = setasflat = setfield = setflags = \
			sort = _blocked_attribute

	def __new__(cls, *args, **kwargs):
		a = np.array(*args, **kwargs)
		a = np.ndarray.__new__(cls, shape=a.shape, dtype=a.dtype, \
				strides=a.strides, buffer=a.data)
		cls.setflags = np.ndarray.setflags
		a.setflags(write=False)
		cls.setflags = _blocked_attribute
		return a

	def __hash__(self):
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(tuple(self))
			return self._hash

	def __eq__(self, other):
		return np.array_equal(self, other)

	def __cmp__(self, other):
		assert self.shape == other.shape
		try:
			i = self != other
			return self[i][0] - other[i][0]
		except IndexError:
			return False

	def __lt__(self, other):
		return min(self.__cmp__(other), 0)

	def __gt__(self, other):
		return max(self.__cmp__(other), 0)

	def __le__(self, other):
		return self == other or self < other

	def __ge__(self, other):
		return self == other or self > other

