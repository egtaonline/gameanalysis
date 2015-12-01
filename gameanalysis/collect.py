import collections


def __blocked_attribute(*args, **kwargs):
    raise TypeError("hashing requires immutability")


class frozendict(collections.Mapping):
    """An immutable hashable dictionary

    Keys are stored in sorted order

    """

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], frozendict):
            self._d = args[0]._d
            self._hash = args[0]._hash
        else:
            self._d = collections.OrderedDict(
                sorted(dict(*args, **kwargs).items()))
            self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self.items()))
        return self._hash

    def __eq__(self, other):
        return all(a == b for a, b in zip(self.items(), other.items()))

    def __repr__(self):
        return '{}({{{}}})'.format(
            self.__class__.__name__,
            ', '.join('{!r}: {!r}'.format(k, v) for k, v in self.items()))


# class harray(np.ndarray):
#     """
#     An immutable (and therefore hashable and sortable) subclass of ndarray.

#     Useful for profile data of strategy counts, payoffs, and mixed strategied
#     """
#     byteswap = fill = itemset = resize = setasflat = setfield = setflags = \
#         sort = __blocked_attribute

#     def __new__(cls, *args, **kwargs):
#         a = np.array(*args, **kwargs)
#         a = np.ndarray.__new__(cls, shape=a.shape, dtype=a.dtype,
#                                strides=a.strides, buffer=a.data)
#         cls.setflags = np.ndarray.setflags
#         a.setflags(write=False)
#         cls.setflags = __blocked_attribute
#         return a

#     def __hash__(self):
#         try:
#             return self._hash
#         except AttributeError:
#             self._hash = hash(tuple(self.flat))
#             return self._hash

#     def __eq__(self, other):
#         return np.array_equal(self, other)

#     def __cmp__(self, other):
#         assert self.shape == other.shape
#         try:
#             i = self != other
#             return self[i][0] - other[i][0]
#         except IndexError:
#             return False

#     def __lt__(self, other):
#         return min(self.__cmp__(other), 0)

#     def __gt__(self, other):
#         return max(self.__cmp__(other), 0)

#     def __le__(self, other):
#         return self == other or self < other

#     def __ge__(self, other):
#         return self == other or self > other


# class Mixture(np.ndarray):
#     def __new__(cls, probabilities, game):
#         a = np.array(probabilities, dtype=float).clip(0)
#         a[a.max(1) == 0] = 1
#         a[game.mask] = 0
#         a = harray.__new__(cls, (a.T/a.sum(1)).T)
#         a.game = game
#         return a

#     def probability(self, profile):
#         return prod((self ** profile).flat) * profile.reps

#     def dist(self, other):
#         """
#         L2-norm gives the euclidian distance between mixture vectors
#         """
#         return np.linalg.norm(self - other, 2)

#     def __str__(self):
#         return repr(self)

#     def __repr__(self):
#         try:
#             string = "{"
#             for i, r in enumerate(self.game.roles):
#                 string += r + ": ("
#                 for j, s in enumerate(self.game.strategies[r]):
#                     if self[i, j] > 0:
#                         string += s + ":" + str(round(100*self[i, j], 1)) + "%, "
#                         string = string[:-2] + "); "
#                 return string[:-2] + "}"
#         except AttributeError:
#             return np.ndarray.__repr__(self)
