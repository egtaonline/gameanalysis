import collections
from collections import abc


class frozendict(abc.Mapping, abc.Hashable):
    """An immutable ordered hashable dictionary"""

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], (fodict, frozendict)):
            self._data = args[0]._data
            self._hash = args[0]._hash
        else:
            self._data = dict(*args, **kwargs)
            assert all(isinstance(v, abc.Hashable) for v
                       in self._data.values()), \
                "frozendict values must be hashable"
            self._hash = None

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self.items()))
        return self._hash

    def __eq__(self, other):
        # Works even when both datas are differently ordered OrderedDicts
        if isinstance(other, dict):
            return self._data == other
        else:
            return dict.__eq__(self._data, other._data)

    def __repr__(self):
        return '{name}({{{items}}})'.format(
            name=self.__class__.__name__,
            items=', '.join('{!r}: {!r}'.format(k, v) for k, v
                            in self.items()))


class fodict(abc.Mapping, abc.Hashable):
    """An immutable ordered hashable dictionary

    Order of keys and values /is/ important for uniqueness. Two fodicts
    with different insertion orders are considered distinct.
    """

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], fodict):
            self._data = args[0]._data
            self._hash = args[0]._hash
        else:
            self._data = collections.OrderedDict(*args, **kwargs)
            assert all(isinstance(v, abc.Hashable) for v
                       in self._data.values()), \
                "fodict values must be hashable"
            self._hash = None

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self.items()))
        return self._hash

    def __eq__(self, other):
        return self._data == other._data

    def __repr__(self):
        return '{name}({{{items}}})'.format(
            name=self.__class__.__name__,
            items=', '.join('{!r}: {!r}'.format(k, v) for k, v
                            in self.items()))
