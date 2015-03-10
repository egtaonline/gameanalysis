"""Useful containers"""
import collections
import heapq

class priorityqueue(object):
    """Priority queue with more sensible interface"""
    def __init__(self, init_elems=[]):
        # pylint: disable=dangerous-default-value
        self._elems = list(init_elems)
        heapq.heapify(self._elems)

    def append(self, elem):
        """Add an element to the priority queue"""
        heapq.heappush(self._elems, elem)

    def pop(self):
        """Removes the highest priority element from the queue"""
        return heapq.heappop(self._elems)

    def extend(self, iterable):
        """Extends the priority queue with the supplied iterable"""
        for elem in iterable:
            self.append(elem)

    def drain(self):
        """consuming iterable of the queue in sorted order"""
        while self:
            yield self.pop()

    def __len__(self):
        return len(self._elems)

    def __iter__(self):
        return iter(self._elems)

    def __repr__(self):
        return "<" + repr(self._elems)[1:-1] + ">"


class frozendict(collections.Mapping):
    """Immutable frozen dictionary"""
    # pylint: disable=too-few-public-methods

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return repr(self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.iteritems()))
        return self._hash
