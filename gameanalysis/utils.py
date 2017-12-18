import functools
import inspect
import itertools
import math
import operator
import random
import string
import warnings
from collections import abc

import numpy as np
import scipy.special as sps


_TINY = np.finfo(float).tiny
# This is the maximum integer that can be exactly represented as a float
_MAX_INT_FLOAT = 2 ** (np.finfo(float).nmant - 1)
_SIMPLEX_BIG = 1 / np.finfo(float).resolution
# XXX A lot of these are candidates for cython


def prod(collection):
    """Product of all elements in the collection"""
    return functools.reduce(operator.mul, collection)


def comb(n, k):
    """Return n choose k

    This function works on arrays, and will properly return a python integer
    object if the number is too large to be stored in a 64 bit integer.
    """
    res = np.rint(sps.comb(n, k, False))
    if np.all(res < _MAX_INT_FLOAT):
        return res.astype(int)
    elif isinstance(n, abc.Iterable) or isinstance(k, abc.Iterable):
        broad = np.broadcast(np.asarray(n), np.asarray(k))
        res = np.empty(broad.shape, dtype=object)
        res.flat = [sps.comb(n_, k_, True) for n_, k_ in broad]
        return res
    else:
        return sps.comb(n, k, True)


# TODO This does a linear scan from an approximate lower bound. This can be
# done with a logarithmic scan
def comb_inv(cmb, k):
    """Return the inverse of `comb`
    
    Given a number of combinations, and the size of subset we're choosing,
    compute the integer lower bound, i.e. return n* such that `comb(n*, k) <=
    cmb < comb(n* + 1, k)`.
    """
    n = np.empty(np.broadcast(cmb, k).shape, int)
    na = n.view()
    na.shape = (n.size,)
    cmba = np.broadcast_to(cmb, n.size)
    ka = np.broadcast_to(k, n.size)
    mask = ka > 0
    na[~mask] = 0
    na[mask] = np.ceil((ka[mask] / np.e * cmba[mask] ** (1 / ka[mask])).astype(float))
    mask[mask] = comb(na[mask] + 1, ka[mask]) <= cmba[mask]
    while mask.any():
        na[mask] += 1
        mask[mask] = comb(na[mask] + 1, ka[mask]) <= cmba[mask]
    if n.ndim == 0:
        return n.item()
    else:
        return n


def game_size(players, strategies):
    """Number of profiles in a symmetric game with players and strategies"""
    return comb(np.asarray(players) + strategies - 1, players)


def game_size_inv(size, players):
    """Inverse of game_size
    
    Given a game size and a number of players, return a lower bound on the
    number of strategies s* such that game_size(players, s*) <= size <
    game_size(players, s* + 1)`.
    """
    return comb_inv(size, players) - players + 1


def only(iterable):
    """Return the only element of an iterable

    Throws a value error if the iterable doesn't contain only one element
    """
    try:
        it = iter(iterable)
        value = next(it)
        try:
            next(it)
        except StopIteration:
            return value
        raise ValueError('Iterable had more than one element')
    except TypeError:
        raise ValueError('Input was not iterable')
    except StopIteration:
        raise ValueError('Input was empty')


def one_line(string, line_width=80):
    """If string s is longer than line width, cut it off and append "..."
    """
    string = string.replace('\n', ' ')
    if len(string) > line_width:
        return "{}...{}".format(string[:3 * line_width // 4],
                                string[-line_width // 4 + 3:])
    return string


def acomb(n, k, repetition=False):
    """Compute an array of all n choose k options

    The result will be an array shape (m, n) where m is n choose k optionally
    with repetitions."""
    if repetition:
        return _acombr(n, k)
    else:
        return _acomb(n, k)


def _acombr(n, k):
    """Combinations with repetitions"""
    # This uses dynamic programming to compute everything
    num = sps.comb(n, k, repetition=True, exact=True)
    grid = np.zeros((num, n), dtype=int)
    memoized = {}

    # This recursion breaks if asking for numbers that are too large (stack
    # overflow), but the order to fill n and k is predictable, it may be better
    # to to use a for loop.
    def fill_region(n, k, region):
        if n == 1:
            region[0, 0] = k
            return
        elif k == 0:
            region.fill(0)
            return
        if (n, k) in memoized:
            np.copyto(region, memoized[n, k])
            return
        memoized[n, k] = region
        o = 0
        for ki in range(k, -1, -1):
            n_ = n - 1
            k_ = k - ki
            m = sps.comb(n_, k_, repetition=True, exact=True)
            region[o:o + m, 0] = ki
            fill_region(n_, k_, region[o:o + m, 1:])
            o += m

    fill_region(n, k, grid)
    return grid


def _acomb(n, k):
    """Combinations"""
    if k == 0:
        return np.zeros((1, n), bool)

    # This uses dynamic programming to compute everything
    num = sps.comb(n, k, exact=True)
    grid = np.empty((num, n), dtype=bool)
    memoized = {}

    # This recursion breaks if asking for numbers that are too large (stack
    # overflow), but the order to fill n and k is predictable, it may be better
    # to to use a for loop.
    def fill_region(n, k, region):
        if n <= k:
            region.fill(True)
            return
        elif k == 1:
            region.fill(False)
            np.fill_diagonal(region, True)
            return
        if (n, k) in memoized:
            np.copyto(region, memoized[n, k])
            return

        memoized[n, k] = region
        trues = sps.comb(n - 1, k - 1, exact=True)
        region[:trues, 0] = True
        fill_region(n - 1, k - 1, region[:trues, 1:])
        region[trues:, 0] = False
        fill_region(n - 1, k, region[trues:, 1:])

    fill_region(n, k, grid)
    return grid


def acartesian2(*arrays):
    """Array cartesian product in 2d

    Produces a new ndarray that has the cartesian product of every row in the
    input arrays. The number of columns is the sum of the number of columns in
    each input. The number of rows is the product of the number of rows in each
    input.

    Arguments
    ---------
    *arrays : [ndarray (xi, s)]
    """
    rows = prod(a.shape[0] for a in arrays)
    columns = sum(a.shape[1] for a in arrays)
    dtype = arrays[0].dtype  # should always have at least one role
    assert all(a.dtype == dtype for a in arrays), \
        "all arrays must have the same dtype"

    result = np.zeros((rows, columns), dtype)
    pre_row = 1
    post_row = rows
    pre_column = 0
    for array in arrays:
        length, width = array.shape
        post_row //= length
        post_column = pre_column + width
        view = result[:, pre_column:post_column]
        view.shape = (pre_row, -1, post_row, width)
        np.copyto(view, array[:, None])
        pre_row *= length
        pre_column = post_column

    return result


def simplex_project(array):
    """Return the projection onto the simplex"""
    array = np.asarray(array, float)
    assert not np.isnan(array).any(), \
        "can't project nan onto simplex: {}".format(array)
    # This fails for really large values, so we normalize the array so the
    # largest element has absolute value at most _SIMPLEX_BIG
    array = np.clip(array, -_SIMPLEX_BIG, _SIMPLEX_BIG)
    size = array.shape[-1]
    sort = -np.sort(-array, -1)
    rho = (1 - sort.cumsum(-1)) / np.arange(1, size + 1)
    inds = size - 1 - np.argmax((rho + sort > 0)[..., ::-1], -1)
    rho.shape = (-1, size)
    lam = rho[np.arange(rho.shape[0]), inds.flat]
    lam.shape = array.shape[:-1] + (1,)
    return np.maximum(array + lam, 0)


def multinomial_mode(p, n):
    """Compute the mode of n samples from multinomial distribution p.

    Notes
    -----
    Algorithm from [3]_, notation follows [4]_.

    .. [3] Finucan 1964. The mode of a multinomial distribution.
    .. [4] Gall 2003. Determination of the modes of a Multinomial distribution.
    """
    p = np.asarray(p, float)
    mask = p > 0
    result = np.zeros(p.size, int)

    p = p[mask]
    f = p * (n + p.size / 2)
    k = f.astype(int)
    f -= k
    n0 = k.sum()
    if n0 < n:
        q = (1 - f) / (k + 1)
        for _ in range(n0, n):
            a = q.argmin()
            k[a] += 1
            f[a] -= 1
            q[a] = (1 - f[a]) / (k[a] + 1)
    elif n0 > n:
        with np.errstate(divide='ignore'):
            q = f / k
            for _ in range(n, n0):
                a = q.argmin()
                k[a] -= 1
                f[a] += 1
                q[a] = f[a] / k[a]

    result[mask] = k
    return result


def axis_to_elem(array, axis=-1):
    """Converts an axis of an array into a unique element

    In general, this returns a copy of the array, unless the data is
    contiguous. This usually requires that the last axis is the one being
    merged.

    Parameters
    ----------
    array : ndarray
        The array to convert an axis to a view.
    axis : int, optional
        The axis to convert into a single element. Defaults to the last axis.
    """
    array = np.asarray(array)
    # ascontiguousarray will make a copy of necessary
    axis_at_end = np.ascontiguousarray(np.rollaxis(array, axis, array.ndim))
    new_shape = axis_at_end.shape
    elems = axis_at_end.view(np.dtype((np.void, array.itemsize *
                                       new_shape[-1])))
    elems.shape = new_shape[:-1]
    return elems


def elem_to_axis(array, dtype, axis=-1):
    """Converts and array of axis elements back to an axis"""
    dtype = np.dtype(dtype)
    new_shape = array.shape + (array.itemsize // dtype.itemsize,)
    return np.rollaxis(array.view(dtype).reshape(new_shape),
                       -1, axis)


def unique_axis(array, axis=-1, **kwargs):
    """Find unique axis elements

    Parameters
    ----------
    array : ndarray
        The array to find unique axis elements of
    axis : int, optional
        The axis to find unique elements of. Defaults to the last axis.
    **kwargs : flags
        The flags to pass to numpys unique function

    Returns
    -------
    uniques : ndarray
        The unique axes as rows of a two dimensional array.
    *args :
        Any other results of the unique functions due to flags
    """
    elems = axis_to_elem(array, axis)
    results = np.unique(elems, **kwargs)
    if isinstance(results, tuple):
        return ((elem_to_axis(results[0], array.dtype, axis),) +
                results[1:])
    else:
        return elem_to_axis(results, array.dtype, axis)


class hash_array(object):

    def __init__(self, array):
        self.array = np.asarray(array)
        self.array.setflags(write=False)
        self._hash = hash(self.array.tobytes())

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (self._hash == other._hash and
                np.all(self.array == other.array))


def random_con_bitmask(prob, shape, mins=1):
    """Generate a random bitmask with constraints

    The functions allows specifying the minimum number of True values along a
    single dimension while counting over the other ones. `mins` can be a scalar
    or a tuple for each dimension and must be less than the product of the size
    of the other dimensions.

    If you just want a random bitmask use np.random.random(shape) < prob"""
    assert len(shape) > 1
    vals = np.random.random(shape)
    mask = vals < prob
    total = vals.size

    if isinstance(mins, abc.Sequence):
        assert len(mins) == vals.ndim
        assert all(0 < s <= total // m for s, m in zip(mins, vals.shape))
    else:
        assert mins > 0
        mins = tuple(min(mins, total // m) for m in vals.shape)

    for dim, num in enumerate(mins):
        aligned = np.rollaxis(vals, dim).reshape(vals.shape[dim], -1)
        thresh = np.partition(aligned, num - 1, 1)[:, num - 1]
        thresh.shape += (1,) * (vals.ndim - dim - 1)
        mask |= vals <= thresh

    return mask


def iunique(iterable):
    """Return an iterable of unique items ordered by first occurrence"""
    seen = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item


def random_strings(min_length, max_length=None, digits=string.ascii_lowercase):
    """Return a random string

    Parameters
    ----------
    min_length : int
        The minimum length string to return.
    max_length : int, optional
        The maximum length string to return. If None or unspecified, this is
        the same as min_length.
    num : int, optional
        The number of strings to return. If None or unspecified this returns a
        single string, otherwise it returns a generator with length `num`.
    digits : str, optional
        The optional digits to select from.
    """
    if max_length is None:
        max_length = min_length
    assert min_length <= max_length, \
        "max_length can't be less than min_length"
    while True:
        length = random.randint(min_length, max_length)
        yield ''.join(random.choice(digits) for _ in range(length))


def prefix_strings(prefix, num):
    """Returns a list of prefixed integer strings"""
    padding = int(math.log10(max(num - 1, 1))) + 1
    return ('{}{:0{:d}d}'.format(prefix, i, padding) for i in range(num))


def is_sorted(iterable, *, key=None, reverse=False, strict=False):
    """Returns true if iterable is sorted

    Parameters
    ----------
    iterable : iterable
        The iterable to check for sorted-ness.
    key : x -> y, optional
        Apply mapping function key to iterable prior to checking. This can be
        done before calling, but this ensures identical calls as sorted.
    reverse : bool, optional
    `key` and `reverse` function as they for `sorted`"""
    if key is not None:
        iterable = map(key, iterable)

    ai, bi = itertools.tee(iterable)
    next(bi, None)  # Don't throw error if empty

    if strict and reverse:
        return all(a > b for a, b in zip(ai, bi))
    elif reverse:
        return all(a >= b for a, b in zip(ai, bi))
    elif strict:
        return all(a < b for a, b in zip(ai, bi))
    else:
        return all(a <= b for a, b in zip(ai, bi))


def memoize(member_function):
    """Memoize computation of single object functions"""
    assert len(inspect.signature(member_function).parameters) == 1, \
        "Can only memoize single object functions"
    member_name = '__' + member_function.__name__

    @functools.wraps(member_function)
    def new_member_function(obj):
        if not hasattr(obj, member_name):
            setattr(obj, member_name, member_function(obj))
        return getattr(obj, member_name)

    return new_member_function


def deprecated(func):
    """Mark a function as deprecated"""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapped
