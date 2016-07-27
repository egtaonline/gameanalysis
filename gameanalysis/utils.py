import functools
import operator
import warnings

import numpy as np
import scipy.misc as spm


def prod(collection):
    """Product of all elements in the collection"""
    return functools.reduce(operator.mul, collection)


def game_size(players, strategies, exact=False):
    """Number of profiles in a symmetric game with players and strategies"""
    if exact:
        return spm.comb(players+strategies-1, players, exact=True)
    else:
        sizes = np.rint(spm.comb(players+strategies-1, players,
                                 exact=False)).astype(int)
        assert np.all(sizes >= 0), "Overflow on game size"
        return sizes


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
        return string[:3*line_width//4] + "..." + string[-line_width//4+3:]
    return string


# def weighted_least_squares(x, y, weights):
#     """appends the ones for you; puts 1D weights into a diagonal matrix"""
#     try:
#         A = np.append(x, np.ones([x.shape[0],1]), axis=1)
#         W = np.zeros([x.shape[0]]*2)
#         np.fill_diagonal(W, weights)
#         return y.T.dot(W).dot(A).dot(np.linalg.inv(A.T.dot(W).dot(A)))
#     except np.linalg.linalg.LinAlgError:
#         z = A.T.dot(W).dot(A)
#         for i in range(z.shape[0]):
#             for j in range(z.shape[1]):
#                 z[i,j] += np.random.uniform(-tiny,tiny)
#         return y.T.dot(W).dot(A).dot(np.linalg.inv(z))


def _reverse(seq, start, end):
    """Helper function needed for ordered_permutations"""
    end -= 1
    if end <= start:
        return
    while True:
        seq[start], seq[end] = seq[end], seq[start]
        if start == end or start+1 == end:
            return
        start += 1
        end -= 1


def compare_by_key(key):
    """Decorator that adds object comparison via a key function"""
    def decorator(cls):
        setattr(cls, '__eq__', lambda self, other: key(self) == key(other))
        setattr(cls, '__ne__', lambda self, other: key(self) != key(other))
        setattr(cls, '__le__', lambda self, other: key(self) <= key(other))
        setattr(cls, '__gw__', lambda self, other: key(self) >= key(other))
        setattr(cls, '__gt__', lambda self, other: key(self) > key(other))
        setattr(cls, '__lt__', lambda self, other: key(self) < key(other))
        return cls
    return decorator


def ordered_permutations(seq):
    """Return an iterable over all of the permutations in seq

    The elements of seq must be orderable. The permutations are taken relative
    to the value of the items in seq, not just their index. Thus:

    >>> list(ordered_permutations([1, 2, 1]))
    [(1, 1, 2), (1, 2, 1), (2, 1, 1)]

    This function is taken from this blog post:
    http://blog.bjrn.se/2008/04/lexicographic-permutations-using.html
    And this stack overflow post:
    https://stackoverflow.com/questions/6534430/why-does-pythons-itertools-permutations-contain-duplicates-when-the-original
    """
    seq = sorted(seq)
    if not seq:
        return
    first = 0
    last = len(seq)
    yield tuple(seq)
    if last == 1:
        return
    while True:
        next = last - 1
        while True:
            next1 = next
            next -= 1
            if seq[next] < seq[next1]:
                mid = last - 1
                while seq[next] >= seq[mid]:
                    mid -= 1
                seq[next], seq[mid] = seq[mid], seq[next]
                _reverse(seq, next1, last)
                yield tuple(seq)
                break
            if next == first:
                return


def acomb(n, k):
    """Compute an array of all n choose k options with repeats

    The result will be an array shape (m, n) where m is n choose k with
    repetitions. Each row is a unique way to allocate k ones to m bins.
    """
    # This uses dynamic programming to compute everything
    num = spm.comb(n, k, repetition=True, exact=True)
    grid = np.zeros((num, n), dtype=int)

    memoized = np.empty((n - 1, k), dtype=object)

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
        saved = memoized[n - 2, k - 1]
        if saved is not None:
            np.copyto(region, saved)
            return
        memoized[n - 2, k - 1] = region
        o = 0
        for ki in range(k, -1, -1):
            n_ = n - 1
            k_ = k - ki
            m = spm.comb(n_, k_, repetition=True, exact=True)
            region[o:o+m, 0] = ki
            fill_region(n_, k_, region[o:o+m, 1:])
            o += m

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
    size = array.shape[-1]
    sort = -np.sort(-array)
    rho = (1 - sort.cumsum(-1)) / np.arange(1, size + 1)
    inds = size - 1 - np.argmax((rho + sort > 0)[..., ::-1], -1)
    rho.shape = (-1, size)
    lam = rho[np.arange(rho.shape[0]), inds.flat]
    lam.shape = array.shape[:-1] + (1,)
    return np.maximum(array + lam, 0)


def multinomial_mode(p, n):
    """Compute the mode of n samples from multinomial distribution p.

    algorithm from: Finucan 1964. The mode of a multinomial distribution.
    notation follows: Gall 2003. Determination of the modes of a Multinomial
                      distribution.
    """
    f = p * (n + p.size/2)
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
    return k


def deprecated(func):
    """Decorator which marks functions as deprecated"""

    @functools.wraps(func)
    def deprecation_wrapper(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return deprecation_wrapper


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
    # ascontiguousarray will make a copy of necessary
    axis_at_end = np.ascontiguousarray(np.rollaxis(array, axis, array.ndim))
    new_shape = axis_at_end.shape
    elems = axis_at_end.view(np.dtype((np.void, array.itemsize *
                                       new_shape[-1])))
    elems.shape = new_shape[:-1]
    return elems


def elem_to_axis(array, dtype, axis=-1):
    """Converts and array of axis elements back to an axis"""
    return np.rollaxis(array.view(dtype).reshape(array.shape + (-1,)),
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
    axis_length = array.shape[axis]
    elems = axis_to_elem(array, axis)
    results = np.unique(elems, **kwargs)
    if isinstance(results, tuple):
        return ((results[0].view(array.dtype).reshape((-1, axis_length)),) +
                results[1:])
    else:
        return results.view(array.dtype).reshape((-1, axis_length))
