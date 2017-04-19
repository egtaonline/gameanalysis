"""Module for finding fixed points of functions on a simplex"""
import numpy as np

from gameanalysis import utils


def fixed_point(func, init, **kwargs):
    """Compute an approximate fixed point of a function

    Parameters
    ----------
    func : ndarray -> ndarray
        A continuous function mapping from the d-simplex to itself.
    init : ndarray
        An initial guess for the fixed point. Since many may exist, the choice
        of starting point will affect the solution.
    kwargs : options
        Additional options to pass on to labeled_subsimplex. See other options
        for details.
    """
    def fixed_func(mix):
        mix.setflags(write=False)
        return np.argmin(func(mix) - mix + (mix == 0))

    return labeled_subsimplex(fixed_func, init, **kwargs)


def labeled_subsimplex(label_func, init, tol=1e-3, stop=None, init_disc=1):
    """Find approximate center of a fully labeled subsimplex

    Parameters
    ----------
    label_func : ndarray -> int
        A proper lableing function. A labeling function takes an element of the
        d-simplex and returns a label in [0, d). It is proper if the label
        always coresponds to a dimension in support.
    init : ndarray
        An initial guess for where the fully labeled element might be. This
        will be projected onto the simplex if it is not already.
    tol : float, optional
        The tolerance for the returned value.
    stop : ndarray -> bool
        Function of the current simplex that returns true when the search
        should stop. By default, this stops when the sub-simplex has sides
        smaller than tol.
    init_disc : int, optional
        The initial discretization amount for the mixture. The initial
        discretization relates to the amount of possible starting points, which
        may achieve different subsimplicies. This setting this higher may make
        finding one subsimplex slower, but allow the possibility of finding
        more. This function uses the `max(init.size, init_disc, 8)`.

    Notes
    -----
    Implementation from [1]_ and [2]_

    .. [1] Kuhn and Mackinnon 1975. Sandwich Method for Finding Fixed Points.
    .. [2] Kuhn 1968. Simplicial Approximation Of Fixed Points.
    """
    k = max(init.size, init_disc, 8)
    # XXX There's definitely a more principled way to set `2 / k`
    thresh = round(1 / tol) * 2 + 1

    if stop is None:
        def stop(_):
            return k > thresh

    disc_simplex = _discretize_mixture(utils.simplex_project(init), k)
    sub_simplex = disc_simplex / k
    while not stop(sub_simplex):
        disc_simplex = _sandwich(label_func, disc_simplex) * 2
        k = (k - 1) * 2
        sub_simplex = disc_simplex / k
    return sub_simplex


def _discretize_mixture(mix, k):
    """Discretize a mixture

    The returned value will have all integer components that sum to k, with the
    minimum error. Thus, discretizing the mixture.
    """
    disc = np.floor(mix * k).astype(int)
    inds = np.argsort(disc - mix * k)[:k - disc.sum()]
    disc[inds] += 1
    return disc


def _sandwich(label_func, init):
    """Actual implementation of the sandwich method

    Parameters
    ----------
    label_func : ndarray -> int
        See sandwich.
    init : ndarray
        A discretized simplex, e.g. an array of nonnegative integers.

    Returns
    -------
    ret : ndarray
        A discretized simplex with 1 coarser resolution (i.e. ret.sum() + 1 ==
        init.sum()) that is fully labeled.
    """
    dim = init.size
    disc = init.sum()
    # Base vertex of the subsimplex currently being used
    base = np.append(init, 0)
    base[0] += 1
    # permutation array of [1,dim] where v0 = base,
    # v{i+1} = [..., vi_{perms[i] - 1} - 1, vi_{perms[i]} + 1, ...]
    perms = np.arange(1, dim + 1)
    # Array of labels for each vertex
    labels = np.arange(dim + 1)
    labels[dim] = label_func(init / disc)
    # Vertex used to label initial vertices (vertex[-1] == 0)
    label_vertex = base[:-1].copy()
    # Last index moved
    index = dim
    # Most recent created index, should be set to
    new_vertex = None

    while labels[index] < dim:
        # Find duplicate index. this is O(dim) but not a bottleneck
        dup_labels, = np.nonzero(labels == labels[index])
        index, = dup_labels[dup_labels != index]

        # Flip simplex over at index
        if index == 0:
            base[perms[0]] += 1
            base[perms[0] - 1] -= 1
            perms = np.roll(perms, -1)
            labels = np.roll(labels, -1)
            index = dim

        elif index == dim:
            base[perms[-1] - 1] += 1
            base[perms[-1]] -= 1
            perms = np.roll(perms, 1)
            labels = np.roll(labels, 1)
            index = 0

        else:  # 0 < index < dim
            perms[index - 1], perms[index] = perms[index], perms[index - 1]

        # Compute actual value of flipped vertex
        new_vertex = base.copy()
        new_vertex[perms[:index]] += 1
        new_vertex[perms[:index] - 1] -= 1

        assert np.all(new_vertex >= 0) and new_vertex.sum() == disc + 1, \
            "vertex rotation failed, check labeling function"

        # Update label of new vertex
        if new_vertex[-1] == 2:
            labels[index] = dim
        elif new_vertex[-1] == 0:
            labels[index] = np.argmax(new_vertex[:-1] - label_vertex)
        else:  # == 1
            labels[index] = label_func(new_vertex[:-1] / disc)
            assert (0 <= labels[index] < dim and
                    new_vertex[labels[index]]), \
                "labeling function was not proper (see help)"

    return new_vertex[:-1]
