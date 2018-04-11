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
        """Labeling function for a fixed point"""
        return np.argmin((mix == 0) - mix + func(mix))
    return labeled_subsimplex(fixed_func, init, **kwargs)


def labeled_subsimplex(label_func, init, disc): # pylint: disable=too-many-locals,too-many-statements
    """Find approximate center of a fully labeled subsimplex

    This runs once at the discretization provided. It is recommended that this
    be run several times with successively finer discretization and warm
    started with the past result.

    Parameters
    ----------
    label_func : ndarray -> int
        A proper lableing function. A labeling function takes an element of the
        d-simplex and returns a label in [0, d). It is proper if the label
        always coresponds to a dimension in support.
    init : ndarray
        An initial guess for where the fully labeled element might be. This
        will be projected onto the simplex if it is not already.
    disc : int
        The discretization to use. Fixed points will be approximated by the
        reciprocal this much.

    Returns
    -------
    ret : ndarray
        A discretized simplex with 1 coarser resolution (i.e. ret.sum() + 1 ==
        init.sum()) that is fully labeled.

    Notes
    -----
    This is an implementation of the sandwhich method from [5]_ and [6]_

    .. [5] Kuhn and Mackinnon 1975. Sandwich Method for Finding Fixed Points.
    .. [6] Kuhn 1968. Simplicial Approximation Of Fixed Points.
    """
    init = np.asarray(init, float)
    dim = init.size
    # Base vertex of the subsimplex currently being used
    dinit = _discretize_mixture(init, disc)
    base = np.append(dinit, 0)
    base[0] += 1
    # permutation array of [1,dim] where v0 = base,
    # v{i+1} = [..., vi_{perms[i] - 1} - 1, vi_{perms[i]} + 1, ...]
    perms = np.arange(1, dim + 1)
    # Array of labels for each vertex
    labels = np.arange(dim + 1)
    labels[dim] = label_func(dinit / disc)
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

        utils.check(
            np.all(new_vertex >= 0) and new_vertex.sum() == disc + 1,
            'vertex rotation failed, check labeling function')

        # Update label of new vertex
        if new_vertex[-1] == 2:
            labels[index] = dim
        elif new_vertex[-1] == 0:
            labels[index] = np.argmax(new_vertex[:-1] - label_vertex)
        else:  # == 1
            labels[index] = label_func(new_vertex[:-1] / disc)
            utils.check(
                0 <= labels[index] < dim and new_vertex[labels[index]],
                'labeling function was not proper (see help)')

    # Average out all vertices in simplex we care about
    current = base
    if index == 0:  # pragma: no cover
        count = 0
        mean = np.zeros(dim)
    else:  # pragma: no cover
        count = 1
        mean = current.astype(float)
    for i, j in enumerate(perms, 1):
        current[j] += 1
        current[j - 1] -= 1
        if i != index:
            count += 1
            mean += (current - mean) / count
    return mean[:-1] / disc


def _discretize_mixture(mix, k):
    """Discretize a mixture

    The returned value will have all integer components that sum to k, with the
    minimum error. Thus, discretizing the mixture.
    """
    disc = np.floor(mix * k).astype(int)
    inds = np.argsort(disc - mix * k)[:k - disc.sum()]
    disc[inds] += 1
    return disc
