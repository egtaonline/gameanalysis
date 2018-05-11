"""Module for base role symmetric game structures

Role symmetric games have a number of common attributes and functions that are
defined in the RsGame class that actual RsGame interfaces should support. In
addition, an implementation of an EmptyGame is provided for convenience
purposes. Note, that the constructor to EmptyGame (and most games) should not
be called, and instead, various convenience functions to create EmptyGames that
start with the prefix `emptygame` should be called instead.

Most structures in a role symmetric game are an array of length
game.num_strats, that lists a value for every strategy in the game. In this
form, each roles strategies are contiguous. To aggregate elements by role, it
is helpful to use numpy ufuncs, such as `np.add.reduceat(profile,
game.role_starts)` will add up all of the players in each role of a profile.
`np.split(profile, game.role_starts[1:])` will return a list where each element
is that role's data. To convert a role array into a strategy array, one can do
something like `role_array.repeat(game.num_role_strats)`.

As a general rule, any attribute of a game that begins with `num_` is an actual
attribute instead of a getter function. Attributes that have the world role in
them tend to be arrays of size `num_roles` and attributes that have `strat` in
the name tend to be arrays of size `num_strats`.
"""
# pylint: disable=too-many-lines
import abc
import contextlib
import functools
import itertools
import string
import warnings
import collections.abc as cabc

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import gamereader
from gameanalysis import utils


class _StratArray(abc.ABC): # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """A class with knowledge of the number of strategies per role

    This has methods common to working with strategy arrays, which essentially
    represent points in a simplotope (a cross product of simplicies), or points
    in a discretized simplotope.

    Parameters
    ----------
    role_names : (str,)
        The name of each role. Names must be unique, sorted, and can only
        contain printable ascii characters less semi-colon and colon.
    strat_names : ((str,),)
        The name of each strategy for each role. Must be the same length as
        role_names. Names must be sorted and unique per role, and can only
        contain printable ascii characters less semi-colon and comma. Must have
        at lease one strategy per role.
    """

    def __init__(self, role_names, strat_names):
        self.num_roles = len(role_names)
        self.num_role_strats = np.fromiter(  # pragma: no branch
            (len(s) for s in strat_names), int, self.num_roles)
        self.num_strats = self.num_role_strats.sum()
        self.role_starts = np.insert(self.num_role_strats[:-1].cumsum(), 0, 0)
        self.role_indices = np.arange(self.num_roles).repeat(
            self.num_role_strats)

        self.role_names = role_names
        self.strat_names = strat_names
        self._named_role_index = {r: i for i, r in enumerate(self.role_names)}
        self._role_strat_index = {
            (r, s): i for i, (r, s)
            in enumerate(itertools.chain.from_iterable(
                ((r, s) for s in strats) for r, strats
                in zip(self.role_names, self.strat_names)))}
        self._role_strat_dev_index = {
            (r, s, d): i for i, (r, s, d)
            in enumerate(itertools.chain.from_iterable(
                itertools.chain.from_iterable(
                    ((r, s, d) for d in strats if d != s)
                    for s in strats)
                for r, strats
                in zip(self.role_names, self.strat_names)))}

        self.num_role_strats.setflags(write=False)
        self.role_starts.setflags(write=False)
        self.role_indices.setflags(write=False)

    @property
    @utils.memoize
    def role_strat_names(self):
        """role and strat names indexed by role strat"""
        return tuple(itertools.chain.from_iterable(
            ((r, s) for s in ses) for r, ses
            in zip(self.role_names, self.strat_names)))

    @property
    @utils.memoize
    def num_all_restrictions(self):
        """Number of unique restrictions"""
        return np.prod(2 ** self.num_role_strats - 1)

    @property
    @utils.memoize
    def num_pure_restrictions(self):
        """The number of pure restrictions

        A pure restrictions has exactly one strategy per role."""
        return self.num_role_strats.prod()

    @property
    @utils.memoize
    def num_strat_devs(self):
        """The number of deviations for each strategy"""
        devs = np.repeat(self.num_role_strats - 1, self.num_role_strats)
        devs.setflags(write=False)
        return devs

    @property
    @utils.memoize
    def num_role_devs(self):
        """The number of deviations for each role"""
        devs = (self.num_role_strats - 1) * self.num_role_strats
        devs.setflags(write=False)
        return devs

    @property
    @utils.memoize
    def num_devs(self):
        """The total number of deviations"""
        return self.num_role_devs.sum()

    @property
    @utils.memoize
    def dev_strat_starts(self):
        """The start index for each strategy deviation"""
        if np.any(self.num_role_strats == 1):
            warnings.warn(
                'using reduceat with dev_strat_starts will not produce '
                'correct results if any role only has one strategy. This '
                'might get fixed at some point, but currently extra care must '
                'be taken for these cases.')
        starts = np.insert(self.num_strat_devs[:-1].cumsum(), 0, 0)
        starts.setflags(write=False)
        return starts

    @property
    @utils.memoize
    def dev_role_starts(self):
        """The start index for each role deviation"""
        if np.any(self.num_role_strats == 1):
            warnings.warn(
                'using reduceat with dev_role_starts will not produce '
                'correct results if any role only has one strategy. This '
                'might get fixed at some point, but currently extra care must '
                'be taken for these cases.')
        starts = np.insert(self.num_role_devs[:-1].cumsum(), 0, 0)
        starts.setflags(write=False)
        return starts

    @property
    @utils.memoize
    def dev_from_indices(self):
        """The strategy deviating from for each deviation"""
        inds = np.arange(self.num_strats).repeat(self.num_strat_devs)
        inds.setflags(write=False)
        return inds

    @property
    @utils.memoize
    def dev_to_indices(self):
        """The strategy deviating to for each deviation"""
        inds = (np.arange(self.num_devs) -
                self.dev_strat_starts.repeat(self.num_strat_devs) +
                self.role_starts.repeat(self.num_role_devs))

        # The use of bincount here allows for one strategy roles
        pos_offset = np.bincount(np.arange(self.num_strats) -
                                 self.role_starts.repeat(self.num_role_strats)
                                 + self.dev_strat_starts,
                                 minlength=self.num_devs + 1)[:-1]
        neg_offset = np.bincount(self.dev_strat_starts[1:],
                                 minlength=self.num_devs + 1)[:-1]
        inds += np.cumsum(pos_offset - neg_offset)

        inds.setflags(write=False)
        return inds

    def all_restrictions(self):
        """Return all valid restrictions"""
        role_subs = [(np.arange(1, 1 << num_strats)[:, None]
                      & (1 << np.arange(num_strats))).astype(bool)
                     for num_strats
                     in self.num_role_strats]
        return utils.acartesian2(*role_subs)

    def pure_restrictions(self):
        """Returns every pure restriction in a game

        A pure restriction has only one strategy per role. This returns the
        pure restrictions in sorted order based off of role and strategy."""
        role_rests = [np.eye(num_strats, dtype=bool) for num_strats
                      in self.num_role_strats]
        return utils.acartesian2(*role_rests)

    def random_restriction(self, *, strat_prob=None, normalize=True):
        """Return a random restriction

        See random_restrictions"""
        return self.random_restrictions(1, strat_prob=strat_prob,
                                        normalize=normalize)[0]

    def random_restrictions(self, num_samples, *, strat_prob=None,
                            normalize=True):
        """Return random restrictions

        Parameters
        ----------
        num_samples : int
            The number of restrictions to be returned.
        strat_prob : float or [float], optional
            The probability that a given strategy is in support. If
            `strat_prob` is None, supports will be sampled uniformly. This can
            either be a scalar, or an iterable of length `num_roles`.
        normalize : bool, optional
            If true, the strategy probabilities are normalized, so that the
            conditional probability of any strategy in support equals
            `strat_prob`. The probability for any strategy must be at least `1
            / num_role_strats` for its role. Individual strategy probabilities
            are thresholded to this value when normalize is set to true.
        """
        if strat_prob is None:
            strat_prob = 1 - ((2 ** (self.num_role_strats - 1) - 1) /
                              (2 ** self.num_role_strats - 1))
        if normalize:
            strat_prob_pre = np.maximum(
                np.broadcast_to(strat_prob, self.num_roles),
                1 / self.num_role_strats)
            strat_prob = np.empty(self.num_roles)
            for i, (strats, prob) in enumerate(zip(
                    self.num_role_strats, strat_prob_pre)):
                if strats <= 1:  # Special case
                    strat_prob[i] = 1
                    continue
                poly = sps.binom(strats, np.arange(strats, -1, -1)) / strats
                poly[-2::-2] *= -1
                poly[-2] += 1
                poly[-1] -= prob
                roots = np.roots(poly)
                strat_prob[i] = roots[np.isreal(roots) & (
                    roots >= 0) & (roots <= prob)][0].real
        rands = rand.random((num_samples, self.num_strats))
        thresh = np.maximum(np.minimum.reduceat(
            rands, self.role_starts, 1), strat_prob)
        return rands <= thresh.repeat(self.num_role_strats, 1)

    def is_restriction(self, restrict, *, axis=-1):
        """Verify that a restriction is valid"""
        restrict = np.asarray(restrict, bool)
        utils.check(
            restrict.shape[axis] == self.num_strats,
            'restriction must have valid shape')
        return np.all(np.bitwise_or.reduceat(restrict, self.role_starts, axis),
                      axis)

    def _is_pure_restriction(self, restrict, *, axis=-1):
        """Test is restriction is pure"""
        return np.all(np.add.reduceat(
            restrict, self.role_starts, axis) == 1, axis)

    def is_pure_restriction(self, restrict, *, axis=-1):
        """Verify a restriction is pure"""
        restrict = np.asarray(restrict, bool)
        return (self.is_restriction(restrict, axis=axis) &
                self._is_pure_restriction(restrict, axis=axis))

    def trim_mixture_support(self, mixture, *, thresh=1e-4, axis=-1):
        """Trims strategies played less than supp_thresh from the support"""
        mixture = np.array(mixture, float)
        utils.check(
            mixture.shape[axis] == self.num_strats,
            'mixtures must have valid shape')
        mixture *= mixture >= thresh
        mixture /= np.add.reduceat(
            mixture, self.role_starts, axis).repeat(self.num_role_strats, axis)
        return mixture

    def trim_mixture_precision(self, mixture, *, resolution=1e-3):
        """Reduce precision of mixture

        This trims the mixture so that it lies on the discretized space, where
        every component is an integer multiple of resolution. By default, trim
        mixture so every component is only accurate to 1 out of 1000, making it
        convenient for serialization. This function returns a valid mixture
        with the minimum absolute error to the input mixture.
        """
        mixture = np.asarray(mixture, float)
        ires = round(1 / resolution)
        utils.check(self.is_mixture(mixture), 'must pass mixtures')
        utils.check(
            np.isclose(ires, 1 / resolution),
            'resolution must be integer inverse')
        pmix = mixture * ires
        imix = np.floor(pmix).astype(int)
        error = imix - pmix
        incs = ires - np.add.reduceat(imix, self.role_starts)
        for rmix, err, inc in zip(
                np.split(imix, self.role_starts[1:]),
                np.split(error, self.role_starts[1:]), incs):
            if inc > 0:
                rmix[np.argpartition(err, inc - 1)[:inc]] += 1
        return imix / ires

    def is_mixture(self, mixture, *, axis=-1):
        """Verify that a mixture is valid for game"""
        mixture = np.asarray(mixture, float)
        utils.check(
            mixture.shape[axis] == self.num_strats,
            'mixture must have valid shape')
        return (np.all(mixture >= 0, axis) &
                np.all(np.isclose(np.add.reduceat(
                    mixture, self.role_starts, axis), 1), axis))

    def is_pure_mixture(self, mixture, *, axis=-1):
        """Verify a mixture is pure"""
        mixture = np.asarray(mixture, float)
        return (self.is_mixture(mixture, axis=axis) &
                self._is_pure_restriction(mixture > 0, axis=axis))

    def mixture_project(self, mixture):
        """Project an array into mixture space"""
        mixture = np.asarray(mixture, float)
        return np.concatenate(
            [utils.simplex_project(r) for r
             in np.split(mixture, self.role_starts[1:], -1)], -1)

    def mixture_to_simplex(self, mixture): # pylint: disable=too-many-locals
        """Convert a mixture to a simplex

        The simplex will have dimension `num_role_strats - num_roles + 1`. This
        uses the ray tracing homotopy. The uniform mixtures are aligned, and
        then rays are extended to the edges to convert proportionally along
        those rays on each object.

        Notes
        -----
        This uses a simple ray alignment algorithm. First line up the uniform
        simplex and the uniform mixture, then define a vector in the mixture as
        all but the last probabilities in each role. Now trace a line from the
        uniform mixture to the current mixture to the edge of the mixture, and
        record what proportion of along that vector the point of interest was.
        Copy that gradient to the simplex, trace it to the boundary, and take
        the point the proportion of the way to the edge.
        """
        # This is relatively simple despite looking verbose. It's going to
        # store the ray as the direction from the uniform mixture to the
        # current mixture (grad)
        mixture = np.asarray(mixture, float)
        center = self.uniform_mixture()
        grad = mixture - center
        # Then we compute alpha, which is the constant to multiply grad by to
        # get a point on an edge, in some sense this is the maximum weighting
        # of the ray, and all valid points lie on w * grad, w \in [0, alpha]
        with np.errstate(divide='ignore'):
            alphas = np.where(np.isclose(grad, 0), np.inf, -center / grad)
        alphas[alphas < 0] = np.inf
        alpha_inds = np.abs(alphas).argmin(-1)[..., None]
        alpha_inds.flat += np.arange(alpha_inds.size) * alphas.shape[-1]
        alpha = alphas.flat[alpha_inds]

        # Now compute the simplex gradient, which just copies the unconstrained
        # gradients for every simplex in the simplotope, and then computes the
        # final element so the gradient sums to 0 (necessary to stay on the
        # simplex) This is done by simply deleting the last element of each
        # role except the last (role_starts[1:] - 1)
        simp_dim = self.num_strats - self.num_roles + 1
        simp_center = np.ones(simp_dim) / simp_dim
        simp_grad = np.delete(grad, self.role_starts[1:] - 1, -1)
        simp_grad[..., -1] = -simp_grad[..., :-1].sum(-1)
        # Then we compute alpha the same way for the simplex, but this is in
        # terms of the simp_grad.
        with np.errstate(divide='ignore'):
            simp_alphas = np.where(np.isclose(simp_grad, 0), np.inf,
                                   -simp_center / simp_grad)
        simp_alphas[simp_alphas < 0] = np.inf
        simp_alpha_inds = np.abs(simp_alphas).argmin(-1)[..., None]
        simp_alpha_inds.flat += np.arange(simp_alpha_inds.size) * \
            simp_alphas.shape[-1]
        simp_alpha = simp_alphas.flat[simp_alpha_inds]
        # The point on the simplex is going to be the ratio of the alphas,
        # where we account for when their both infinite. They're infinite when
        # the mixture is uniform, and when it's uniform there's no ray so we
        # don't change the projection.
        with np.errstate(invalid='ignore'):
            ratio = np.where(np.isposinf(simp_alpha) & np.isposinf(alpha), 0,
                             simp_alpha / alpha)
        simp = simp_center + ratio * simp_grad
        simp *= simp > 0
        return simp / simp.sum(-1, keepdims=True)

    def mixture_from_simplex(self, simp):
        """Convert a simplex back into a valid mixture

        This is the inverse function of mixture_to_simplex."""
        # See to_simplex for an understanding of what these steps are doing.
        simp = np.asarray(simp, float)
        simp_dim = self.num_strats - self.num_roles + 1
        simp_center = np.ones(simp_dim) / simp_dim
        center = self.uniform_mixture()
        simp_grad = simp - simp_center
        with np.errstate(divide='ignore'):
            simp_alphas = np.where(np.isclose(simp_grad, 0), np.inf,
                                   -simp_center / simp_grad)
        simp_alphas[simp_alphas < 0] = np.inf
        simp_alpha_inds = np.abs(simp_alphas).argmin(-1)[..., None]
        simp_alpha_inds.flat += np.arange(simp_alpha_inds.size) * \
            simp_alphas.shape[-1]
        simp_alpha = simp_alphas.flat[simp_alpha_inds]

        grad = np.insert(simp_grad,
                         self.role_starts[1:] - np.arange(1, self.num_roles),
                         0, -1)
        grad[..., -1] = 0
        grad[..., self.role_starts + self.num_role_strats - 1] = \
            -np.add.reduceat(grad, self.role_starts, -1)
        with np.errstate(divide='ignore'):
            alphas = np.where(np.isclose(grad, 0), np.inf, -center / grad)
        alphas[alphas < 0] = np.inf
        alpha_inds = np.abs(alphas).argmin(-1)[..., None]
        alpha_inds.flat += np.arange(alpha_inds.size) * alphas.shape[-1]
        alpha = alphas.flat[alpha_inds]  # >= 1
        with np.errstate(invalid='ignore'):
            ratio = np.where(np.isposinf(simp_alpha) & np.isposinf(alpha), 0,
                             alpha / simp_alpha)
        return self.trim_mixture_support(center + ratio * grad, thresh=0)

    def uniform_mixture(self):
        """Returns the uniform mixed profile"""
        return np.repeat(1 / self.num_role_strats, self.num_role_strats)

    def random_mixture(self, *, alpha=1):
        """Return a random mixture

        See random_mixtures"""
        return self.random_mixtures(1, alpha=alpha)[0]

    def random_mixtures(self, num_samples, *, alpha=1):
        """Return a random mixed profile

        Mixed profiles are sampled from a Dirichlet distribution with parameter
        alpha. If alpha = 1 (the default) this is a uniform distribution over
        the simplex for each role. Alpha \\in (0, 1) is baised towards high
        entropy mixtures, i.e. mixtures where one strategy is played in
        majority. Alpha \\in (1, oo) is baised towards low entropy (uniform)
        mixtures.
        """
        mixtures = rand.gamma(alpha, 1, (num_samples, self.num_strats))
        mixtures /= np.add.reduceat(mixtures, self.role_starts,
                                    1).repeat(self.num_role_strats, 1)
        return mixtures

    def random_sparse_mixture(self, *, alpha=1, support_prob=None,
                              normalize=True):
        """Return a random sparse mixture

        See random_sparse_mixtures"""
        return self.random_sparse_mixtures(
            1, alpha=alpha, support_prob=support_prob, normalize=normalize)[0]

    def random_sparse_mixtures(self, num_samples, *, alpha=1,
                               support_prob=None, normalize=True):
        """Return a random sparse mixed profile

        Parameters
        ----------
        num_samples : int
            The number of mixtures to be returned.
        alpha : float, optional
            Mixed profiles are sampled from a dirichlet distribution with
            parameter alpha. If alpha = 1 (the default) this is a uniform
            distribution over the simplex for each role. Alpha \\in (0, 1) is
            baised towards high entropy mixtures, i.e. mixtures where one
            strategy is played in majority. Alpha \\in (1, oo) is baised towards
            low entropy (uniform) mixtures.
        support_prob : float or [float], optional
            The probability that a given strategy is in support. If support
            prob is None, supports will be sampled uniformly.
        normalize : bool, optional
            If true, the mixtures are normalized, so that the conditional
            probability of any strategy in support equals support prob. If
            true, the support_prob for any role must be at least `1 /
            num_role_strats`.
        """
        mixtures = rand.gamma(alpha, 1, (num_samples, self.num_strats))
        mixtures *= self.random_restrictions(
            num_samples, strat_prob=support_prob, normalize=normalize)
        mixtures /= np.add.reduceat(
            mixtures, self.role_starts, 1).repeat(self.num_role_strats, 1)
        return mixtures

    def biased_mixtures(self, bias=0.9):
        """Generates mixtures biased towards one strategy for each role

        Each role has one strategy played with probability bias; the remaining
        1-bias probability is distributed uniformly over the remaining S or S-1
        strategies. If there's only one strategy, it is played with probability
        1."""
        utils.check(
            0 <= bias <= 1, 'probabilities must be between zero and one')

        role_mixtures = []
        for num_strats in self.num_role_strats:
            if num_strats == 1:
                mixture = np.ones((1, 1))
            else:
                mixture = np.full(
                    (num_strats,) * 2, (1 - bias) / (num_strats - 1))
                np.fill_diagonal(mixture, bias)
            role_mixtures.append(mixture)

        return utils.acartesian2(*role_mixtures)

    def role_biased_mixtures(self, bias=0.9):
        """Generates mixtures where one role-strategy is played with bias

        If no roles have more than one strategy (a degenerate game), then this
        returns nothing."""
        utils.check(
            0 <= bias <= 1, 'probabilities must be between zero and one')

        if np.all(self.num_role_strats == 1):
            return np.ones((1, self.num_roles))

        num = self.num_role_strats[self.num_role_strats > 1].sum()
        mixes = self.uniform_mixture()[None].repeat(num, 0)
        prof_offset = 0
        strat_offset = 0
        for num_strats in self.num_role_strats:
            if num_strats > 1:
                view = mixes[prof_offset:prof_offset + num_strats,
                             strat_offset:strat_offset + num_strats]
                view.fill((1 - bias) / (num_strats - 1))
                np.fill_diagonal(view, bias)
                prof_offset += num_strats
            strat_offset += num_strats
        return mixes

    def pure_mixtures(self):
        """Returns all mixtures where the probability is either 1 or 0."""
        return self.pure_restrictions().astype(float)

    def grid_mixtures(self, num_points):
        """Returns all of the mixtures in a grid with n points

        Arguments
        ---------
        num_points : int > 1
            The number of points to have along one dimensions
        """
        utils.check(
            num_points > 1, 'Must have at least two points on a dimensions')
        role_mixtures = [utils.acomb(num_strats, num_points - 1, True) /
                         (num_points - 1)
                         for num_strats in self.num_role_strats]
        return utils.acartesian2(*role_mixtures)

    def role_index(self, role):
        """Return the index of a role by name"""
        return self._named_role_index[role]

    def role_strat_index(self, role, strat):
        """Return the index of a role strat pair"""
        return self._role_strat_index[role, strat]

    def role_strat_dev_index(self, role, strat, dev):
        """Return the index of a role strat deviating strat pair

        `dev` and `strat` must both be strategies in role, and be distinct."""
        return self._role_strat_dev_index[role, strat, dev]

    def strat_name(self, role_strat_index):
        """Get the strategy name from a full index"""
        role_index = self.role_indices[role_strat_index]
        return self.strat_names[role_index][role_strat_index -
                                            self.role_starts[role_index]]

    def mixture_from_json(self, mixture, dest=None, *, verify=True):
        """Read a json mixture into an array"""
        if dest is None:
            dest = np.empty(self.num_strats, float)
        else:
            utils.check(
                dest.dtype.kind == 'f', 'dest dtype must be floating')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(0)

        for role, strats in mixture.items():
            for strat, prob in strats.items():
                dest[self.role_strat_index(role, strat)] = prob

        utils.check(
            not verify or self.is_mixture(dest),
            '"{}" does not define a valid mixture', mixture)
        return dest

    def _to_arr_json(self, arr):
        """Convert array to json"""
        return {role: {strat: val.item() for strat, val
                       in zip(strats, values) if val != 0}
                for values, role, strats
                in zip(np.split(arr, self.role_starts[1:]),
                       self.role_names, self.strat_names)
                if np.any(values != 0)}

    def mixture_to_json(self, mixture):
        """Convert a mixture array to json"""
        return self._to_arr_json(mixture)

    def _from_arr_repr(self, arr_str, dtype, parse, dest=None):
        """Read an array from a string"""
        if dest is None:
            dest = np.empty(self.num_strats, dtype)
        else:
            utils.check(
                dest.dtype.kind == np.dtype(dtype).kind,
                "dest dtype doesn't match specified dtype")
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(0)
        for role_str in arr_str.split(';'):
            role, strats = (s.strip() for s in role_str.split(':', 1))
            for sstrat in strats.split(','):
                val, strat = (s.strip() for s in sstrat.strip().split(' ', 1))
                dest[self.role_strat_index(role, strat)] = parse(val)
        return dest

    def mixture_from_repr(self, mix_str, dest=None, *, verify=True):
        """Read a mixture from it's repr"""
        mixture = self._from_arr_repr(mix_str, float, _parse_percent, dest)
        utils.check(
            not verify or self.is_mixture(mixture),
            '"{}" is not a valid mixture', mix_str)
        return mixture

    def _to_arr_repr(self, arr, fmt):
        """Convert an array to a string"""
        return '; '.join(
            '{}: {}'.format(role, ', '.join(
                '{:{}} {}'.format(val, fmt, strat)
                for strat, val in zip(strats, values) if val > 0))
            for role, strats, values
            in zip(self.role_names, self.strat_names,
                   np.split(arr, self.role_starts[1:])))

    def mixture_to_repr(self, mixture):
        """Convert a mixture to a string"""
        return self._to_arr_repr(
            self.trim_mixture_precision(mixture, resolution=1e-4), '.2%')

    def _from_arr_str(self, arr_str, dtype, parse, dest=None):
        """Get array from string representation"""
        if dest is None:
            dest = np.empty(self.num_strats, dtype)
        else:
            utils.check(
                dest.dtype.kind == np.dtype(dtype).kind,
                'dest dtype must be the same type as specified')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(0)

        role = None
        for line in arr_str.split('\n'):
            if line[0] != ' ':
                role = line[:-1]
            else:
                strat, val_str = line[4:].split(':', 1)
                dest[self.role_strat_index(role, strat)] = parse(val_str)
        return dest

    def mixture_from_str(self, mix_str, dest=None, *, verify=True):
        """Read a mixture from a verbose string"""
        mixture = self._from_arr_str(mix_str, float, _parse_percent, dest)
        utils.check(
            not verify or self.is_mixture(mixture),
            '"{}" is not a valid mixture', mix_str)
        return mixture

    def _to_arr_str(self, arr, fmt):
        """Convert an array to a printable string"""
        return '\n'.join(
            '{}:\n{}'.format(role, '\n'.join(
                '    {}: {:{}}'.format(s, p, fmt)
                for p, s in zip(probs, strats)
                if p > 0))
            for probs, role, strats
            in zip(np.split(arr, self.role_starts[1:]),
                   self.role_names, self.strat_names))

    def mixture_to_str(self, mixture):
        """Convert a mixture to a printable string"""
        return self._to_arr_str(
            self.trim_mixture_precision(mixture, resolution=1e-4), '>7.2%')

    def restriction_from_json(self, jrest, dest=None, *, verify=True):
        """Read a restriction from json

        Json format is {role: [strat]}"""
        if dest is None:
            dest = np.empty(self.num_strats, bool)
        else:
            utils.check(
                dest.dtype.kind == 'b',
                'dest dtype must be boolean')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(False)

        for role, strats in jrest.items():
            for strat in strats:
                dest[self.role_strat_index(role, strat)] = True

        utils.check(
            not verify or self.is_restriction(dest),
            '"{}" does not define a valid restriction', jrest)
        return dest

    def restriction_to_json(self, rest):
        """Convert a restriction to json"""
        return {role: [strat for strat, inc in zip(strats, mask) if inc]
                for mask, role, strats
                in zip(np.split(rest, self.role_starts[1:]),
                       self.role_names, self.strat_names)
                if mask.any()}

    def restriction_from_repr(self, rrest, dest=None, *, verify=True):
        """Read a restriction from a repr string

        A restriction repr is "role: strat, ...; ..."
        """
        if dest is None:
            dest = np.empty(self.num_strats, bool)
        else:
            utils.check(
                dest.dtype.kind == 'b',
                'dest dtype must be boolean')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(False)
        for role_str in rrest.split(';'):
            role, strats = (s.strip() for s in role_str.split(':', 1))
            for strat in strats.split(','):
                dest[self.role_strat_index(role, strat.strip())] = True
        utils.check(
            not verify or self.is_restriction(dest),
            '"{}" does not define a valid restriction', rrest)
        return dest

    def restriction_to_repr(self, rest):
        """Convert a restriction to a repr string"""
        return '; '.join(
            '{}: {}'.format(role, ', '.join(
                strat for strat, inc
                in zip(strats, mask) if inc > 0))
            for role, strats, mask
            in zip(self.role_names, self.strat_names,
                   np.split(rest, self.role_starts[1:])))

    def restriction_from_str(self, srest, dest=None, *, verify=True):
        """Read a restriction from a string"""
        if dest is None:
            dest = np.empty(self.num_strats, bool)
        else:
            utils.check(
                dest.dtype.kind == 'b',
                'dest dtype must be boolean')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(False)

        role = None
        for line in srest.split('\n'):
            if line[0] != ' ':
                role = line[:-1]
            else:
                dest[self.role_strat_index(role, line[4:])] = True
        utils.check(
            not verify or self.is_restriction(dest),
            '"{}" does not define a valid restriction', srest)
        return dest

    def restriction_to_str(self, rest):
        """Convert a restriction to a string"""
        return '\n'.join(
            '{}:\n{}'.format(role, '\n'.join(
                '    {}'.format(s)
                for m, s in zip(mask, strats)
                if m))
            for mask, role, strats
            in zip(np.split(np.asarray(rest), self.role_starts[1:]),
                   self.role_names, self.strat_names))

    def role_from_json(self, role_json, dest=None, dtype=float):
        """Read role array from json"""
        if dest is None:
            dest = np.empty(self.num_roles, dtype)
        else:
            utils.check(
                dest.dtype.kind == np.dtype(dtype).kind,
                'dest dtype must match dtype')
            utils.check(
                dest.shape == (self.num_roles,),
                'dest shape must be num strats')
        for role, val in role_json.items():
            dest[self.role_index(role)] = val
        return dest

    def role_to_json(self, role_info):
        """Format role data as json"""
        return {role: info.item() for role, info
                in zip(self.role_names, np.asarray(role_info))
                if info != 0}

    def role_to_repr(self, role_info):
        """Format role data as repr"""
        return '; '.join(
            '{}: {}'.format(role, val) for role, val
            in zip(self.role_names, role_info))

    def role_from_repr(self, rrole, dest=None, dtype=float):
        """Read role data from repr

        A role repr is `role:info` delimited by commas or semicolons.
        """
        if dest is None:
            dest = np.empty(self.num_roles, dtype)
        else:
            utils.check(
                dest.dtype.kind == np.dtype(dtype).kind,
                'dest dtype must match dtype')
            utils.check(
                dest.shape == (self.num_roles,),
                'dest shape must be num strats')
        for rinfo in rrole.replace(',', ';').split(';'):
            role, val = (s.strip() for s in rinfo.split(':', 1))
            dest[self.role_index(role)] = val
        return dest

    @utils.memoize
    def __hash__(self):
        return hash((type(self), self.num_roles,
                     self.num_role_strats.tobytes(),
                     self.role_names,
                     self.strat_names))

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.num_roles == other.num_roles and
                np.all(self.num_role_strats == other.num_role_strats) and
                self.role_names == other.role_names and
                self.strat_names == other.strat_names)


class _GameLike(_StratArray): # pylint: disable=too-many-public-methods
    """Role-symmetric game representation

    This object only contains methods and information about definition of the
    game, without defining how payoff data is generated / accessed.

    Parameters
    ----------
    role_names : (str,)
        The name of each role.
    strat_names : ((str,),)
        The name of each strategy for each role.
    num_role_players :  ndarray
        The number of players in each role. Must contain non-negative integers.
    """

    def __init__(self, role_names, strat_names, num_role_players):
        super().__init__(role_names, strat_names)
        self.num_role_players = num_role_players
        self.num_role_players.setflags(write=False)
        self.num_players = num_role_players.sum()

    @property
    @utils.memoize
    def num_all_role_profiles(self):
        """The number of profiles in each role (independent of others)"""
        return utils.game_size(self.num_role_players, self.num_role_strats)

    @property
    @utils.memoize
    def num_all_profiles(self):
        """The total number of profiles in the game

        Not just the ones with data."""
        return self.num_all_role_profiles.astype(object).prod()

    @property
    @utils.memoize
    def num_all_payoffs(self):
        """The number of payoffs in all profiles"""
        dev_players = self.num_role_players - np.eye(self.num_roles, dtype=int)
        return np.sum(utils.game_size(dev_players, self.num_role_strats)
                      .astype(object).prod(1) * self.num_role_strats)

    @property
    @utils.memoize
    def num_all_dpr_profiles(self):
        """The number of unique dpr profiles

        This calculation takes time exponential in the number of roles.
        """
        # Get all combinations of "pure" roles and then filter by ones with
        # support at least 2. Thus, 0, 1, and 2 can be safely ignored
        pure = (np.arange(3, 1 << self.num_roles)[:, None] &
                (1 << np.arange(self.num_roles))).astype(bool)
        cards = pure.sum(1)
        pure = pure[cards > 1]
        cards = cards[cards > 1] - 1
        # For each combination of pure roles, compute the number of profiles
        # conditioned on those roles being pure, then multiply them by the
        # cardinality of the pure roles.
        pure_counts = np.prod(self.num_role_strats * pure + ~pure, 1)
        unpure_counts = np.prod((utils.game_size(self.num_role_players,
                                                 self.num_role_strats) -
                                 self.num_role_strats) * ~pure + pure, 1)
        overcount = np.sum(cards * pure_counts * unpure_counts)
        return self.num_all_payoffs - overcount

    @property
    @utils.memoize
    def _prof_id_base(self):
        """Base for profile ids"""
        # Base is reversed so that profile_ids are ascending
        rprofs = self.num_all_role_profiles
        if self.num_all_profiles > np.iinfo(int).max:
            rprofs = rprofs.astype(object)
        return np.insert(rprofs[:0:-1].cumprod()[::-1], self.num_roles - 1, 1)

    @property
    @utils.memoize
    def _id_play(self):
        """Ids for players"""
        rev = -np.ones(self.num_strats, int)
        rev[self.role_starts] += self.num_role_strats
        return rev.cumsum()

    def profile_to_id(self, profiles):
        """Return a unique integer representing a profile"""
        profiles = -np.asarray(profiles, int)
        profiles[..., self.role_starts] += self.num_role_players
        profiles = profiles.cumsum(-1)
        sizes = utils.game_size(self._id_play, profiles)
        if self.num_all_profiles > np.iinfo(int).max:
            sizes = sizes.astype(object)

        return np.add.reduceat(sizes, self.role_starts, -1).dot(
            self._prof_id_base)

    def profile_from_id(self, ids):
        """Return a profile from its integer representation"""
        ids = np.asarray(ids)
        role_ids = (ids[..., None] // self._prof_id_base %
                    self.num_all_role_profiles)
        dec_profs = np.zeros(ids.shape + (self.num_strats,), int)

        role_ids_iter = role_ids.view()
        role_ids_iter.shape = (-1, self.num_roles)
        dec_profs_iter = dec_profs.view()
        dec_profs_iter.shape = (-1, self.num_strats)

        # This can't be vectorized further, because the sizes are dependent
        for sizes, profs in zip(
                role_ids_iter.T,
                np.split(dec_profs_iter.T, self.role_starts[1:])):
            for prof, rem in zip(profs[:-1], range(profs.shape[0] - 1, 0, -1)):
                np.copyto(prof, utils.game_size_inv(sizes, rem))
                sizes -= utils.game_size(rem, prof)
        profiles = np.delete(np.diff(
            np.insert(dec_profs, self.role_starts, 0, -1),
            1, -1), self.role_starts[1:] + np.arange(self.num_roles - 1), -1)
        profiles[..., self.role_starts] -= self.num_role_players
        return -profiles

    def is_profile(self, prof, *, axis=-1):
        """Verify that a profile is valid for game"""
        prof = np.asarray(prof, int)
        utils.check(
            prof.shape[axis] == self.num_strats,
            'profiles must have valid shape')
        play_shape = [1] * prof.ndim
        play_shape[axis] = self.num_roles
        return (
            np.all(self.num_role_players.reshape(play_shape) ==
                   np.add.reduceat(prof, self.role_starts, axis), axis) &
            np.all(prof >= 0, axis))

    def is_pure_profile(self, prof, *, axis=-1):
        """Verify a profile is pure"""
        prof = np.asarray(prof, int)
        return (self.is_profile(prof, axis=axis) &
                self._is_pure_restriction(prof > 0, axis=axis))

    def all_profiles(self):
        """Return all profiles"""
        role_arrays = [utils.acomb(n_strats, players, True)
                       for n_strats, players
                       in zip(self.num_role_strats, self.num_role_players)]
        return utils.acartesian2(*role_arrays)

    def pure_profiles(self):
        """Return all pure profiles

        A pure profile is a profile where only one strategy is played per
        role."""
        return (self.pure_restrictions() *
                self.num_role_players.repeat(self.num_role_strats))

    def nearby_profiles(self, profile, num_devs):
        """Returns profiles reachable by at most num_devs deviations"""
        utils.check(num_devs >= 0, 'num devs must be nonnegative')
        profile = np.asarray(profile, int)
        dev_players = utils.acomb(self.num_roles, num_devs, True)
        mask = np.all(dev_players <= self.num_role_players, 1)
        dev_players = dev_players[mask]
        supp = profile > 0
        sub_strats = np.add.reduceat(supp, self.role_starts)

        profiles = [profile[None]]
        for players in dev_players:
            to_dev_profs = empty(
                players, self.num_role_strats).all_profiles()
            sub = empty(players, sub_strats)
            from_dev_profs = np.zeros((sub.num_all_profiles,
                                       self.num_strats), int)
            from_dev_profs[:, supp] = sub.all_profiles()
            before_devs = profile - from_dev_profs
            before_devs = before_devs[np.all(before_devs >= 0, 1)]
            before_devs = utils.axis_from_elem(np.unique(utils.axis_to_elem(
                before_devs)))
            nearby = before_devs[:, None] + to_dev_profs
            nearby.shape = (-1, self.num_strats)
            profiles.append(utils.axis_from_elem(np.unique(utils.axis_to_elem(
                nearby))))
        return utils.axis_from_elem(np.unique(utils.axis_to_elem(
            np.concatenate(profiles))))

    def random_profile(self, mixture=None):
        """Return a random profile"""
        return self.random_profiles(1, mixture)[0]

    def random_profiles(self, num_samples, mixture=None):
        """Sample profiles from a mixture

        Parameters
        ----------
        num_samples : int
            Number of samples to return.
        mixture : ndarray, optional
            Mixture to sample from, of None or omitted, then uses the uniform
            mixture.
        """
        if mixture is None:
            mixture = self.uniform_mixture()
        role_samples = [rand.multinomial(n, probs, num_samples) for n, probs
                        in zip(self.num_role_players,
                               np.split(mixture, self.role_starts[1:]))]
        return np.concatenate(role_samples, 1)

    def round_mixture_to_profile(self, mixture):
        """Round a mixture to the nearest profile

        This finds the profile with the minimum absolute error to the product
        of the profile and the number of players per role.
        """
        float_prof = mixture * self.num_role_players.repeat(
            self.num_role_strats)
        profile = np.floor(float_prof).astype(int)
        missing = self.num_role_players - np.add.reduceat(
            profile, self.role_starts, -1)
        errors = profile - float_prof + np.arange(self.num_roles).repeat(
            self.num_role_strats)
        rank = errors.argsort(-1) - self.role_starts.repeat(
            self.num_role_strats)
        profile[rank < missing.repeat(self.num_role_strats, -1)] += 1
        return profile

    def random_role_deviation_profile(self, mixture=None):
        """Return a random role deviation profile"""
        return self.random_role_deviation_profiles(1, mixture)[0]

    def random_role_deviation_profiles(self, num_samples, mixture=None):
        """Return partial profiles where dev player is missing

        Resulting shape of profiles is (num_samples, num_roles,
        num_role_strats). The first dimension is the sample, the next is the
        deviating role, leaving the last dimension for the partial profile.

        Parameters
        ----------
        mixture : ndarray
            Mixture to sample from.
        num_samples : int, optional
            Number of samples to return. If None or omitted, then a single
            sample, without a leading singleton dimension is returned.
        """
        if mixture is None:
            mixture = self.uniform_mixture()
        dev_players = self.num_role_players - np.eye(self.num_roles, dtype=int)
        profs = np.empty((num_samples, self.num_roles, self.num_strats),
                         int)
        for i, players in enumerate(dev_players):
            base = empty(players, self.num_role_strats)
            profs[:, i] = base.random_profiles(num_samples, mixture)
        return profs

    def random_deviation_profile(self, mixture=None):
        """Return a random deviation profile"""
        return self.random_deviation_profiles(1, mixture)[0]

    def random_deviation_profiles(self, num_samples, mixture=None):
        """Return a profiles where one player is deviating from mix

        Resulting shape of profiles is (num_samples, num_role_strats,
        num_role_strats). The first dimension is the sample, the next is the
        deviating strategy, leaving the last dimension for the actual
        profile.

        Parameters
        ----------
        mixture : ndarray
            Mixture to sample from.
        num_samples : int, optional
            Number of samples to return. If None or omitted, then a single
            sample, without a leading singleton dimension is returned.
        """
        devs = self.random_role_deviation_profiles(num_samples, mixture)
        return (devs.repeat(self.num_role_strats, -2) +
                np.eye(self.num_strats, dtype=int))

    def max_prob_prof(self, mixture):
        """Returns the pure strategy profile with highest probability."""
        mixture = np.asarray(mixture, float)
        return np.concatenate(
            [utils.multinomial_mode(m, p) for m, p
             in zip(np.split(mixture, self.role_starts[1:]),
                    self.num_role_players)], -1)

    @utils.memoize
    def is_symmetric(self):
        """Returns true if this game is symmetric"""
        return self.num_roles == 1

    @utils.memoize
    def is_asymmetric(self):
        """Returns true if this game is asymmetric"""
        return np.all(self.num_role_players == 1)

    def profile_from_json(self, prof, dest=None, *, verify=True):
        """Read a profile from json

        Parameters
        ----------
        prof : {role: {strat: count}}
            A description of a profile.
        dest : ndarray, optional
            If supplied, ``dest`` will be written to instead of allocating a
            new array.
        verify : bool, optional
            If true, verify that a proepr profile for this game was read.
        """
        if dest is None:
            dest = np.empty(self.num_strats, int)
        else:
            utils.check(
                dest.dtype.kind == 'i',
                'dest dtype must be integral')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(0)

        for role, strats in prof.items():
            for strat, count in strats.items():
                dest[self.role_strat_index(role, strat)] = count

        utils.check(
            not verify or self.is_profile(dest),
            '"{}" is not a valid profile', prof)
        return dest

    def payoff_from_json(self, pays, dest=None):
        """Read a payoff from json

        Parameters
        ----------
        pays : {role: {strat: payoff}}
            A description of a payoff.
        dest : ndarray, optional
            If supplied, ``dest`` will be written to instead of allocating a
            new array.
        """
        if dest is None:
            dest = np.empty(self.num_strats, float)
        else:
            utils.check(
                dest.dtype.kind == 'f',
                'dest dtype must be floating')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(0)

        for role, strats in pays.items():
            for strat, pay in strats.items():
                dest[self.role_strat_index(role, strat)] = _mean(pay)

        return dest

    def profile_to_json(self, prof):
        """Convert a profile array to json"""
        return self._to_arr_json(prof)

    def payoff_to_json(self, payoffs):
        """Format payoffs as json

        Parameters
        ----------
        payoffs : ndarray
            The payoffs to serialize.
        """
        return self._to_arr_json(payoffs)

    def profile_from_repr(self, prof_str, dest=None, *, verify=True):
        """Read a profile from a string"""
        prof = self._from_arr_repr(prof_str, int, int, dest)
        utils.check(
            not verify or self.is_profile(prof),
            '"{}" does not define a profile', prof_str)
        return prof

    def profile_to_repr(self, prof):
        """Convert a profile to a string"""
        return self._to_arr_repr(prof, 'd')

    def profile_from_str(self, prof_str, dest=None, *, verify=True):
        """Get profile from string"""
        prof = self._from_arr_str(prof_str, int, int, dest)
        utils.check(
            not verify or self.is_profile(prof),
            '"{}" is not a valid profile', prof_str)
        return prof

    def profile_to_str(self, prof):
        """Convert a profile to a printable string"""
        return self._to_arr_str(prof, 'd')

    def devpay_from_json(self, deviations, dest=None):
        """Get deviation payoffs from json"""
        if dest is None:
            dest = np.empty(self.num_devs)
        else:
            utils.check(
                dest.dtype.kind == 'f',
                'dest dtype must be floating')
            utils.check(
                dest.shape == (self.num_devs,),
                'dest shape must be num strats')
        dest.fill(0)

        for role, strats in deviations.items():
            for strat, devs in strats.items():
                for dev, val in devs.items():
                    dest[self.role_strat_dev_index(role, strat, dev)] = val

        return dest

    def devpay_to_json(self, payoffs):
        """Format a profile and deviation payoffs as json"""
        payoffs = np.asarray(payoffs, float)
        return {r: {s: {d: pay.item() for pay, d  # pragma: no branch
                        in zip(spays, (d for d in ses if d != s))
                        if pay != 0}
                    for spays, s
                    in zip(np.split(rpay, n), ses)
                    if np.any(spays != 0)}
                for r, ses, n, rpay
                in zip(self.role_names, self.strat_names,
                       self.num_role_strats,
                       np.split(payoffs, self.dev_role_starts[1:]))
                if np.any(rpay != 0)}

    def to_json(self):
        """Format game as json"""
        return {
            'players': dict(zip(self.role_names,
                                map(int, self.num_role_players))),
            'strategies': dict(zip(self.role_names,
                                   map(list, self.strat_names)))
        }

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__[1:],
            self.num_role_players,
            self.num_role_strats)

    def __str__(self):
        """Format game as a printable string"""
        return (
            ('{}:\n    Roles: {}\n    Players:\n        {}\n    '
             'Strategies:\n        {}').format(
                 self.__class__.__name__[1:],
                 ', '.join(self.role_names),
                 '\n        '.join(
                     '{:d}x {}'.format(count, role)
                     for role, count
                     in zip(self.role_names, self.num_role_players)),
                 '\n        '.join(
                     '{}:\n            {}'.format(
                         role, '\n            '.join(strats))
                     for role, strats
                     in zip(self.role_names, self.strat_names))))

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_role_players.tobytes()))

    def __eq__(self, other):
        return (super().__eq__(other) and
                np.all(self.num_role_players == other.num_role_players))


class _RsGame(_GameLike):
    """Role-symmetric game representation

    This object only contains methods and information about definition of the
    game, without defining how payoff data is generated / accessed.

    Parameters
    ----------
    role_names : (str,)
        The name of each role.
    strat_names : ((str,),)
        The name of each strategy for each role.
    num_role_players :  ndarray
        The number of players in each role. Must contain non-negative integers.
    """

    def __init__(self, role_names, strat_names, num_role_players):
        super().__init__(role_names, strat_names, num_role_players)
        self.zero_prob = np.finfo(float).tiny * (self.num_role_players + 1)
        self.zero_prob.setflags(write=False)

    # ----------------
    # Abstract Methods
    # ----------------

    @property
    @abc.abstractmethod
    def num_profiles(self):
        """The number of profiles with any payoff information"""
        pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def num_complete_profiles(self):
        """The number of profiles with complete payoff information"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def profiles(self):
        """An array all of the profiles with any data"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def payoffs(self):
        """An array with all of the payoff corresponding to profiles()"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def max_strat_payoffs(self):
        """An upper bound on the payoff for each strategy"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def min_strat_payoffs(self):
        """A lower bound on the payoff for each strategy"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def get_payoffs(self, profiles):
        """The payoffs for all profiles"""
        pass  # pragma: no cover

    # Note: This should allow arbitrary keyword arguments which is ignores if
    # they're invalid.
    @abc.abstractmethod
    def deviation_payoffs(self, mixture, *, jacobian=False, **_):
        """The payoffs for deviating from mixture

        Optionally with the jacobian with respect to mixture. This is the
        primary method that needs to implemented for nash finding."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def restrict(self, restriction):
        """Restrict viable strategies"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def __contains__(self, profile):
        """Return true if full payoff data for profile exists"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _add_constant(self, constant):
        """Add a constant to each roles payoffs"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _multiply_constant(self, constant):
        """Multiply each roles payoffs by a positive constant"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def _add_game(self, othr):
        """Add two games together, so payoffs are the sum

        othr is guaranteed to be a game like object with the same structure as
        self. If no add can be done efficiently, then this should return
        NotImplemented."""
        pass  # pragma: no cover

    # --------------------
    # End Abstract Methods
    # --------------------

    def min_role_payoffs(self):
        """Returns the minimum payoff for each role"""
        return np.fmin.reduceat(self.min_strat_payoffs(), self.role_starts)

    def max_role_payoffs(self):
        """Returns the maximum payoff for each role"""
        return np.fmax.reduceat(self.max_strat_payoffs(), self.role_starts)

    def normalize(self):
        """Return a new game where the max payoff is 1 and min payoff is 0"""
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0) | np.isnan(scale)] = 1
        offset = self.min_role_payoffs()
        offset[np.isnan(offset)] = 0
        return (self - offset) / scale

    def get_dev_payoffs(self, dev_profs):
        """Compute the payoffs for deviating

        Given partial profiles per role, compute the mean
        payoff for deviating to each strategy.

        Parameters
        ----------
        dev_profs : array-like, shape = (num_samples, num_roles, num_strats)
            A list of partial profiles by role. This is the same structure as
            returned by `random_dev_profiles`.
        """
        return np.diagonal(self.get_payoffs(
            np.repeat(dev_profs, self.num_role_strats, -2) +
            np.eye(self.num_strats, dtype=int)), 0, -2, -1)

    def is_empty(self):
        """Returns true if no profiles have data"""
        return self.num_profiles == 0

    def is_complete(self):
        """Returns true if every profile has data"""
        return self.num_complete_profiles == self.num_all_profiles

    def is_constant_sum(self):
        """Returns true if this game is constant sum"""
        if self.is_empty():
            return True

        profile_sums = np.einsum(
            'ij,ij->i', self.profiles(), self.payoffs())
        return np.allclose(profile_sums, profile_sums[0])

    def expected_payoffs(self, mixture):
        """Returns the payoff of each role under mixture"""
        mixture = np.asarray(mixture)
        deviations = self.deviation_payoffs(mixture)
        return np.add.reduceat(mixture * np.where(
            mixture > 0, deviations, 0), self.role_starts)

    def best_response(self, mixture):
        """Returns the best response to a mixture

        The result is a new mixture with uniform support over all best
        deviating strategies.
        """
        responses = self.deviation_payoffs(mixture)
        bests = np.maximum.reduceat(responses, self.role_starts)
        best_resps = responses == bests.repeat(self.num_role_strats)
        with np.errstate(invalid='ignore'):  # nan
            return best_resps / np.add.reduceat(
                best_resps, self.role_starts).repeat(self.num_role_strats)

    def __mul__(self, constant):
        try:
            constant = np.asarray(constant, float)
            assert np.all(constant > 0)
            return self._multiply_constant(constant)
        except (TypeError, AssertionError, ValueError):
            return NotImplemented

    def __rmul__(self, constant):
        return self.__mul__(constant)

    def __truediv__(self, constant):
        try:
            return self.__mul__(1 / np.asarray(constant, float))
        except (TypeError, ValueError):
            return NotImplemented

    def __add__(self, othr):
        with contextlib.suppress(TypeError, ValueError):
            return self._add_constant(np.asarray(othr, float))
        try:
            assert empty_copy(self) == empty_copy(othr)
            attempt = self._add_game(othr)
            if attempt is NotImplemented and type(self) is type(othr): # pylint: disable=no-else-return
                return add(self, othr)
            else:
                return attempt
        except (AttributeError, AssertionError):
            return NotImplemented

    def __radd__(self, othr):
        with contextlib.suppress(TypeError, ValueError):
            return self._add_constant(np.asarray(othr, float))
        try:
            assert empty_copy(self) == empty_copy(othr)
            attempt = self._add_game(othr)
            return add(self, othr) if attempt is NotImplemented else attempt
        except (AttributeError, AssertionError):
            return NotImplemented

    def __sub__(self, othr):
        try:
            return self._add_constant(-np.asarray(othr, float))
        except (TypeError, ValueError):
            return NotImplemented


class _EmptyGame(_RsGame):
    """A game with no payoff data"""

    def __init__(self, role_names, strat_names, num_role_players):
        super().__init__(role_names, strat_names, num_role_players)
        self._num_profiles = self._num_complete_profiles = 0

    @property
    def num_profiles(self):
        return self._num_profiles

    @property
    def num_complete_profiles(self):
        return self._num_complete_profiles

    def profiles(self):
        return np.empty((0, self.num_strats), int)

    def payoffs(self):
        return np.empty((0, self.num_strats), float)

    @utils.memoize
    def max_strat_payoffs(self):
        maxs = np.full(self.num_strats, np.nan)
        maxs.setflags(write=False)
        return maxs.view()

    @utils.memoize
    def min_strat_payoffs(self):
        mins = np.full(self.num_strats, np.nan)
        mins.setflags(write=False)
        return mins.view()

    def get_payoffs(self, profiles):
        utils.check(
            self.is_profile(profiles).all(),
            'profiles must be valid')
        pays = np.empty(profiles.shape)
        pays.fill(np.nan)
        pays[profiles == 0] = 0
        pays.setflags(write=False)
        return pays.view()

    def deviation_payoffs(self, mixture, *, jacobian=False, **_):
        utils.check(
            self.is_mixture(mixture),
            'mixtures must be valid')
        devs = np.full(self.num_strats, np.nan)
        if not jacobian:
            return devs

        jac = np.full((self.num_strats,) * 2, np.nan)
        return devs, jac

    def restrict(self, restriction):
        utils.check(
            self.is_restriction(restriction),
            'restrictions must be valid')
        new_strats = tuple(
            tuple(s for s, m in zip(strats, mask) if m)
            for strats, mask in zip(
                self.strat_names, np.split(restriction, self.role_starts[1:])))
        return _EmptyGame(self.role_names, new_strats, self.num_role_players)

    def _add_constant(self, _):
        return self

    def _multiply_constant(self, _):
        return self

    def _add_game(self, _):
        return self

    def __contains__(self, profile):
        utils.check(
            self.is_profile(profile),
            'profile must be valid')
        return False

    def to_json(self):
        res = super().to_json()
        res['type'] = 'empty.1'
        return res


def empty(num_role_players, num_role_strats):
    """Create an empty game with default names

    Parameters
    ----------
    num_role_players : ndarray-like, int
        The number of players in each role in order, or the number of players
        per role if identical (will be broadcast to match the number of roles).
    num_role_strats : ndarray-like, int
        The number of strategies in each role in order, or the number of
        strategies per role if identical (will be broadcast to match the number
        of roles).
    """
    num_role_players = np.asarray(num_role_players, int)
    num_role_strats = np.asarray(num_role_strats, int)
    utils.check(
        np.all(num_role_players >= 0),
        "can't have a negative number of players")
    utils.check(
        np.all(num_role_strats > 0),
        'must have at least one strategy per role')
    num_roles = max(num_role_players.size, num_role_strats.size)
    num_role_players = np.broadcast_to(num_role_players, num_roles)
    num_role_strats = np.broadcast_to(num_role_strats, num_roles)
    role_names = tuple(utils.prefix_strings('r', num_roles))
    strats = utils.prefix_strings('s', num_role_strats.sum())
    strat_names = tuple(tuple(itertools.islice(strats, int(n)))
                        for n in num_role_strats)
    return _EmptyGame(role_names, strat_names, num_role_players)


def empty_names(role_names, num_role_players, strat_names):
    """Create an empty game with names

    PArameters
    ----------
    roles_names : [str]
        The name for each role.
    num_role_players : ndarray, int, [int]
        The number of players in each role.
    strat_names : [[str]]
        The name of each strategy for each role.
    """
    utils.check(
        len(role_names) == len(strat_names),
        'number of roles must be consistent')
    for role in role_names:
        utils.check(
            isinstance(role, str), 'role {} is not a string', role)
    for strats in strat_names:
        for strat in strats:
            utils.check(
                isinstance(strat, str),
                'strategy {} is not a string', strat)
    utils.check(
        utils.is_sorted(role_names, strict=True),
        'role names must be sorted')
    for i, strats in enumerate(strat_names):
        utils.check(
            utils.is_sorted(strats, strict=True),
            'strategies in role {:d} must be sorted', i)
    for i, strats in enumerate(strat_names):
        utils.check(
            strats, 'role {:d} must have at least one strategy', i)
    num_role_players = np.broadcast_to(np.asarray(num_role_players, int),
                                       len(role_names))
    # This test for equality because we get games with zero players when
    # deviating, in the same way that 1 strategy is technically degenerate
    utils.check(
        np.all(num_role_players >= 0),
        'number of players must be non-negative')
    for role in role_names:
        utils.check(
            _LEG_ROLE.issuperset(role),
            'role {} contains illegal characters', role)
    for strats in strat_names:
        for strat in strats:
            utils.check(
                _LEG_STRAT.issuperset(strat),
                'strategy {} contains illegal characters', strat)
    return _EmptyGame(
        tuple(role_names), tuple(map(tuple, strat_names)), num_role_players)


def empty_json(json):
    """Read a EmptyGame from json

    Parameters
    ----------
    json : {...}
        A json representation of a basic game with names. Must either be
        {roles: [{name: <role>, strategies: [<strat>]}]}, or {strategies:
        {<role>: [<strat>]}}.
    """
    if 'roles' in json:
        desc = [(j['name'], j['count'], sorted(j['strategies']))
                for j in json['roles']]
    elif {'strategies', 'players'}.issubset(json):
        players = json['players']
        desc = [(r, players[r], sorted(s)) for r, s
                in json['strategies'].items()]
    else:
        raise ValueError('"{}" does not describe a game'.format(json))
    desc.sort()
    role_names = tuple(r for r, _, _ in desc)
    strat_names = tuple(tuple(sorted(s)) for _, _, s in desc)
    num_role_players = np.fromiter((c for _, c, _ in desc), int, len(desc))  # pragma: no branch pylint: disable=line-too-long
    utils.check(
        all(isinstance(r, str) for r in role_names),
        'role names must be strings')
    utils.check(
        all(all(isinstance(s, str) for s in strats) for strats in strat_names),
        'strategy names must be strings')
    utils.check(
        all(len(s) > 0 for s in strat_names),
        'must have at least one strategy per role')
    utils.check(
        np.all(num_role_players >= 0),
        'number of players must be non-negative')
    utils.check(
        all(_LEG_ROLE.issuperset(r) for r in role_names),
        'role names must be valid')
    utils.check(
        all(all(_LEG_STRAT.issuperset(s) for s in strats)
            for strats in strat_names),
        'strat names must be valid')
    return _EmptyGame(role_names, strat_names, num_role_players)


def empty_copy(copy_game):
    """Copy parameters of a game into an empty game

    This method is useful to keep convenience methods of game without attached
    data.

    Parameters
    ----------
    copy_game : RsGame
        Game to copy info from.
    """
    return _EmptyGame(
        copy_game.role_names, copy_game.strat_names,
        copy_game.num_role_players)


class _AddGame(_RsGame):
    """A Game representing the addition of two games

    Payoffs in this game are the sum of the payoffs from each game. Some game
    types may support native addition, this is the fallback.
    """

    def __init__(self, games):
        super().__init__(
            games[0].role_names, games[0].strat_names,
            games[0].num_role_players)
        self._games = games

    @property
    @utils.memoize
    def num_complete_profiles(self):
        if all(game.is_complete() for game in self._games):
            return self.num_all_profiles

        return len(frozenset.intersection(*[
            frozenset(utils.hash_array(prof) for prof, pay
                      in zip(game.profiles(), game.payoffs())
                      if not np.isnan(pay).any())
            for game in self._games]))

    @property
    @utils.memoize
    def num_profiles(self):
        if all(game.is_complete() for game in self._games): # pylint: disable=no-else-return
            return self.num_all_profiles
        else:
            return self.profiles().shape[0]

    @functools.lru_cache(maxsize=1)
    def profiles(self):
        if all(game.is_complete() for game in self._games):
            return self.all_profiles()

        profs = frozenset.intersection(*[
            frozenset(utils.hash_array(prof) for prof in game.profiles())
            for game in self._games])
        if profs: # pylint: disable=no-else-return
            return np.stack([h.array for h in profs])
        else:
            return np.empty((0, self.num_strats), int)

    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        return self.get_payoffs(self.profiles())

    def deviation_payoffs(self, mixture, *, jacobian=False, **kw):
        if not jacobian:
            return sum(game.deviation_payoffs(mix, **kw)
                       for game in self._games)
        return map(sum, zip(*[
            game.deviation_payoffs(mixture, jacobian=True, **kw)
            for game in self._games]))

    def get_payoffs(self, profiles):
        return sum(game.get_payoffs(profiles) for game in self._games)

    @utils.memoize
    def max_strat_payoffs(self):
        return sum(game.max_strat_payoffs() for game in self._games)

    @utils.memoize
    def min_strat_payoffs(self):
        return sum(game.min_strat_payoffs() for game in self._games)

    def restrict(self, restriction):
        return _AddGame(
            tuple(game.restrict(restriction) for game in self._games))

    def _add_constant(self, constant):
        avg_const = constant / len(self._games)
        return _AddGame(
            tuple(game + avg_const for game in self._games))

    def _multiply_constant(self, constant):
        return _AddGame(
            tuple(game * constant for game in self._games))

    def _add_game(self, othr):
        return add(self, othr)

    def to_json(self):
        base = super().to_json()
        base['games'] = [game.to_json() for game in self._games]
        base['type'] = 'add.1'
        return base

    def __contains__(self, profile):
        return all(profile in game for game in self._games)

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return (super().__eq__(othr) and
                frozenset(self._games) == frozenset(othr._games))

    def __hash__(self):
        return hash(frozenset(self._games))

    def __repr__(self):
        return '{}, {:d} / {:d})'.format(
            super().__repr__()[:-1], self.num_profiles, self.num_all_profiles)


def add(*games):
    """Add games together to that the payoff is the sum of each game

    Parameters
    ----------
    games : RsGame
        The games to add together
    """
    utils.check(games, 'must add at least one game')
    base = empty_copy(games[0])
    utils.check(
        all(base == empty_copy(game) for game in games[1:]),
        'all games must have same structure')

    def get_games(game):
        """Get the games if it's an add game"""
        try:
            return game._games # pylint: disable=protected-access
        except AttributeError:
            return [game]

    # Expand games in base forms
    games = list(itertools.chain.from_iterable(map(get_games, games)))
    # This attempts to add any game that can be added
    final_games = []
    while games:
        current_game = games.pop()
        unmerged_games = []
        for game in games:
            # pylint: disable-msg=protected-access
            attempt = current_game._add_game(game)
            if attempt is not NotImplemented:
                current_game = attempt
                continue
            attempt = game._add_game(current_game)
            if attempt is not NotImplemented:
                current_game = attempt
                continue
            unmerged_games.append(game)
        final_games.append(current_game)
        games = unmerged_games
    if len(final_games) == 1: # pylint: disable=no-else-return
        return final_games[0]
    else:
        return _AddGame(tuple(final_games))


def add_json(jgame):
    """Read added games from json"""
    base = empty_json(jgame)
    games = [gamereader.loadj(jg) for jg in jgame['games']]
    utils.check(
        all(base == empty_copy(game) for game in games),
        "game structure didn't match each added game")
    return add(*games)


def mix(game0, game1, prob):
    """Mix games together

    The resulting payoff is a (1-prob) fraction of game0 and a prob fraction of
    game1.

    Parameters
    ----------
    game0 : RsGame
        The first game to mix.
    game1 : RsGame
        The second game to mix.
    prob : float
        The fraction to merge the games. 0 corresponds to a copy of `game0`, 1
        corresponds to `game1`, and somewhere between corresponds to the linear
        interpolation between them.
    """
    utils.check(
        0 <= prob <= 1, 't must be in [0, 1] but was {:g}', prob)
    if prob == 0: # pylint: disable=no-else-return
        return game0
    elif prob == 1:
        return game1
    else:
        return (1 - prob) * game0 + prob * game1


class _CompleteGame(_RsGame): # pylint: disable=abstract-method
    """A game that defines everything for complete games

    Extend this if your game by default has payoff data for every profile."""

    @property
    def num_profiles(self):
        return self.num_all_profiles

    @property
    def num_complete_profiles(self):
        return self.num_all_profiles

    @functools.lru_cache(maxsize=1)
    def profiles(self):
        return self.all_profiles()

    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        return self.get_payoffs(self.profiles())

    def __contains__(self, profile):
        utils.check(self.is_profile(profile), 'profile must be valid')
        return True


class _ConstantGame(_CompleteGame):
    """A game with constant payoffs"""
    def __init__(self, role_names, strat_names, num_role_players, constant):
        super().__init__(role_names, strat_names, num_role_players)
        self._role_const = np.broadcast_to(constant, self.num_roles)
        self._role_const.setflags(write=False)
        self._strat_const = self._role_const.repeat(self.num_role_strats)
        self._strat_const.setflags(write=False)

    def deviation_payoffs(self, _, *, jacobian=False, **_kw):
        if jacobian: # pylint: disable=no-else-return
            return self._strat_const.copy(), np.zeros([self.num_strats] * 2)
        else:
            return self._strat_const.copy()

    def get_payoffs(self, profiles):
        return np.where(profiles > 0, self._strat_const, 0)

    def max_strat_payoffs(self):
        return self._strat_const.view()

    def min_strat_payoffs(self):
        return self._strat_const.view()

    def restrict(self, restriction):
        base = empty_copy(self).restrict(restriction)
        return _ConstantGame(
            base.role_names, base.strat_names, base.num_role_players,
            self._role_const)

    def _add_constant(self, constant):
        return _ConstantGame(
            self.role_names, self.strat_names, self.num_role_players,
            self._role_const + constant)

    def _multiply_constant(self, constant):
        return _ConstantGame(
            self.role_names, self.strat_names, self.num_role_players,
            self._role_const * constant)

    def _add_game(self, othr):
        return othr + self._role_const

    def to_json(self):
        base = super().to_json()
        base['const'] = self._role_const.tolist()
        base['type'] = 'const.1'
        return base

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return (super().__eq__(othr) and
                np.allclose(self._role_const, othr._role_const))

    @utils.memoize
    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return '{}, {})'.format(super().__repr__()[:-1], self._role_const)


def const(num_role_players, num_role_strats, constant):
    """Create a new constant game"""
    return const_replace(
        empty(num_role_players, num_role_strats), constant)


def const_names(role_names, num_role_players, strat_names, constant):
    """Create a new constant game with names"""
    return const_replace(
        empty_names(role_names, num_role_players, strat_names), constant)


def const_replace(copy_game, constant):
    """Replace a game with constant payoffs"""
    return _ConstantGame(
        copy_game.role_names, copy_game.strat_names,
        copy_game.num_role_players, np.asarray(constant, float))


def const_json(jgame):
    """Read a constant game from json"""
    base = empty_json(jgame)
    constant = np.asarray(jgame['const'], float)
    return const_replace(base, constant)


# Legal characters for roles and strategies
_LEG_ROLE = frozenset(set(string.printable) - set(';:'))
_LEG_STRAT = frozenset(set(string.printable) - set(';,'))


def _parse_percent(perc):
    """Parse percentile"""
    return float(perc[:-1]) / 100


def _mean(vals):
    """Compute mean"""
    if not isinstance(vals, cabc.Iterable):
        return vals

    count = 0
    mean = 0
    for val in vals:
        count += 1
        mean += (val - mean) / count
    return mean if count > 0 else float('nan')
