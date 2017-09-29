"""Module for Role Symmetric Game data structures

There are three types of games:
    BaseGame   - Abstract class of game, with a few functions that should be
                 provided by implementing classes.
    Game       - Contains payoff data at a profile level, but that data can be
                 sparse.
    SampleGame - Contain several samples of payoffs for every profile. Access
                 to the sample data is relatively limited and intended mostly
                 for other functions that operate on the entire game at once.

Everything internally is represented as an array. Most methods will take any
dimensional array as input, treating the last axis as a profile  / payoff /
etc, and treating all other axes as multiple data points, but this isn't
universal.

Most game objects have attributes that start with num, these will always be an
attribute or a property, not a method, so to get the number of profiles, it's
just `num_profiles` not `num_profiles()`. These will also only be numbers,
either a single int, or an array of them depending on the attribute."""
import functools
import itertools
import warnings

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import utils


_TINY = np.finfo(float).tiny


class _StratArray(object):
    """A class with knowledge of the number of strategies per role

    This has methods common to working with strategy arrays, which essentially
    represent points in a simplotope (a cross product of simplicies), or points
    in a discretized simplotope.
    """

    def __init__(self, num_role_strats):
        assert num_role_strats.ndim == 1
        assert np.all(num_role_strats > 0)
        self.num_roles = num_role_strats.size
        self.num_strats = num_role_strats.sum()
        self.num_role_strats = num_role_strats
        self.role_starts = np.insert(num_role_strats[:-1].cumsum(), 0, 0)
        self.role_indices = np.arange(self.num_roles).repeat(num_role_strats)

        self.num_role_strats.setflags(write=False)
        self.role_starts.setflags(write=False)
        self.role_indices.setflags(write=False)

    @property
    @utils.memoize
    def num_all_subgames(self):
        """Number of unique subgames"""
        return np.prod(2 ** self.num_role_strats - 1)

    @property
    @utils.memoize
    def num_pure_subgames(self):
        """The number of pure subgames

        A pure subgame is is one with only one strategy per role."""
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
                "using reduceat with dev_strat_starts will not produce "
                "correct results if any role only has one strategy. This "
                "might get fixed at some point, but currently extra care must "
                "be taken for these cases.")
        starts = np.insert(self.num_strat_devs[:-1].cumsum(), 0, 0)
        starts.setflags(write=False)
        return starts

    @property
    @utils.memoize
    def dev_role_starts(self):
        """The start index for each role deviation"""
        if np.any(self.num_role_strats == 1):
            warnings.warn(
                "using reduceat with dev_role_starts will not produce "
                "correct results if any role only has one strategy. This "
                "might get fixed at some point, but currently extra care must "
                "be taken for these cases.")
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

        # XXX The use of bincount here allows for one strategy roles
        pos_offset = np.bincount(np.arange(self.num_strats) -
                                 self.role_starts.repeat(self.num_role_strats)
                                 + self.dev_strat_starts,
                                 minlength=self.num_devs + 1)[:-1]
        neg_offset = np.bincount(self.dev_strat_starts[1:],
                                 minlength=self.num_devs + 1)[:-1]
        inds += np.cumsum(pos_offset - neg_offset)

        inds.setflags(write=False)
        return inds

    def all_subgames(self):
        """Return all valid subgames"""
        role_subs = [(np.arange(1, 1 << num_strats)[:, None]
                      & (1 << np.arange(num_strats))).astype(bool)
                     for num_strats
                     in self.num_role_strats]
        return utils.acartesian2(*role_subs)

    def pure_subgames(self):
        """Returns every pure subgame mask in a game

        A pure subgame is a subgame where each role only has one strategy. This
        returns the pure subgames in sorted order based off of role and
        strategy."""
        role_subgames = [np.eye(num_strats, dtype=bool) for num_strats
                         in self.num_role_strats]
        return utils.acartesian2(*role_subgames)

    def random_subgames(self, num_samples=None, strat_prob=None,
                        normalize=True):
        """Return random subgames

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to be retuned, if None or unspecified, a
            single sample without the extra dimension is returned.
        strat_prob : float, ndarray, optional, (0, 1)
            The probability that a given strategy is in support. If support
            prob is None, supports will be sampled uniformly. This can either
            be a scalar, or an ndarray of size `num_roles`.
        normalize : bool, optional
            If true, the mixtures are normalized, so that the conditional
            probability of any strategy in support equals support prob. If
            true, the support_prob for any role must be at least `1 /
            num_role_strats`. Individual role probabilities are thresholded to
            this value.
        """
        if num_samples is None:
            return self.random_subgames(1, strat_prob, normalize)[0]
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

    def is_subgame(self, subg, *, axis=-1):
        """Verify that a subgame or array of subgames are valid"""
        subg = np.asarray(subg, bool)
        assert subg.shape[axis] == self.num_strats
        return np.all(np.bitwise_or.reduceat(subg, self.role_starts, axis),
                      axis)

    def trim_mixture_support(self, mixture, *, thresh=1e-3, axis=-1):
        """Trims strategies played less than supp_thresh from the support"""
        assert mixture.shape[axis] == self.num_strats
        mixture *= mixture >= thresh
        mixture /= np.add.reduceat(mixture, self.role_starts,
                                   axis).repeat(self.num_role_strats, axis)
        return mixture

    def is_mixture(self, mix, *, axis=-1):
        """Verify that a mixture is valid for game"""
        mix = np.asarray(mix, float)
        assert mix.shape[axis] == self.num_strats
        return (np.all(mix >= 0, axis) &
                np.all(np.isclose(np.add.reduceat(
                    mix, self.role_starts, axis), 1), axis))

    def mixture_project(self, mixture):
        """Project a mixture array onto the simplotope"""
        mixture = np.asarray(mixture, float)
        return np.concatenate(
            [utils.simplex_project(r) for r
             in np.split(mixture, self.role_starts[1:], -1)], -1)

    def to_simplex(self, mixture):
        """Convert a mixture to a simplex

        The simplex will have dimension `num_role_strats - num_roles + 1`. This
        uses the ray tracing homotopy. The uniform mixtures are aligned, and
        then rays are extended to the edges to convert proportionally along
        those rays on each object.
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
        return simp_center + ratio * simp_grad

    def from_simplex(self, simp):
        """Convery a simplex back into a valid mixture

        This is the inverse function of to_simplex."""
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
        return center + ratio * grad

    def uniform_mixture(self):
        """Returns a uniform mixed profile"""
        return np.repeat(1 / self.num_role_strats, self.num_role_strats)

    def random_mixtures(self, num_samples=None, *, alpha=1):
        """Return a random mixed profile

        Mixed profiles are sampled from a dirichlet distribution with parameter
        alpha. If alpha = 1 (the default) this is a uniform distribution over
        the simplex for each role. alpha \in (0, 1) is baised towards high
        entropy mixtures, i.e. mixtures where one strategy is played in
        majority. alpha \in (1, oo) is baised towards low entropy (uniform)
        mixtures. If `num_samples` is None, a single mixture is returned.
        """
        if num_samples is None:
            return self.random_mixtures(1, alpha=alpha)[0]
        mixtures = rand.gamma(alpha, 1, (num_samples, self.num_strats))
        mixtures /= np.add.reduceat(mixtures, self.role_starts,
                                    1).repeat(self.num_role_strats, 1)
        return mixtures

    def random_sparse_mixtures(self, num_samples=None, *, alpha=1,
                               support_prob=None, normalize=True):
        """Return a random sparse mixed profile

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to be retuned, if None or unspecified, a
            single sample without the extra dimension is returned.
        alpha : float, optional, (0, oo)
            Mixed profiles are sampled from a dirichlet distribution with
            parameter alpha. If alpha = 1 (the default) this is a uniform
            distribution over the simplex for each role. alpha \in (0, 1) is
            baised towards high entropy mixtures, i.e. mixtures where one
            strategy is played in majority. alpha \in (1, oo) is baised towards
            low entropy (uniform) mixtures.
        support_prob : float, ndarray, optional, (0, 1)
            The probability that a given strategy is in support. If support
            prob is None, supports will be sampled uniformly.
        normalize : bool, optional
            If true, the mixtures are normalized, so that the conditional
            probability of any strategy in support equals support prob. If
            true, the support_prob for any role must be at least `1 /
            num_role_strats`.
        """
        if num_samples is None:
            return self.random_sparse_mixtures(
                1, alpha=alpha, support_prob=support_prob,
                normalize=normalize)[0]
        mixtures = rand.gamma(alpha, 1, (num_samples, self.num_strats))
        mixtures *= self.random_subgames(num_samples, support_prob, normalize)
        mixtures /= np.add.reduceat(mixtures, self.role_starts,
                                    1).repeat(self.num_role_strats, 1)
        return mixtures

    def biased_mixtures(self, bias=.9):
        """Generates mixtures biased towards one strategy for each role

        Each role has one strategy played with probability bias; the reamaining
        1-bias probability is distributed uniformly over the remaining S or S-1
        strategies. If there's only one strategy, it is played with probability
        1."""
        assert 0 <= bias <= 1, "probabilities must be between zero and one"

        role_mixtures = []
        for num_strats in self.num_role_strats:
            if num_strats == 1:
                mix = np.ones((1, 1))
            else:
                mix = np.empty((num_strats, num_strats))
                mix.fill((1 - bias) / (num_strats - 1))
                np.fill_diagonal(mix, bias)
            role_mixtures.append(mix)

        return utils.acartesian2(*role_mixtures)

    def role_biased_mixtures(self, bias=0.9):
        """Generates mixtures where one role-strategy is played with bias

        If no roles have more than one strategy (a degenerate game), then this
        returns nothing."""
        assert 0 <= bias <= 1, "probabilities must be between zero and one"

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
        return self.pure_subgames().astype(float)

    def grid_mixtures(self, num_points):
        """Returns all of the mixtures in a grid with n points

        Arguments
        ---------
        num_points : int > 1
            The number of points to have along one dimensions
        """
        assert num_points > 1, "Must have at least two points on a dimensions"
        role_mixtures = [utils.acomb(num_strats, num_points - 1, True) /
                         (num_points - 1)
                         for num_strats in self.num_role_strats]
        return utils.acartesian2(*role_mixtures)


class _BaseGame(_StratArray):
    """Role-symmetric game representation

    This object only contains methods and information about definition of the
    game, and does not contain methods to operate on game data.

    Parameters
    ----------
    num_role_players :  ndarray
        The number of players in each role in order, or the number of players
        per role if identical (will be broadcast to match the number of roles).
    num_role_strats : ndarray
        The number of strategies in each role in order, or the number of
        strategies per role if identical (will be broadcast to match the number
        of roles).

    Notes
    -----
    The number of roles is deduced from the number of entries in
    num_role_players and num_role_strats. If either is an integer or has length
    1, the other is used; if both are integers or have length 1, the game will
    have one role.

    A few functions should be provided by subclasses:
    - `max_strat_payoffs()`
    - `min_strat_payoffs()`
    - `get_payoffs()`
    - `deviation_payoffs()`
    - `subgame()`
    - `normalize()`
    - `__contains__()`
    - `num_profiles`
    - `num_complete_profiles`
    - `profiles`
    - `payoffs`

    Attributes
    ----------
    num_role_players : ndarray, int, (num_roles,)
        The number of players in each role.
    num_players : int
        The total number of players.
    num_role_strats : ndarray, int, (num_roles,)
        The number of strategies in each role.
    num_role_strats : int
        The total number of strategies.
    zero_prob : ndarray, float, (num_roles,)
        The probability of a mixture for a role, below which it should be
        considered zero.
    """

    def __init__(self, num_role_players, num_role_strats):
        assert num_role_players.ndim == 1
        assert num_role_players.shape == num_role_strats.shape
        # This test for equality because we get games with zero players when
        # deviating, in the same way that 1 strategy is technically degenerate
        assert np.all(num_role_players >= 0)
        super().__init__(num_role_strats)
        self.num_role_players = num_role_players
        self.num_role_players.setflags(write=False)
        self.num_players = self.num_role_players.sum()
        self.zero_prob = np.finfo(float).tiny * (self.num_role_players + 1)
        self.zero_prob.setflags(write=False)

    def min_role_payoffs(self):
        """Returns the minimum payoff for each role"""
        return np.fmin.reduceat(self.min_strat_payoffs(), self.role_starts)

    def max_role_payoffs(self):
        """Returns the maximum payoff for each role"""
        return np.fmax.reduceat(self.max_strat_payoffs(), self.role_starts)

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
        # XXX Ideally this would be self.num_all_role_profiles.prod() with a
        # check for overflow, but there is no check for overflow on array
        # operations, so we have to do this manually. Another option would be
        # to cast as a float and then for overflow on returning to an int, but
        # this seems more straightforward
        return self.num_all_role_profiles.astype(object).prod()

    @property
    @utils.memoize
    def num_all_payoffs(self):
        """The number of payoffs in all profiles"""
        dev_players = self.num_role_players - np.eye(self.num_roles, dtype=int)
        return np.sum(utils.game_size(dev_players, self.num_role_strats)
                      .prod(1) * self.num_role_strats)

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
        else:
            profile_sums = np.sum(self.profiles * self.payoffs, 1)
            return np.allclose(profile_sums, profile_sums[0])

    # TODO Implement inverse
    def profile_id(self, profiles):
        """Return a unique integer representing a profile"""
        profiles = -np.asarray(profiles, int)
        profiles[..., self.role_starts] += self.num_role_players
        profiles = profiles.cumsum(-1)
        rev_arange = -np.ones(self.num_strats, int)
        rev_arange[self.role_starts] += self.num_role_strats
        rev_arange = rev_arange.cumsum()
        rprofs = self.num_all_role_profiles
        sizes = utils.game_size(rev_arange, profiles)
        if self.num_all_profiles > np.iinfo(int).max:
            rprofs = rprofs.astype(object)
            sizes = sizes.astype(object)

        # TODO Base should probably be cached
        # XXX Base is reversed so that profile_ids are ascending
        base = np.insert(rprofs[:0:-1].cumprod()[::-1], self.num_roles - 1, 1)
        return np.add.reduceat(sizes, self.role_starts, -1).dot(base)

    def get_expected_payoffs(self, mix, *, jacobian=False,
                             deviations=None):
        """Returns the payoff of each role under mixture

        If the payoffs for deviating from `mix` is already known, that an be
        passed in to save computation."""
        mix = np.asarray(mix, float)
        if jacobian:
            if deviations is None:
                deviations, dev_jac = self.deviation_payoffs(
                    mix, jacobian=True)
            else:
                deviations, dev_jac = deviations
            # Don't care about that missing data
            deviations[mix < self.zero_prob.repeat(self.num_role_strats)] = 0
            # Don't care about that missing data
            dev_jac[mix < self.zero_prob.repeat(self.num_role_strats)] = 0
            expected_payoffs = np.add.reduceat(
                mix * deviations, self.role_starts)
            jac = np.add.reduceat(
                mix[:, None] * dev_jac, self.role_starts, 0) + deviations
            return expected_payoffs, jac

        else:
            if deviations is None:
                deviations = self.deviation_payoffs(mix)
            # Don't care about that missing data
            deviations[mix < self.zero_prob.repeat(self.num_role_strats)] = 0
            return np.add.reduceat(mix * deviations, self.role_starts)

    def best_response(self, mix):
        """Returns the best response to a mixture

        The result is a new mixture with uniform support over all best
        deviating strategies.
        """
        responses = self.deviation_payoffs(mix)
        bests = np.maximum.reduceat(responses, self.role_starts)
        best_resps = responses == bests.repeat(self.num_role_strats)
        return best_resps / np.add.reduceat(
            best_resps, self.role_starts).repeat(self.num_role_strats)

    def is_profile(self, prof, *, axis=-1):
        """Verify that a profile is valid for game"""
        prof = np.asarray(prof, int)
        assert prof.shape[axis] == self.num_strats
        play_shape = [1] * prof.ndim
        play_shape[axis] = self.num_roles
        return (
            np.all(self.num_role_players.reshape(play_shape) ==
                   np.add.reduceat(prof, self.role_starts, axis), axis) &
            np.all(prof >= 0, axis))

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
        return (self.pure_subgames() *
                self.num_role_players.repeat(self.num_role_strats))

    def nearby_profiles(self, profile, num_devs):
        """Returns profiles reachable by at most num_devs deviations"""
        # TODO This is pretty slow and could probably be sped up
        assert num_devs >= 0
        profile = np.asarray(profile, int)
        dev_players = utils.acomb(self.num_roles, num_devs, True)
        mask = np.all(dev_players <= self.num_role_players, 1)
        dev_players = dev_players[mask]
        supp = profile > 0
        sub_strats = np.add.reduceat(supp, self.role_starts)

        profiles = [profile[None]]
        for players in dev_players:
            to_dev_profs = emptygame(
                players, self.num_role_strats).all_profiles()
            sub = emptygame(players, sub_strats)
            from_dev_profs = np.zeros((sub.num_all_profiles,
                                       self.num_strats), int)
            from_dev_profs[:, supp] = sub.all_profiles()
            before_devs = profile - from_dev_profs
            before_devs = before_devs[np.all(before_devs >= 0, 1)]
            before_devs = utils.unique_axis(before_devs)
            nearby = before_devs[:, None] + to_dev_profs
            nearby.shape = (-1, self.num_strats)
            profiles.append(utils.unique_axis(nearby))
        return utils.unique_axis(np.concatenate(profiles))

    def random_profiles(self, num_samples=None, mixture=None):
        """Sample profiles from a mixture

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to return. If None or omitted, then a single
            sample, without a leading singleton dimension is returned.
        mixture : ndarray, optional
            Mixture to sample from, of None or omitted, then uses the uniform
            mixture.
        """
        if num_samples is None:
            return self.random_profiles(1, mixture)[0]
        if mixture is None:
            mixture = self.uniform_mixture()
        role_samples = [rand.multinomial(n, probs, num_samples) for n, probs
                        in zip(self.num_role_players,
                               np.split(mixture, self.role_starts[1:]))]
        return np.concatenate(role_samples, 1)

    def random_dev_profiles(self, mixture, num_samples=None):
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
        if num_samples is None:
            return self.random_dev_profiles(mixture, 1)[0]
        dev_players = self.num_role_players - np.eye(self.num_roles, dtype=int)
        profs = np.empty((num_samples, self.num_roles, self.num_strats),
                         int)
        for i, players in enumerate(dev_players):
            base = emptygame(players, self.num_role_strats)
            profs[:, i] = base.random_profiles(num_samples, mixture)
        return profs

    def random_deviator_profiles(self, mixture, num_samples=None):
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
        devs = self.random_dev_profiles(mixture, num_samples)
        return (devs.repeat(self.num_role_strats, -2) +
                np.eye(self.num_strats, dtype=int))

    def max_prob_prof(self, mix):
        """Returns the pure strategy profile with highest probability."""
        mix = np.asarray(mix, float)
        return np.concatenate(
            [utils.multinomial_mode(m, p) for m, p
             in zip(np.split(mix, self.role_starts[1:]),
                    self.num_role_players)], -1)

    @utils.memoize
    def is_symmetric(self):
        """Returns true if this game is symmetric"""
        return self.num_roles == 1

    @utils.memoize
    def is_asymmetric(self):
        """Returns true if this game is asymmetric"""
        return np.all(self.num_role_players == 1)

    @utils.memoize
    def __hash__(self):
        return hash((type(self), self.num_roles,
                     self.num_role_strats.tobytes(),
                     self.num_role_players.tobytes()))

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.num_roles == other.num_roles and
                np.all(self.num_role_strats == other.num_role_strats) and
                np.all(self.num_role_players == other.num_role_players))

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.num_role_players,
            self.num_role_strats)


class Game(_BaseGame):
    """Role-symmetric game representation

    This representation uses a sparse mapping from profiles to payoffs for role
    symmetric games. Payoffs for specific players in a profile can be nan to
    indicate they are missing. The profiles will not be listed in
    `num_complete_profiles` or counted as `in` the game, but their data can be
    accessed via `get_payoffs`, and they will be used for calculating deviation
    payoffs if possible.

    Parameters
    ----------
    num_role_players : ndarray or int
        The number of players per role. See BaseGame.
    num_role_strats : ndarray or int or
        The number of strategies per role.
    profiles : ndarray-like, (num_payoffs, num_role_strats)
        The profiles for the game.
    payoffs : ndarray-like, (num_payoffs, num_role_strats)
        The payoffs for the game.

    Attributes
    ----------
    profiles : ndarray, int
        All of the profiles
    payoffs : ndarray, float
        All of the payoffs
    num_profiles : int
        Number of profiles with at least partial data.
    num_complete_profiles : int
        Number of profiles with complete data.
    """

    def __init__(self, num_role_players, num_role_strats, profiles, payoffs):
        super().__init__(num_role_players, num_role_strats)

        assert profiles.shape == payoffs.shape, \
            "profiles and payoffs must be the same shape {} {}".format(
                profiles.shape, payoffs.shape)
        assert profiles.shape[1:] == (self.num_strats,), \
            "profiles must have proper end shape : expected {} but was {}" \
            .format((self.num_strats,), profiles.shape[1:])
        assert np.all(profiles >= 0), "profiles was negative"
        assert np.all(
            np.add.reduceat(profiles, self.role_starts, 1) ==
            self.num_role_players), \
            "not all profiles equaled player total"
        assert np.all(payoffs[profiles == 0] == 0), \
            "there were nonzero payoffs for strategies without players"
        assert not np.all(np.isnan(payoffs) | (profiles == 0), 1).any(), \
            "A profile had entirely nan payoffs"

        self.profiles = profiles
        self.profiles.setflags(write=False)
        self.payoffs = payoffs
        self.payoffs.setflags(write=False)
        self.num_profiles = profiles.shape[0]

        # compute log dev reps
        player_factorial = np.sum(sps.gammaln(self.profiles + 1), 1)
        totals = (np.sum(sps.gammaln(self.num_role_players + 1)) -
                  player_factorial)
        with np.errstate(divide='ignore'):
            self._dev_reps = (
                totals[:, None] + np.log(self.profiles) -
                np.log(self.num_role_players).repeat(self.num_role_strats))
        self._dev_reps.setflags(write=False)

        # Add profile lookup
        self._profile_map = dict(zip(map(utils.hash_array, self.profiles),
                                     self.payoffs))
        if np.isnan(self.payoffs).any():
            self._complete_profiles = frozenset(
                prof for prof, pay in self._profile_map.items()
                if not np.isnan(pay).any())
        else:  # Don't need to store duplicate lookup object
            self._complete_profiles = self._profile_map
        self.num_complete_profiles = len(self._complete_profiles)
        assert len(self._profile_map) == self.num_profiles, \
            "There was at least one duplicate profile"

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns the minimum payoff for each role"""
        if not self.num_profiles:
            pays = np.empty(self.num_strats)
            pays.fill(np.nan)
        else:
            pays = np.fmin.reduce(np.where(
                self.profiles > 0, self.payoffs, np.nan), 0)
        pays.setflags(write=False)
        return pays

    @utils.memoize
    def max_strat_payoffs(self):
        """Returns the maximum payoff for each role"""
        if not self.num_profiles:
            pays = np.empty(self.num_strats)
            pays.fill(np.nan)
        else:
            pays = np.fmax.reduce(np.where(
                self.profiles > 0, self.payoffs, np.nan), 0)
        pays.setflags(write=False)
        return pays

    def get_payoffs(self, profiles):
        """Returns an array of profile payoffs

        If profile is not in game, an array of nans is returned where profile
        has support."""
        profiles = np.asarray(profiles, int)
        assert self.is_profile(profiles).all()
        prof_view = profiles.reshape((-1, self.num_strats))
        payoffs = np.empty(prof_view.shape, float)
        for prof, pay in zip(prof_view, payoffs):
            hashed = utils.hash_array(prof)
            if hashed not in self._profile_map:
                pay[prof == 0] = 0
                pay[prof > 0] = np.nan
            else:
                np.copyto(pay, self._profile_map[hashed])
        return payoffs.reshape(profiles.shape)

    def deviation_payoffs(self, mix, *, jacobian=False):
        """Computes the expected value of each pure strategy played against all
        opponents playing mix.

        Parameters
        ----------
        mix : ndarray
            The mix all other players are using
        jacobian : bool
            If true, the second returned argument will be the jacobian of the
            deviation payoffs with respect to the mixture. The first axis is
            the deviating strategy, the second axis is the strategy in the mix
            the jacobian is taken with respect to. The values that are marked
            nan are not very aggressive, so don't rely on accurate nan values
            in the jacobian.
        """
        # TODO It wouldn't be hard to extend this to multiple mixtures, which
        # would allow array calculation of mixture regret. Support would have
        # to be iterative though.
        mix = np.asarray(mix, float)
        nan_mask = np.empty_like(mix, dtype=bool)

        # Fill out mask where we don't have data
        if self.is_complete():
            nan_mask.fill(False)
        elif self.is_empty():
            nan_mask.fill(True)
        else:
            # These calculations are approximate, but for games we can do
            # anything with, the size is bounded, and so numeric methods are
            # actually exact.
            support = mix > 0
            strats = np.add.reduceat(support, self.role_starts)
            devs = self.profiles[:, ~support]
            num_supp = utils.game_size(self.num_role_players, strats).prod()
            dev_players = self.num_role_players - \
                np.eye(self.num_roles, dtype=int)
            role_num_dev = utils.game_size(dev_players, strats).prod(1)
            num_dev = role_num_dev.repeat(self.num_role_strats)[~support]

            nan_mask[support] = np.all(devs == 0, 1).sum() < num_supp
            nan_mask[~support] = devs[devs.sum(1) == 1].sum(0) < num_dev

        # Compute values
        if not nan_mask.all():
            # zero_prob effectively makes 0^0=1 and 0/0=0.
            log_mix = np.log(mix + _TINY)
            prof_prob = np.sum(self.profiles * log_mix, 1, keepdims=True)
            with np.errstate(under='ignore'):
                # Ignore underflow caused when profile probability is not
                # representable in floating point.
                probs = np.exp(prof_prob + self._dev_reps - log_mix)
            zero_prob = _TINY * self.num_players
            # Mask out nans
            weighted_payoffs = probs * np.where(probs > zero_prob,
                                                self.payoffs, 0)
            devs = np.sum(weighted_payoffs, 0)

        else:
            devs = np.empty(self.num_strats)

        devs[nan_mask] = np.nan

        if not jacobian:
            return devs

        if not nan_mask.all():
            tmix = mix + self.zero_prob.repeat(self.num_role_strats)
            product_rule = self.profiles[:, None] / tmix - np.diag(1 / tmix)
            dev_jac = np.sum(weighted_payoffs[..., None] * product_rule, 0)
            dev_jac -= np.repeat(
                np.add.reduceat(dev_jac, self.role_starts, 1) /
                self.num_role_strats, self.num_role_strats, 1)
        else:
            dev_jac = np.empty((self.num_strats, self.num_strats))

        dev_jac[nan_mask] = np.nan
        return devs, dev_jac

    def normalize(self):
        """Return a normalized game"""
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0)] = 1
        offset = np.repeat(self.min_role_payoffs(), self.num_role_strats)
        payoffs = (self.payoffs - offset) / scale.repeat(self.num_role_strats)
        payoffs[self.profiles == 0] = 0
        return game_replace(self, self.profiles, payoffs)

    def subgame(self, subgame_mask):
        """Remove possible strategies from consideration"""
        subgame_mask = np.asarray(subgame_mask, bool)
        assert self.is_subgame(subgame_mask), \
            "subgame_mask must be a valid subgame"
        num_strats = np.add.reduceat(subgame_mask, self.role_starts)
        prof_mask = ~np.any(self.profiles * ~subgame_mask, 1)
        profiles = self.profiles[prof_mask][:, subgame_mask]
        payoffs = self.payoffs[prof_mask][:, subgame_mask]
        return game(self.num_role_players, num_strats, profiles, payoffs)

    def __contains__(self, profile):
        """Returns true if all data for that profile exists"""
        return (utils.hash_array(np.asarray(profile, int))
                in self._complete_profiles)

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_profiles,
                     self.num_complete_profiles))

    def __eq__(self, other):
        return (super().__eq__(other) and
                # Identical profiles
                self.num_profiles == other.num_profiles and
                self.num_complete_profiles == other.num_complete_profiles and
                not np.setxor1d(utils.axis_to_elem(self.profiles),
                                utils.axis_to_elem(other.profiles)).size and
                # Identical payoffs
                all(np.allclose(pay, other.get_payoffs(prof), equal_nan=True)
                    for prof, pay in zip(self.profiles, self.payoffs)))

    def __repr__(self):
        return '{old}, {data:d} / {total:d})'.format(
            old=super().__repr__()[:-1],
            data=self.num_profiles,
            total=self.num_all_profiles)


def emptygame(num_role_players, num_role_strats):
    """Create an empty game

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
    num_roles = max(num_role_players.size, num_role_strats.size)
    num_role_players = np.broadcast_to(num_role_players, num_roles)
    num_role_strats = np.broadcast_to(num_role_strats, num_roles)
    num_strats = num_role_strats.sum()
    profiles = np.empty((0, num_strats), int)
    payoffs = np.empty((0, num_strats), float)
    return Game(num_role_players, num_role_strats, profiles, payoffs)


def emptygame_copy(copy_game):
    """Copy parameters of a game into an empty game

    Useful to keep convenience methods of game without attached data.

    Parameters
    ----------
    copy_game : BaseGame
        Game to copy info from.
    """
    return emptygame(copy_game.num_role_players, copy_game.num_role_strats)


def game(num_role_players, num_role_strats, profiles, payoffs):
    """Static game constructor

    Parameters
    ----------
    num_role_players : ndarray-like, int,
        The number of players per role. See emptygame.
    num_role_strats : ndarray-like, int,
        The number of strategies per role.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    payoffs : ndarray-like, float, (num_profiles, num_strats)
        The payoffs for the game.
    """
    return game_replace(emptygame(num_role_players, num_role_strats), profiles,
                        payoffs)


def game_copy(copy_game):
    """Copy structure and payoffs from an existing game

    Parameters
    ----------
    copy_game : BaseGame
        Game to copy data from
    """
    return game_replace(copy_game, copy_game.profiles, copy_game.payoffs)


def game_replace(copy_game, profiles, payoffs):
    """Copy structure from an existing game with new data

    Parameters
    ----------
    copy_game : Game
        Game to copy information out of.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    payoffs : ndarray-like, float, (num_profiles, num_strats)
        The payoffs for the game.
    """
    return Game(copy_game.num_role_players, copy_game.num_role_strats,
                np.asarray(profiles, int), np.asarray(payoffs, float))


class SampleGame(Game):
    """A Role Symmetric Game that has multiple samples per observation

    This behaves the same as a normal Game object, except that it has a
    `resample` method, which will resample the used payoffs from the empirical
    distribution of payoffs, allowing bootstrapping over arbitrary statistics,
    and a `get_sample_payoffs` method that will return all of the sample
    payoffs.

    Parameters
    ----------
    num_role_players : ndarray, int
        The number of players per role.
    num_role_strats : ndarray, int
        The number of strategies per role.
    profiles : ndarray-like
        The profiles for the game.
    sample_payoffs : [ndarray-like]
        The sample payoffs for the game. Each element of the list is a set of
        payoff observations grouped by number of observations and parallel with
        profiles.

    Attributes
    ----------
    sample_payoffs : [ndarray]
        This structure contains all of the sample payoffs grouped by number of
        observations and alligned with `profiles`. Each element array is
        indexed by profile, then strategy, then observation number.
    """

    def __init__(self, num_role_players, num_role_strats, profiles,
                 sample_payoffs):
        assert len(set(x.shape[1] for x in sample_payoffs)) <= 1, \
            "Not all sample payoffs had compatible numbers of strategies"
        assert not any(pays.size == 0 for pays in sample_payoffs), \
            "Sample payoffs can't be empty"

        # In case an empty list is passed
        if sample_payoffs:
            payoffs = np.concatenate([s.mean(2) for s in sample_payoffs])
        else:
            payoffs = np.empty((0, profiles.shape[1]))

        super().__init__(num_role_players, num_role_strats, profiles, payoffs)

        self.sample_payoffs = sample_payoffs
        for spay in self.sample_payoffs:
            spay.setflags(write=False)
        self.num_sample_profs = np.fromiter(
            (x.shape[0] for x in self.sample_payoffs),
            int, len(self.sample_payoffs))
        self.sample_starts = np.insert(self.num_sample_profs[:-1].cumsum(), 0,
                                       0)
        self.num_samples = np.fromiter(
            (v.shape[2] for v in self.sample_payoffs),
            int, len(self.sample_payoffs))
        assert self.num_samples.size == np.unique(self.num_samples).size, \
            "Each set of observations must have a unique number or be merged"

        assert not self.sample_payoffs or all(
            (samp[np.broadcast_to(count[..., None], samp.shape) == 0] == 0)
            .all() for count, samp
            in zip(np.split(self.profiles, self.sample_starts[1:]),
                   self.sample_payoffs)), \
            "some sample payoffs were nonzero for invalid payoffs"

        self._sample_profile_map = None

    def resample(self, num_resamples=None, *, independent_profile=False,
                 independent_role=False, independent_strategy=False):
        """Overwrite payoff values with a bootstrap resample

        Keyword Arguments
        -----------------
        num_resamples:        The number of resamples to take for each realized
                              payoff. By default this is equal to the number of
                              observations for that profile.
        independent_profile:  Sample each profile independently. In general,
                              only profiles with a different number of
                              observations will be resampled independently.
                              (default: False)
        independent_role:     Sample each role independently. Within a profile,
                              the payoffs for each role will be drawn
                              independently. (default: False)
        independent_strategy: Sample each strategy independently. Within a
                              profile, the payoffs for each strategy will be
                              drawn independently. (default: False)

        Each of the `independent_` arguments will increase the time to do a
        resample. `independent_strategy` doesn't make any particular sense.
        """
        if independent_role and independent_strategy:
            dim2 = self.num_strats
        elif independent_role:
            dim2 = self.num_roles
        elif independent_strategy:
            dim2 = self.num_role_strats.max()
            rep_inds = np.arange(self.num_strats) - \
                self.role_starts[self.role_indices]
        else:
            dim2 = 1

        payoffs = np.empty_like(self.payoffs)
        for obs, pays in zip(self.sample_payoffs,
                             np.split(payoffs, self.sample_starts[1:])):
            num_samples = obs.shape[2]
            num_obs_resamples = (num_samples if num_resamples is None
                                 else num_resamples)
            dim1 = obs.shape[0] if independent_profile else 1
            sample = rand.multinomial(num_obs_resamples,
                                      np.ones(num_samples) / num_samples,
                                      (dim1, dim2))
            if independent_role and not independent_strategy:
                sample = sample.repeat(self.num_role_strats, 1)
            elif independent_strategy and not independent_role:
                sample = sample[:, rep_inds]
            np.copyto(pays, (np.mean(obs * sample, 2) * (num_samples /
                                                         num_obs_resamples)))
        return Game(self.num_role_players, self.num_role_strats, self.profiles,
                    payoffs)

    def get_sample_payoffs(self, profile):
        """Get sample payoffs associated with a profile

        This returns an array of shape (num_observations, num_role_strats). If
        profile has no data, num_observations will be 0."""
        if self._sample_profile_map is None:
            self._sample_profile_map = dict(zip(
                map(utils.hash_array, self.profiles),
                itertools.chain.from_iterable(self.sample_payoffs)))
        profile = np.asarray(profile, int)
        assert self.is_profile(profile)
        hashed = utils.hash_array(profile)
        if hashed not in self._sample_profile_map:
            return np.empty((0, self.num_strats), float)
        else:
            return self._sample_profile_map[hashed].T

    @property
    @functools.lru_cache(maxsize=1)
    def flat_profiles(self):
        """Profiles in parallel with flat_payoffs"""
        return self.profiles.repeat(
            self.num_samples.repeat(self.num_sample_profs), 0)

    @property
    @functools.lru_cache(maxsize=1)
    def flat_payoffs(self):
        """All sample payoffs linearly concatenated together"""
        return np.concatenate([
            np.rollaxis(pay, 2, 1).reshape((-1, self.num_strats))
            for pay in self.sample_payoffs])

    def normalize(self):
        """Return a normalized SampleGame"""
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0)] = 1
        scale = scale.repeat(self.num_role_strats)[:, None]
        offset = self.min_role_payoffs().repeat(self.num_role_strats)[:, None]
        spayoffs = [(pays - offset) / scale for pays in self.sample_payoffs]
        for profs, pays in zip(np.split(self.profiles, self.sample_starts[1:]),
                               spayoffs):
            pays[profs == 0] = 0
        return samplegame_replace(self, self.profiles, spayoffs)

    def subgame(self, subgame_mask):
        """Remove possible strategies from consideration"""
        subgame_mask = np.asarray(subgame_mask, bool)
        assert self.is_subgame(subgame_mask), \
            "subgame_mask must be a valid subgame"
        num_strats = np.add.reduceat(subgame_mask, self.role_starts)
        prof_mask = ~np.any(self.profiles * ~subgame_mask, 1)
        profiles = self.profiles[prof_mask][:, subgame_mask]
        sample_payoffs = [pays[pmask][:, subgame_mask]
                          for pays, pmask
                          in zip(self.sample_payoffs,
                                 np.split(prof_mask, self.sample_starts[1:]))
                          if pmask.any()]
        return samplegame(self.num_role_players, num_strats, profiles,
                          sample_payoffs)

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), tuple(sorted(self.num_samples))))

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            # Identical sample payoffs
            all(_sample_payoffs_equal(pay.T, other.get_sample_payoffs(prof))
                for prof, pay in zip(
                    self.profiles,
                    itertools.chain.from_iterable(self.sample_payoffs))))

    def __repr__(self):
        samples = self.num_samples
        if samples.size == 0:
            sample_str = '0'
        elif samples.size == 1:
            sample_str = str(samples[0])
        else:
            sample_str = '{:d} - {:d}'.format(samples.min(), samples.max())
        return '{}, {})'.format(super().__repr__()[:-1], sample_str)


def _sample_payoffs_equal(p1, p2):
    """Returns true if two sample payoffs are almost equal"""
    # FIXME Pathological payoffs will make this fail
    return (p1.shape[0] == p2.shape[0] and
            np.allclose(p1[np.lexsort(p1.T)], p2[np.lexsort(p2.T)],
                        equal_nan=True))


def samplegame(num_role_players, num_role_strats, profiles,
               sample_payoffs):
    """Static SampleGame constructor

    Parameters
    ----------
    num_role_players : ndarray-like, int
        The number of players per role. See emptygame.
    num_role_strats : ndarray-like, int
        The number of strategies per role. See emptygame.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    sample_payoffs : [ndarray-like, float]
        The sample payoffs for the game. Each list element is a set of payoff
        observations grouped by number of observations and parallel with
        profiles. The shape of each element must be (num_profiles_i,
        num_strats, num_obs_i), where num_profiles_i is the number of profiles
        that have that observation count, and num_obs_i is the number of
        observations for that bucket.
    """
    return samplegame_replace(emptygame(num_role_players, num_role_strats),
                              profiles, sample_payoffs)


def samplegame_copy(copy_game):
    """Construct SampleGame from a copy

    Parameters
    ----------
    copy_game : Game
        Game to copy data from.
    """
    if hasattr(copy_game, 'sample_payoffs'):
        sample_payoffs = copy_game.sample_payoffs
    elif not copy_game.is_empty():
        sample_payoffs = [copy_game.payoffs[..., None]]
    else:
        sample_payoffs = []
    return samplegame_replace(copy_game, copy_game.profiles, sample_payoffs)


def samplegame_replace(copy_game, profiles, sample_payoffs):
    """Construct SampleGame from base game

    Parameters
    ----------
    copy_game : BaseGame, optional
        Game to copy information out of.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    sample_payoffs : [ndarray-like, float]
        The sample payoffs for the game. See samplegame.
    """
    return SampleGame(copy_game.num_role_players, copy_game.num_role_strats,
                      np.asarray(profiles, int),
                      tuple(np.asarray(p, float) for p in sample_payoffs))


class _CompleteGame(_BaseGame):
    """A game that defines everything for complete games"""

    def __init__(self, num_role_players, num_role_strats):
        super().__init__(num_role_players, num_role_strats)
        self.num_profiles = self.num_complete_profiles = self.num_all_profiles

    def __contains__(self, profile):
        assert self.is_profile(profile)
        return True
