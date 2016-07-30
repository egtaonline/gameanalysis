"""Module for Role Symmetric Game data structures

There are three types of games:
    BaseGame   - Have no data but do contain convenience methods for working
                 with games in general. This should be extended by every object
                 that can function as a game.
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

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import utils


_TINY = np.finfo(float).tiny


class BaseGame(object):
    """Role-symmetric game representation

    This object only contains methods and information about definition of the
    game, and does not contain methods to operate on observation data.

    Parameters (from game)
    ----------------------
    game : BaseGame
        Copies info from game. Useful to keep convenience methods of game
        without attached data.

    Parameters (default constructor)
    --------------------------------
    num_players : int or [int] or ndarray
        The number of players in each role in order, or the number of players
        per role if identical (will be broadcast to match the number of roles).
    num_strategies : int or [int] or ndarray
        The number of strategies in each role in order, or the number of
        strategies per role if identical (will be broadcast to match the number
        of roles).

    The number of roles is deduced from the number of entries in num_players
    and num_strategies. If either is an integer or has length 1, the other is
    used; if both are integers or have length 1, the game will have one role.
    """
    def __init__(self, *args):
        if len(args) == 1:
            # From Game
            num_players = args[0].num_players
            num_strategies = args[0].num_strategies
        elif len(args) == 2:
            # Default constructor
            num_players = args[0]
            num_strategies = args[1]
        else:
            raise ValueError('Invalid constructor arguments')

        num_players = np.asarray(num_players, int)
        num_strategies = np.asarray(num_strategies, int)
        self.num_roles = max(num_players.size, num_strategies.size)
        self.num_players = np.broadcast_to(num_players, self.num_roles)
        self.num_strategies = np.broadcast_to(num_strategies, self.num_roles)
        self.num_role_strats = self.num_strategies.sum()
        self.role_starts = np.insert(self.num_strategies[:-1].cumsum(), 0, 0)
        self.role_index = self.role_repeat(np.arange(self.num_roles))
        self.num_strategies.setflags(write=False)
        self.num_players.setflags(write=False)
        self.role_starts.setflags(write=False)
        self.role_index.setflags(write=False)
        self._hash = hash((self.num_strategies.data.tobytes(),
                           self.num_players.data.tobytes()))

        assert np.all(self.num_players >= 0)
        assert np.all(self.num_strategies > 0)

    # Functions that need to be overridden for all game functionality

    def min_payoffs(self):
        """Returns the minimum payoff for each role"""
        raise NotImplementedError('This must be overridden in deriving class')

    def max_payoffs(self):
        """Returns the maximum payoff for each role"""
        raise NotImplementedError('This must be overridden in deriving class')

    def deviation_payoffs(self, mix, assume_complete=False, jacobian=False):
        """Returns the payoff for deviating to each role from mixture

        If assume_complete, then expensive checks for missing data won't be
        made. If jacobian, a tuple is returned, where the second value is the
        jacobian with respect to the mixture."""
        raise NotImplementedError('This must be overridden in deriving class')

    # Provided functionality

    @property
    @functools.lru_cache()
    def role_sizes(self):
        """The number of profiles in each role (independent of others)"""
        return utils.game_size(self.num_players, self.num_strategies)

    @property
    @functools.lru_cache()
    def num_all_profiles(self):
        """The total number of profiles in the game

        Not just the ones with data."""
        return self.role_sizes.prod()

    @property
    @functools.lru_cache()
    def num_all_payoffs(self):
        """The number of payoffs in all profiles"""
        dev_players = self.num_players - np.eye(self.num_roles, dtype=int)
        return np.sum(utils.game_size(dev_players, self.num_strategies)
                      .prod(1) * self.num_strategies)

    @property
    @functools.lru_cache()
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
        pure_counts = np.prod(self.num_strategies * pure + ~pure, 1)
        unpure_counts = np.prod((utils.game_size(self.num_players,
                                                 self.num_strategies) -
                                 self.num_strategies) * ~pure + pure, 1)
        overcount = np.sum(cards * pure_counts * unpure_counts)
        return self.num_all_payoffs - overcount

    def role_reduce(self, array, axis=-1, ufunc=np.add, keepdims=False):
        """Reduce an array over roles

        Use this to sum the payoffs by role for a payoff array, etc.

        Parameters
        ----------
        array : ndarray
            Input array.
        ufunc : ufunc
            Numpy function to reduce with
        axis : int
            The axis to reduce over
        keepdims : bool
            If true, the shape of array will be unchanged
        """
        red = ufunc.reduceat(array, self.role_starts, axis)
        if keepdims:
            return self.role_repeat(red, axis)
        else:
            return red

    def role_split(self, array, axis=-1):
        """Split an array by roles

        Parameters
        ----------
        array : ndarray
            The array to split
        axis : int
            The axis to split along
        """
        return np.split(array, self.role_starts[1:], axis)

    def role_repeat(self, array, axis=-1):
        """Repeat an array by role

        Takes an array of shape num_roles and turns it into shape
        num_role_strats so that the arrays can interract."""
        return array.repeat(self.num_strategies, axis)

    def profile_id(self, profiles):
        """Return a unique integer representing a profile"""
        profiles = -np.asarray(profiles, int)
        profiles[..., self.role_starts] += self.num_players
        profiles = profiles.cumsum(-1)
        rev_arange = -np.ones(self.num_role_strats, int)
        rev_arange[self.role_starts] += self.num_strategies
        rev_arange = rev_arange.cumsum()
        base = np.insert(self.role_sizes[:-1].cumprod(), 0, 1)
        return self.role_reduce(utils.game_size(
            rev_arange, profiles)).dot(base)

    def get_expected_payoffs(self, mix, deviations=None):
        """Returns the payoff of each role under mixture

        If the payoffs for deviating from `mix` is already known, that an be
        passed in to save computation."""
        mix = np.asarray(mix, float)
        if deviations is None:
            deviations = self.deviation_payoffs(mix)
        deviations[mix == 0] = 0  # Don't care about that missing data
        return self.role_reduce(mix * deviations)

    def trim_mixture_support(self, mixture, supp_thresh=1e-3):
        """Trims strategies played less than supp_thresh from the support"""
        mixture *= mixture >= supp_thresh
        mixture /= self.role_reduce(mixture, keepdims=True)
        return mixture

    def verify_profile(self, prof, axis=-1):
        """Verify that a profile is valid for game"""
        prof = np.asarray(prof, int)
        return (prof.shape[axis] == self.num_role_strats and
                np.all(self.num_players == self.role_reduce(prof, axis), axis))

    def verify_mixture(self, mix, axis=-1):
        """Verify that a mixture is valid for game"""
        return np.all(np.isclose(self.role_reduce(mix, axis), 1), axis)

    def simplex_project(self, mixture):
        """Project an invalid mixture array onto the simplex"""
        return np.concatenate(list(map(utils.simplex_project,
                                       self.role_split(mixture))), -1)

    def all_profiles(self):
        """Return all profiles"""
        role_arrays = [utils.acomb(n_strats, players) for n_strats, players
                       in zip(self.num_strategies, self.num_players)]
        return utils.acartesian2(*role_arrays)

    def pure_profiles(self):
        """Return all pure profiles

        A pure profile is a profile where only one strategy is played per
        role."""
        role_profiles = [num_play * np.eye(num_strats, dtype=int)
                         for num_play, num_strats
                         in zip(self.num_players, self.num_strategies)]
        return utils.acartesian2(*role_profiles)

    def uniform_mixture(self):
        """Returns a uniform mixed profile"""
        return 1 / self.num_strategies.repeat(self.num_strategies)

    def random_profiles(self, mixture, num_samples=1):
        """Sample profiles from a mixture"""
        role_samples = [rand.multinomial(n, probs, num_samples) for n, probs
                        in zip(self.num_players, self.role_split(mixture))]
        return np.concatenate(role_samples, 1)

    def random_dev_profiles(self, mixture, num_samples=1):
        """Return partial profiles where dev player is missing

        Resulting shape of profiles is (num_samples, num_roles,
        num_role_strats). The first dimension is the sample, the next is the
        deviating role, leaving the last dimension for the partial profile."""
        dev_players = self.num_players - np.eye(self.num_roles, dtype=int)
        profs = np.empty((num_samples, self.num_roles, self.num_role_strats),
                         int)
        for i, players in enumerate(dev_players):
            base = BaseGame(players, self.num_strategies)
            profs[:, i] = base.random_profiles(mixture, num_samples)
        return profs

    def random_deviator_profiles(self, mixture, num_samples=1):
        """Return a profiles where one player is deviating from mix

        Resulting shape of profiles is (num_samples, num_role_strats,
        num_role_strats). The first dimension is the sample, the next is the
        deviating strategy, leaving the last dimension for the actual
        profile."""
        devs = self.random_dev_profiles(mixture, num_samples)
        return (self.role_repeat(devs, 1) + np.eye(self.num_role_strats,
                                                   dtype=int))

    def random_mixtures(self, num_samples=1, alpha=1):
        """Return a random mixed profile

        Mixed profiles are sampled from a dirichlet distribution with parameter
        alpha. If alpha = 1 (the default) this is a uniform distribution over
        the simplex for each role. alpha \in (0, 1) is baised towards high
        entropy mixtures, i.e. mixtures where one strategy is played in
        majority. alpha \in (1, oo) is baised towards low entropy (uniform)
        mixtures."""
        mixtures = rand.gamma(alpha, 1, (num_samples, self.num_role_strats))
        mixtures /= self.role_reduce(mixtures, axis=1, keepdims=True)
        return mixtures

    def biased_mixtures(self, bias=.9):
        """Generates mixtures biased towards one strategy for each role

        Each role has one strategy played with probability bias; the reamaining
        1-bias probability is distributed uniformly over the remaining S or S-1
        strategies. If there's only one strategy, it is played with probability
        1."""
        assert 0 <= bias <= 1, "probabilities must be between zero and one"

        role_mixtures = []
        for num_strats in self.num_strategies:
            if num_strats == 1:
                mix = np.ones((1, 1))
            else:
                mix = np.empty((num_strats, num_strats))
                mix.fill((1 - bias)/(num_strats - 1))
                np.fill_diagonal(mix, bias)
            role_mixtures.append(mix)

        return utils.acartesian2(*role_mixtures)

    def role_biased_mixtures(self, bias=0.9):
        """Generates mixtures where one role-strategy is played with bias

        If no roles have more than one strategy (a degenerate game), then this
        returns nothing."""
        assert 0 <= bias <= 1, "probabilities must be between zero and one"

        num = self.num_strategies[self.num_strategies > 1].sum()
        mixes = self.uniform_mixture()[None].repeat(num, 0)
        prof_offset = 0
        strat_offset = 0
        for num_strats in self.num_strategies:
            if num_strats > 1:
                view = mixes[prof_offset:prof_offset+num_strats,
                             strat_offset:strat_offset+num_strats]
                view.fill((1 - bias)/(num_strats - 1))
                np.fill_diagonal(view, bias)
                prof_offset += num_strats
            strat_offset += num_strats
        return mixes

    def pure_mixtures(self):
        """Returns all mixtures where the probability is either 1 or 0."""
        return self.biased_mixtures(bias=1)

    def grid_mixtures(self, num_points):
        """Returns all of the mixtures in a grid with n points

        Arguments
        ---------
        num_points : int > 1
            The number of points to have along one dimensions
        """
        assert num_points > 1, "Must have at least two points on a dimensions"
        role_mixtures = [utils.acomb(num_strats, num_points - 1) /
                         (num_points - 1)
                         for num_strats in self.num_strategies]
        return utils.acartesian2(*role_mixtures)

    def max_prob_prof(self, mix):
        """Returns the pure strategy profile with highest probability."""
        mix = np.asarray(mix, float)
        return np.concatenate(
            [utils.multinomial_mode(m, p) for m, p
             in zip(self.role_split(mix), self.num_players)], -1)

    def is_symmetric(self):
        """Returns true if this game is symmetric"""
        return self.num_roles == 1

    def is_asymmetric(self):
        """Returns true if this game is asymmetric"""
        return np.all(self.num_players == 1)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.num_players,
            self.num_strategies)


class Game(BaseGame):
    """Role-symmetric game representation

    This representation uses a sparse mapping from profiles to payoffs for role
    symmetric games. There are several variants on constructors that are all
    valid, and use combinations of various inputs, listed below. Payoffs for
    specific players in a profile can be nan to indicate they are missing. The
    profiles will not be listed in `num_complete_profiles` or counted as `in`
    the game, but their data can be accessed via `get_payoffs`, and they will
    be used for calculating deviation payoffs if possible.

    Parameters (from game)
    ----------------------
    game : BaseGame
        Game to copy information out of. This will copy as much information out
        of the game as possible.
    profiles : ndarray-like, optional
        The profiles for the game, if unspecified, this will try to be grabbed
        from `game`. Must be specified with payoffs.
    payoffs : ndarray-like, optional
        The payoffs for the game, if unspecified, payoffs will try to be
        grabbed from `game`. Must be specified with profiles.

    Parameters (from game description)
    ----------------------------------
    num_players : int or [int] or ndarray
        The number of players per role. See BaseGame.
    num_strategies : int or [int] or ndarray
        The number of strategies per role. See BaseGame.
    profiles : ndarray-like, optional
        The profiles for the game, if unspecified, game will be empty. Must be
        specified with payoffs.
    payoffs : ndarray-like, optional
        The payoffs for the game, if unspecified, game will be empty. Must be
        specified with profiles.

    Parameters (from asymmetric game)
    ---------------------------------
    matrix : ndarray-like
        The matrix of payoffs for an asymmetric game. The last axis is the
        payoffs for each player, the first axes are the strategies for each
        player. matrix.shape[:-1] must correspond to the number of strategies
        for each player. matrix.ndim - 1 must equal matrix.shape[-1].
    """

    def __init__(self, *args, verify=True):
        if len(args) == 1 and isinstance(args[0], Game):
            # From Game
            game = args[0]
            num_players = game.num_players
            num_strategies = game.num_strategies
            profiles = game.profiles.copy()
            payoffs = game.payoffs.copy()
            verify = False
        elif len(args) == 1 and isinstance(args[0], BaseGame):
            # From BaseGame
            game = args[0]
            num_players = game.num_players
            num_strategies = game.num_strategies
            profiles = np.empty((0, game.num_role_strats), int)
            payoffs = np.empty((0, game.num_role_strats))
            verify = False
        elif len(args) == 1:
            # Matrix constructor
            matrix = np.asarray(args[0], float)
            assert matrix.shape[-1] == matrix.ndim - 1, \
                "matrix shape is inconsistent with a matrix game {}".format(
                    matrix.shape)
            num_players = np.ones(matrix.shape[-1], int)
            num_strategies = np.array(matrix.shape[:-1], int)
            profiles = utils.acartesian2(*[np.eye(s, dtype=int)
                                           for s in num_strategies])
            payoffs = np.zeros(profiles.shape, float)
            payoffs[profiles > 0] = matrix.flat
            verify = False
        elif len(args) == 2:
            # Empty game
            num_players, num_strategies, = args
            num_role_strats = BaseGame(num_players,
                                       num_strategies).num_role_strats
            profiles = np.empty((0, num_role_strats), int)
            payoffs = np.empty((0, num_role_strats), float)
        elif len(args) == 3:
            # Copy base from game
            num_players = args[0].num_players
            num_strategies = args[0].num_strategies
            profiles, payoffs = args[1:]
        elif len(args) == 4:
            # Specify everything
            num_players, num_strategies, profiles, payoffs = args
        else:
            raise ValueError('Invalid constructor arguments')

        super().__init__(num_players, num_strategies)
        profiles = np.asarray(profiles, int)
        payoffs = np.asarray(payoffs)

        assert profiles.shape == payoffs.shape, \
            "profiles and payoffs must be the same shape : profiles {0}, payoffs {1}"\
            .format(profiles.shape, payoffs.shape)
        expected_shape = (profiles.shape[0], self.num_role_strats)
        assert profiles.shape == expected_shape, \
            "profiles must have proper shape : expected {0} but was {1}"\
            .format(expected_shape, profiles.shape)
        assert np.issubdtype(profiles.dtype, int), \
            "profiles must contain integers : dtype {0}".format(
                profiles.dtype)
        assert not verify or np.all(profiles >= 0), \
            "profiles was not non negative {} {}".format(
                np.any(profiles < 0, 1).nonzero(),
                profiles[profiles < 0])
        assert not verify or np.all(self.role_reduce(profiles, axis=1) ==
                                    self.num_players), \
            "not all profiles equaled player total {} {}".format(
                np.any(self.role_reduce(profiles, axis=1) ==
                       self.num_players, 1).nonzero(),
                profiles[np.any(self.role_reduce(profiles, axis=1) ==
                                self.num_players, 1)])
        assert not verify or np.all(payoffs[profiles == 0] == 0), \
            "there were nonzero payoffs for strategies without players"

        self.profiles = profiles
        self.profiles.setflags(write=False)
        self.payoffs = payoffs
        self.payoffs.setflags(write=False)
        self.num_profiles = profiles.shape[0]
        self._writeable_payoffs()  # Reset

        # compute log dev reps
        player_factorial = np.sum(sps.gammaln(self.profiles + 1), 1)
        totals = (np.sum(sps.gammaln(self.num_players + 1)) -
                  player_factorial)
        with np.errstate(divide='ignore'):
            self._dev_reps = (totals[:, None] + np.log(self.profiles) -
                              self.role_repeat(np.log(self.num_players)))
        self._dev_reps.setflags(write=False)

        # Add profile lookup
        self._profile_id_map = dict(zip(map(utils.hash_array, self.profiles),
                                        self.payoffs))
        if np.isnan(self.payoffs).any():
            self._complete_profiles = frozenset(
                prof for prof, pay in self._profile_id_map.items()
                if not np.isnan(pay).any())
        else:
            self._complete_profiles = self._profile_id_map
        self.num_complete_profiles = len(self._complete_profiles)
        assert len(self._profile_id_map) == self.num_profiles, \
            "There was at least one duplicate profile"

    def _writeable_payoffs(self):
        """Get a writable version of the payoff array

        This makes sure that internal bookkeeping is kept up to date
        """
        self.payoffs.setflags(write=True)
        view = self.payoffs.view()
        self.payoffs.setflags(write=False)
        self._min_payoffs = None
        self._max_payoffs = None
        return view

    def min_payoffs(self):
        """Returns the minimum payoff for each role"""
        if self._min_payoffs is None:
            if not self.num_profiles:
                self._min_payoffs = np.empty(self.num_strategies.shape)
                self._min_payoffs.fill(np.nan)
            else:
                masked = np.ma.masked_array(self.payoffs, self.profiles == 0)
                self._min_payoffs = self.role_reduce(
                    masked.min(0), ufunc=np.minimum).filled(np.nan)
            self._min_payoffs.setflags(write=False)
        return self._min_payoffs.view()

    def max_payoffs(self):
        """Returns the maximum payoff for each role"""
        if self._max_payoffs is None:
            if not self.num_profiles:
                self._max_payoffs = np.empty(self.num_strategies.shape)
                self._max_payoffs.fill(np.nan)
            else:
                masked = np.ma.masked_array(self.payoffs, self.profiles == 0)
                self._max_payoffs = self.role_reduce(
                    masked.max(0), ufunc=np.maximum).filled(np.nan)
            self._max_payoffs.setflags(write=False)
        return self._max_payoffs.view()

    def get_payoffs(self, profile):
        """Returns an array of profile payoffs

        if default is not None and game doesn't have profile data, then an
        array populated by default is returned."""
        profile = np.asarray(profile, int)
        assert self.verify_profile(profile)
        hashed = utils.hash_array(profile)
        if hashed not in self._profile_id_map:
            pay = np.zeros(self.num_role_strats)
            pay[profile > 0] = np.nan
            return pay
        else:
            return self._profile_id_map[hashed]

    def get_max_social_welfare(self, by_role=False):
        """Returns the maximum social welfare over the known profiles.

        If by_role is specified, then max social welfare applies to each role
        independently."""
        if by_role:
            if self.num_profiles:
                welfares = self.role_reduce(self.profiles * self.payoffs)
                prof_inds = np.nanargmax(welfares, 0)
                return (welfares[prof_inds, np.arange(self.num_roles)],
                        self.profiles[prof_inds])
            else:
                welfares = np.empty(self.num_roles)
                welfares.fill(np.nan)
                profiles = np.empty(self.num_roles, dtype=object)
                profiles.fill(None)
                return welfares, profiles

        else:
            if self.num_profiles:
                welfares = np.sum(self.profiles * self.payoffs, 1)
                prof_ind = np.nanargmax(welfares)
                return welfares[prof_ind], self.profiles[prof_ind]
            else:
                return np.nan, None

    def deviation_payoffs(self, mix, assume_complete=False, jacobian=False):
        """Computes the expected value of each pure strategy played against all
        opponents playing mix.

        Parameters
        ----------
        mix : ndarray
            The mix all other players are using
        assume_complete : bool
            If true, don't compute missing data and replace with nans. Just
            return the potentially inaccurate results.
        jacobian : bool
            If true, the second returned argument will be the jacobian of the
            deviation payoffs with respect to the mixture. The first axis is
            the deviating strategy, the second axis is the strategy in the mix
            the jacobian is taken with respect to. The values that are marked
            nan are not very aggressive, so don't rely on accurate nan values
            in the jacobian.
        """
        # TODO It wouldn't be hard to extend this to multiple mixtures, which
        # would allow array calculation of mixture regret.
        mix = np.asarray(mix, float)
        nan_mask = np.empty_like(mix, dtype=bool)

        # Fill out mask where we don't have data
        if self.is_complete() or assume_complete:
            nan_mask.fill(False)
        elif self.is_empty():
            nan_mask.fill(True)
        else:
            # These calculations are approximate, but for games we can do
            # anything with, the size is bounded, and so numeric methods are
            # actually exact.
            support = mix > 0
            strats = self.role_reduce(support)
            devs = self.profiles[:, ~support]
            num_supp = utils.game_size(self.num_players, strats).prod()
            dev_players = self.num_players - np.eye(self.num_roles, dtype=int)
            role_num_dev = utils.game_size(dev_players, strats).prod(1)
            num_dev = role_num_dev.repeat(self.num_strategies)[~support]

            nan_mask[support] = np.all(devs == 0, 1).sum() < num_supp
            nan_mask[~support] = devs[devs.sum(1) == 1].sum(0) < num_dev

        # Compute values
        if not nan_mask.all():
            # _TINY effectively makes 0^0=1 and 0/0=0.
            log_mix = np.log(mix + _TINY)
            prof_prob = np.sum(self.profiles * log_mix, 1, keepdims=True)
            with np.errstate(under='ignore'):
                # Ignore underflow caused when profile probability is not
                # representable in floating point.
                probs = np.exp(prof_prob + self._dev_reps - log_mix)
            zero_prob = self.role_repeat(_TINY * self.num_players)
            weighted_payoffs = probs * np.where(probs > zero_prob,
                                                self.payoffs, 0)
            values = np.sum(weighted_payoffs, 0)

        else:
            values = np.empty(self.num_role_strats)

        values[nan_mask] = np.nan

        if not jacobian:
            return values

        if not nan_mask.all():
            tmix = mix + self.role_repeat(_TINY * self.num_players)
            product_rule = self.profiles[:, None] / tmix - np.diag(1 / tmix)
            dev_jac = np.sum(weighted_payoffs[..., None] * product_rule, 0)
        else:
            dev_jac = np.empty((self.num_role_strats, self.num_role_strats))

        dev_jac[nan_mask] = np.nan
        return values, dev_jac

    def is_empty(self):
        """Returns true if no profiles have data"""
        return self.num_profiles == 0

    def is_complete(self):
        """Returns true if every profile has data"""
        return self.num_profiles == self.num_all_profiles

    def is_constant_sum(self):
        """Returns true if this game is constant sum"""
        if self.is_empty():
            return True
        else:
            profile_sums = np.sum(self.profiles * self.payoffs, 1)
            return np.allclose(profile_sums, profile_sums[0])

    def __contains__(self, profile):
        """Returns true if all data for that profile exists"""
        # TODO This may be slow. Potentially we should just keep a set of all
        # the ones with complete data...
        return (utils.hash_array(np.asarray(profile, int))
                in self._complete_profiles)

    def __repr__(self):
        return '{old}, {data:d} / {total:d})'.format(
            old=super().__repr__()[:-1],
            data=self.num_profiles,
            total=self.num_all_profiles)


class SampleGame(Game):
    """A Role Symmetric Game that has multiple samples per observation

    This behaves the same as a normal Game object, except that it has a
    `resample` method, which will resample the used payoffs from the empirical
    distribution of payoffs, allowing bootstrapping over arbitrary statistics.

    Parameters
    ----------
    game : BaseGame
        Game to copy information out of. This will copy as much information out
        of the game as possible.
    profiles : ndarray-like, optional
        The profiles for the game, if unspecified, this will try to be grabbed
        from `game`. Must be specified with payoffs.
    sample_payoffs : [ndarray-like], optional
        The sample payoffs for the game. Each list is a set of payoff
        observations grouped by number of observations and parallel with
        profiles. If unspecified, payoffs will try to be grabbed from `game`.
        Must be specified with profiles.

    Parameters
    ----------
    num_players : int or [int] or ndarray
        The number of players per role. See BaseGame.
    num_strategies : int or [int] or ndarray
        The number of strategies per role. See BaseGame.
    profiles : ndarray-like, optional
        The profiles for the game, if unspecified, game will be empty. Must be
        specified with payoffs.
    sample_payoffs : [ndarray-like], optional
        The sample payoffs for the game. Each list is a set of payoff
        observations grouped by number of observations and parallel with
        profiles. If unspecified, game will be empty.  Must be specified with
        profiles.

    Parameters
    ----------
    matrix : ndarray-like
        The matrix of payoffs for an asymmetric game. The last axis is the
        number of observations for each payoff, the second to last axis is
        payoffs for each player, the first axes are the strategies for each
        player. matrix.shape[:-2] must correspond to the number of strategies
        for each player. matrix.ndim - 2 must equal matrix.shape[-2].
    """

    def __init__(self, *args, verify=True):
        if len(args) == 1 and isinstance(args[0], SampleGame):
            # From SampleGame
            game = args[0]
            num_players = game.num_players
            num_strategies = game.num_strategies
            profiles = game.profiles.copy()
            sample_payoffs = [p.copy() for p in game.sample_payoffs]
            verify = False
        elif len(args) == 1 and isinstance(args[0], Game):
            # From Game
            game = args[0]
            num_players = game.num_players
            num_strategies = game.num_strategies
            profiles = game.profiles.copy()
            sample_payoffs = [game.payoffs.copy()[..., None]]
            verify = False
        elif len(args) == 1 and isinstance(args[0], BaseGame):
            # From BaseGame
            game = args[0]
            num_players = game.num_players
            num_strategies = game.num_strategies
            profiles = np.empty((0, game.num_role_strats), int)
            sample_payoffs = []
            verify = False
        elif len(args) == 1:
            # Matrix constructor
            matrix = np.asarray(args[0], float)
            assert matrix.shape[-2] == matrix.ndim - 2, \
                ("matrix shape is inconsistent with a matrix sample game {}"
                 .format(matrix.shape))
            num_players = np.ones(matrix.shape[-2], int)
            num_strategies = np.array(matrix.shape[:-2], int)
            num_samples = matrix.shape[-1]
            profiles = utils.acartesian2(*[np.eye(s, dtype=int)
                                           for s in num_strategies])
            payoffs = np.zeros(profiles.shape + (num_samples,))
            # This next set of steps is a hacky way of avoiding duplicating
            # mask by num_samples
            pview = payoffs.view()
            pview.shape = (-1, num_samples)
            mask = profiles > 0
            mask.shape = (-1, 1)
            mask = np.broadcast_to(mask, (mask.size, num_samples))
            np.place(pview, mask, matrix.flat)
            sample_payoffs = [payoffs]
            verify = False
        elif len(args) == 2:
            # Empty game
            num_players, num_strategies, = args
            num_role_strats = BaseGame(num_players,
                                       num_strategies).num_role_strats
            profiles = np.empty((0, num_role_strats), int)
            sample_payoffs = []
        elif len(args) == 3:
            # Copy base from game
            num_players = args[0].num_players
            num_strategies = args[0].num_strategies
            profiles, sample_payoffs = args[1:]
        elif len(args) == 4:
            # Specify everything
            num_players, num_strategies, profiles, sample_payoffs = args
        else:
            raise ValueError('Invalid constructor arguments')

        sample_payoffs = tuple(np.asarray(p) for p in sample_payoffs)
        assert len(set(x.shape[1] for x in sample_payoffs)) <= 1, \
            "Not all sample payoffs shapes compatible"

        # In case an empty list is passed
        if sample_payoffs:
            payoffs = np.concatenate([s.mean(2) for s in sample_payoffs])
        else:
            payoffs = np.empty((0,) + profiles.shape[1:])

        super().__init__(num_players, num_strategies, profiles, payoffs,
                         verify=verify)

        self.sample_payoffs = sample_payoffs
        for spay in self.sample_payoffs:
            spay.setflags(write=False)
        self.num_sample_profs = np.fromiter(
            (x.shape[0] for x in self.sample_payoffs),
            int, len(self.sample_payoffs))
        self.sample_starts = np.insert(self.num_sample_profs.cumsum(),
                                       0, 0)
        self.num_samples = np.fromiter(
            (v.shape[2] for v in self.sample_payoffs),
            int, len(self.sample_payoffs))

        assert not self.sample_payoffs or not verify or all(
            (samp[np.broadcast_to(count[..., None], samp.shape) == 0] == 0)
            .all() for count, samp
            in zip(np.split(self.profiles, self.sample_starts[1:]),
                   self.sample_payoffs)), \
            "some sample payoffs were nonzero for invalid payoffs"

    def remean(self):
        """Overwrite payoff values with mean payoff"""
        payoffs = self._writeable_payoffs()
        for obs, begin, end in zip(self.sample_payoffs, self.sample_starts,
                                   self.sample_starts[1:]):
            payoffs[begin:end] = obs.mean(2)

    def resample(self, num_resamples=None, independent_profile=False,
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
        switches = (independent_profile, independent_role,
                    independent_strategy)
        payoffs = self._writeable_payoffs()
        for obs, begin, end in zip(self.sample_payoffs, self.sample_starts,
                                   self.sample_starts[1:]):
            num_samples = obs.shape[2]
            num_obs_resamples = (num_samples if num_resamples is None
                                 else num_resamples)
            shape = [dim if switch else 1
                     for dim, switch in zip(obs.shape, switches)]
            sample = rand.multinomial(
                num_obs_resamples, [1/num_samples]*num_samples, shape)
            payoffs[begin:end] = ((obs * sample).mean(3) *
                                  (num_samples / num_obs_resamples))

    def __repr__(self):
        samples = self.num_samples
        if samples.size == 0:
            sample_str = '0'
        elif samples.size == 1:
            sample_str = str(samples[0])
        else:
            sample_str = '{:d} - {:d}'.format(samples.min(), samples.max())
        return '{}, {})'.format(super().__repr__()[:-1], sample_str)
