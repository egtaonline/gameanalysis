"""Module for Role Symmetric Game data structures

There are three types of games:
    EmptyGames  - Have no data but do contain convenience methods for working
                  with games in general
    Games       - Contain payoff data at a profile level, but that data can be
                  sparse
    SampleGames - Contain several samples of payoffs for every profile

There are several ways to instantiate a game:
    Constructor        - The most efficient, but requires knowledge of the
                         internal representation and can result in errors if
                         created improperly
    from_payoff_format - The standard way to represent and store games, see the
                         specific constructors. This method is not
                         significantly sparse or efficient. This input method
                         also makes sure that the role and strategy order is
                         lexicographical.
    from_json          - Similar to from_payoff_format, but can take the python
                         parsed version of game data.

A fundamental object to operating with games is the profile. A profile is a
mapping of role strategy pairs a number of agents playing that strategy, or to
probability of playing that strategy. There are two general forms: the nice
human readable form which looks like a nested dictionary, and has convenience
methods if converted to a Profile or Mixture object (see profile), and the
array version, which is a 1d ndarray with corresponding numbers that are
raveled into the 1d array. Most methods can return both versions, but usually
the array representation is more efficient.
"""
import collections
import itertools
import warnings
from collections import abc

import numpy as np
import scipy.special as sps

from gameanalysis import collect
from gameanalysis import gameio
from gameanalysis import profile
from gameanalysis import utils


_TINY = np.finfo(float).tiny


class EmptyGame(object):
    """Role-symmetric game representation

    This object only contains methods and information about definition of the
    game, and does not contain methods to operate on observation data.

    Parameters
    ----------
    players : {role: count}
        Mapping from roles to number of players per role.
    strategies : {role: {strategy}}
        Mapping from roles to per-role strategy sets. Role order is preserved
        and copied from this input dictionary. To specify an order use an
        ordered dictionary.
    """
    def __init__(self, players, strategies):
        assert frozenset(players.keys()) == \
            frozenset(strategies.keys()), \
            "Player roles and strategy roles do not agree"
        assert all(len(strats) == len(frozenset(strats)) for strats
                   in strategies.values()), \
            "Not all strategies are unique: {0}".format(strategies)

        # Dictionary of all strategies
        self.strategies = collect.fodict((r, tuple(s))
                                         for r, s in strategies.items())
        # Dictionary of all players i.e. number of players per role
        # Ensures players and strategies are in the same order
        self.players = collect.fodict((r, players[r]) for r
                                      in self.strategies)
        # Array of all role counts in order
        self.aplayers = np.fromiter(self.players.values(), int,
                                    len(self.players))
        self.aplayers.setflags(write=False)
        # Array of the number of strategies per role in order
        self.astrategies = np.fromiter(
            (len(s) for s in self.strategies.values()),
            int, len(self.strategies))
        self.astrategies.setflags(write=False)
        # The total number of strategies over roles
        self.num_role_strats = self.astrategies.sum()
        # Array necessary for doing reduce operations over role
        self._at_indices = np.hstack(([0], self.astrategies[:-1].cumsum()))
        self._at_indices.setflags(write=False)
        # Total number of profiles this game can have
        self.size = utils.prod(utils.game_size(self.players[r], len(strats))
                               for r, strats in self.strategies.items())

        # A mapping of role to index
        self._role_index = {r: i for i, r in enumerate(self.strategies)}
        # A mapping from a tuple of role, strategy to it's index in an array
        # profile
        self._role_strat_index = {
            (r, s): i for i, (r, s)
            in enumerate(itertools.chain.from_iterable(
                ((r, s) for s in strats)
                for r, strats in self.strategies.items()))}

    @staticmethod
    def from_payoff_format(players, strategies):
        sorted_strategies = collections.OrderedDict(sorted(
            (r, sorted(s)) for r, s in strategies.items()))
        return EmptyGame(players, sorted_strategies)

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        players, strategies, *_ = gameio._game_from_json(json_)
        return EmptyGame.from_payoff_format(players, strategies)

    def role_reduce(self, array, axis=0, ufunc=np.add, keepdims=False):
        """Reduce an array over roles

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
        red = ufunc.reduceat(array, self._at_indices, axis)
        if keepdims:
            return red.repeat(self.astrategies, axis)
        else:
            return red

    def role_split(self, array, axis=0):
        """Split an array by roles

        Parameters
        ----------
        array : ndarray
            The array to split
        axis : int
            The axis to split along
        """
        return np.split(array, self._at_indices[1:])

    def trim_mixture_array_support(self, mixture, supp_thresh=1e-3):
        """Trims strategies played less than supp_thresh from the support"""
        mixture *= mixture >= supp_thresh
        mixture /= self.role_reduce(mixture, keepdims=True)
        return mixture

    def verify_array_profile(self, prof):
        """Verify that an array profile is valid for game"""
        return ((self.num_role_strats,) == prof.shape and
                np.all(self.aplayers == self.role_reduce(prof)))

    def verify_array_mixture(self, mix):
        """Verify that an array mixture is valid for game"""
        return ((self.num_role_strats,) == mix.shape and
                np.allclose(self.role_reduce(mix), 1))

    def all_profiles(self, as_array=False):
        """Returns a generator over all profiles"""
        role_arrays = [utils.acomb(n_strats, players) for n_strats, players
                       in zip(self.astrategies, self.aplayers)]
        profiles = utils.acartesian2(*role_arrays)
        if as_array or as_array is None:
            return profiles
        else:
            return (self.as_profile(p, verify=False) for p in profiles)

    def as_dict(self, array, data_type=float, dict_type=dict,
                filter_zeros=True):
        """Converts to dictionary profile representation

        Parameters
        ----------
        array : ndarray or Mapping
            The input to convert.
        data_type : type
            The type to convert the data stored for each role and strategy to.
        dict_type : type
            The type of the global returned dictionary. Common types are dict,
            Profile, Mixture.
        filter_zeros : bool
            Whether to remove entries whose value is zero.
        """
        if isinstance(array, dict_type):
            return array
        elif isinstance(array, abc.Mapping):
            return dict_type(array)
        elif isinstance(array, (np.ndarray, abc.Sequence)):
            array = np.asarray(array)
            assert array.shape == (self.num_role_strats,), \
                "Invalid shape for conversion to dict {} instead of {}"\
                .format(array.shape, (self.num_role_strats,))
            return dict_type({role: {strat: data_type(count) for strat, count
                                     in zip(strats, counts)
                                     if not filter_zeros or count > 0}
                              for counts, (role, strats)
                              in zip(self.role_split(array),
                                     self.strategies.items())})
        else:
            raise ValueError("Unknown instance for conversion to dict {}"
                             .format(array))

    def as_array(self, prof, dtype):
        """Converts a dictionary representation into an array representation

        The array representation is an array of roles with shape
        (num_role_strats,), where the mapping is defined by the order in the
        strategies dictionary.

        If an array is passed in, nothing is changed.
        """
        if isinstance(prof, (np.ndarray, abc.Sequence)):  # Already an array
            return np.asarray(prof, dtype=dtype)
        array = np.zeros(self.num_role_strats, dtype=dtype)
        for role, strats in prof.items():
            for strat, value in strats.items():
                array[self._role_strat_index[(role, strat)]] = value
        return array

    def as_profile(self, prof, as_array=False, verify=True):
        """Converts an array profile representation into a dictionary representation

        The dictionary representation is a mapping from roles to strategies to
        counts.

        If a profile is passed in, nothing is changed.
        """
        if as_array is None:
            return prof

        elif as_array:
            prof = self.as_array(prof, int)
            assert not verify or self.verify_array_profile(prof), \
                "input was not a valid profile {}".format(prof)
            return prof

        else:
            prof = self.as_dict(prof, int, profile.Profile, True)
            assert not verify or prof.is_valid(self.players, self.strategies), \
                "invalid profile {}".format(prof)
            return prof

    def as_mixture(self, mix, as_array=False, verify=True):
        """Converts an array profile representation into a dictionary representation

        The dictionary representation is a mapping from roles to strategies to
        counts.

        If a profile is passed in, nothing is changed.
        """
        if as_array is None:
            return mix

        elif as_array:
            mix = self.as_array(mix, float)
            assert not verify or self.verify_array_mixture(mix), \
                "input was not a valid mixture".format(mix)
            return mix

        else:
            mix = self.as_dict(mix, float, profile.Mixture, True)
            assert not verify or mix.is_valid(self.strategies), \
                "invalid mixture {}".format(mix)
            return mix

    def uniform_mixture(self, as_array=False):
        """Returns a uniform mixed profile

        Set as_array to True to return the array representation of the profile.

        """
        mix = 1 / self.astrategies.repeat(self.astrategies)
        return self.as_mixture(mix, as_array=as_array, verify=False)

    def random_mixtures(self, num=1, alpha=1, as_array=False):
        """Return a random mixed profile

        Mixed profiles are sampled from a dirichlet distribution with parameter
        alpha. If alpha = 1 (the default) this is a uniform distribution over
        the simplex for each role. alpha \in (0, 1) is baised towards high
        entropy mixtures, i.e. mixtures where one strategy is played in
        majority. alpha \in (1, oo) is baised towards low entropy (uniform)
        mixtures.

        Set as_array to True to return an array representation of the profile.
        """
        mixtures = np.random.gamma(alpha, 1, (num, self.num_role_strats))
        mixtures /= self.role_reduce(mixtures, axis=1, keepdims=True)
        if as_array or as_array is None:
            return mixtures
        else:
            return (self.as_mixture(m, verify=False) for m in mixtures)

    def biased_mixtures(self, bias=.9, as_array=False):
        """Generates mixtures biased towards one strategy for each role

        Gives a generator of all mixtures of the following form: each role has
        one strategy played with probability bias; the reamaining 1-bias
        probability is distributed uniformly over the remaining S or S-1
        strategies. If there's only one strategy, it is played with probability
        1.
        """
        assert 0 <= bias <= 1, "probabilities must be between zero and one"

        role_mixtures = []
        for num_strats in self.astrategies:
            if num_strats == 1:
                mix = np.ones((1, 1))
            else:
                mix = np.empty((num_strats, num_strats))
                mix.fill((1 - bias)/(num_strats - 1))
                np.fill_diagonal(mix, bias)
            role_mixtures.append(mix)

        mixtures = utils.acartesian2(*role_mixtures)
        if as_array or as_array is None:
            return mixtures
        else:
            return (self.as_mixture(m, verify=False) for m in mixtures)

    def role_biased_mixtures(self, bias=0.9, as_array=False):
        """Generates mixtures where one role-strategy is played with bias

        If no roles have more than one strategy (a degenerate game), then this
        returns nothing.
        """
        assert 0 <= bias <= 1, "probabilities must be between zero and one"

        num = self.astrategies[self.astrategies > 1].sum()
        mixes = self.uniform_mixture(as_array=True)[None].repeat(num, 0)
        prof_offset = 0
        strat_offset = 0
        for num_strats in self.astrategies:
            if num_strats > 1:
                view = mixes[prof_offset:prof_offset+num_strats,
                             strat_offset:strat_offset+num_strats]
                view.fill((1 - bias)/(num_strats - 1))
                np.fill_diagonal(view, bias)
                prof_offset += num_strats
            strat_offset += num_strats

        if as_array or as_array is None:
            return mixes
        else:
            return (self.as_mixture(m, verify=False) for m in mixes)

    def pure_mixtures(self, as_array=False):
        """Returns a generator over all mixtures where the probability of playing a
        strategy is either 1 or 0.

        Set as_array to True to return the mixed profiles in array form.
        """
        return self.biased_mixtures(bias=1, as_array=as_array)

    def grid_mixtures(self, num_points, as_array=False):
        """Returns all of the mixtures in a grid with n points

        Arguments
        ---------
        num_points : int > 1
            The number of points to have along one dimensions
        """
        assert num_points > 1, "Must have at least two points on a dimensions"
        role_mixtures = [utils.acomb(num_strats, num_points - 1) /
                         (num_points - 1)
                         for num_strats in self.astrategies]
        mixtures = utils.acartesian2(*role_mixtures)

        if as_array or as_array is None:
            return mixtures
        else:
            return (self.as_mixture(m, verify=False) for m in mixtures)

    def is_symmetric(self):
        """Returns true if this game is symmetric"""
        return self.astrategies.size == 1

    def is_asymmetric(self):
        """Returns true if this game is asymmetric"""
        return np.all(self.aplayers == 1)

    def _payoff_dict(self, aprofile, apayoffs, conv=float):
        """Merges a value/payoff array and a counts array into a payoff dict"""
        return {role: {strat: conv(payoff) for strat, count, payoff
                       in zip(strats, s_profile, s_payoffs) if count > 0}
                for (role, strats), s_profile, s_payoffs
                in zip(self.strategies.items(),
                       self.role_split(aprofile),
                       self.role_split(apayoffs))}

    def to_json(self):
        """Convert to a json serializable format"""
        return {'players': dict(self.players),
                'strategies': dict(self.strategies)}

    def __repr__(self):
        return '<{}: {}, {}>'.format(
            self.__class__.__name__,
            dict(self.players),
            {r: set(ses) for r, ses in self.strategies.items()})

    def __str__(self):
        return (('{}:\n\t'
                 'Roles: {}\n\t'
                 'Players:\n\t\t{}\n\t'
                 'Strategies:\n\t\t{}\n').format(
                     self.__class__.__name__,
                     ', '.join(self.strategies),
                     '\n\t\t'.join('{:d}x {}'.format(count, role)
                                   for role, count
                                   in sorted(self.players.items())),
                     '\n\t\t'.join('{}:\n\t\t\t{}'.format(
                         role,
                         '\n\t\t\t'.join(strats))
                         for role, strats
                         in sorted(self.strategies.items()))
            )).expandtabs(4)


class Game(EmptyGame):
    """Role-symmetric game representation

    This constructor uses knowledge about the internal representation to
    efficiently generate games. Generally one will want to use one of the
    constructor methods prefixed with "from" e.g. `from_payoff_format` or
    `from_json`.

    Parameters
    ----------
    players : {role: count}
        Number of players per role
    strategies : {role: [strategies]}
        Strategies available for each role. The order of iteration for the
        roles and strategies must align with aprofiles and apayoffs. Since
        dictionaries don't provide consistent iteration order, and OrderedDict,
        or fodict will need to be used.
    aprofiles : ndarray, shape (num_profiles, num_role_strats)
        A description of the profiles. A parallel array with apayoffs.
    apayoffs : ndarray, shape (num_profiles, num_role_strats)
        The payoffs for each role strat per profile. A parallel array with
        aprofiles.
    verify : bool
        If true, this will perform expensive checks to make sure input is
        valid. Note, not everything can be checked, e.g. permutation of
        strategies or roles, and some things that are cheap will always be
        checked.
    """
    def __init__(self, players, strategies, aprofiles, apayoffs, verify=True):
        super().__init__(players, strategies)

        assert aprofiles.shape == apayoffs.shape, \
            "aprofiles and apayoffs must be the same shape : aprofiles {0}, apayoffs {1}"\
            .format(aprofiles.shape, apayoffs.shape)
        expected_shape = (aprofiles.shape[0], self.num_role_strats)
        assert aprofiles.shape == expected_shape, \
            "aprofiles must have proper shape : expected {0} but was {1}"\
            .format(expected_shape, aprofiles.shape)
        assert np.issubdtype(aprofiles.dtype, int), \
            "aprofiles must contain integers : dtype {0}".format(
                aprofiles.dtype)
        if verify:
            assert np.all(aprofiles >= 0), \
                "aprofiles was not non negative {} {}".format(
                    np.any(aprofiles < 0, 1).nonzero(),
                    aprofiles[aprofiles < 0])
            assert np.all(self.role_reduce(aprofiles, axis=1) ==
                          self.aplayers), \
                "not all aprofiles equaled player total {} {}".format(
                    np.any(self.role_reduce(aprofiles, axis=1) ==
                           self.aplayers, 1).nonzero(),
                    aprofiles[np.any(self.role_reduce(aprofiles, axis=1) ==
                                     self.aplayers, 1)])
            assert np.all(apayoffs * (aprofiles == 0) == 0), \
                ("there were nonzero payoffs for strategies without players"
                 " {} {} {}").format(
                     np.any(apayoffs * (aprofiles == 0) != 0, 1).nonzero(),
                     aprofiles[np.any(apayoffs * (aprofiles == 0) != 0, 1)],
                     apayoffs[np.any(apayoffs * (aprofiles == 0) != 0, 1)])

        self._aprofiles = aprofiles
        self._aprofiles.setflags(write=False)
        self._apayoffs = apayoffs
        self._apayoffs.setflags(write=False)
        self._aprofile_map = {collect.frozenarray(aprof): apay for aprof, apay
                              in self.profile_payoffs(as_array=True)}
        self._dev_reps_ = None
        self._writeable_payoffs()  # Reset

        assert len(self._aprofile_map) == len(aprofiles), \
            """There was at least one duplicate profile"""

    @staticmethod
    def from_payoff_format(players, strategies, payoff_data, length=None):
        """Create game from generic payoff format

        Strategies and roles are sorted for their order in the game.

        Parameters
        ----------
        players : {role: count}
            Mapping from roles to number of players per role.
        strategies : {role: {strategy}}
            Mapping from roles to per-role strategy sets.
        payoff_data : ({role: [(strat, count, [payoff])]})
            Collection of data objects mapping roles to collections of
            (strategy, count, value) tuples.
        length : int
            The total number of payoffs. This can make constructing the final
            game more efficient if known, which it usually is. If unknown, and
            payoff_data doesn't have a len, then the payoff data is first
            copied into a list.
        """
        # When loaded in this form we sort roles and strategies
        game = EmptyGame.from_payoff_format(players, strategies)

        if length is None:
            if not isinstance(payoff_data, abc.Sized):
                warnings.warn('Copying profile data, this usually indicates '
                              'something went wrong')
                payoff_data = list(payoff_data)
            length = len(payoff_data)
        aprofiles = np.zeros((length, game.num_role_strats), dtype=int)
        apayoffs = np.zeros((length, game.num_role_strats))

        p = 0
        for profile_data in payoff_data:
            if any(any(p is None or len(p) == 0 for _, _, p in sym_grp)
                   for sym_grp in profile_data.values()):
                prof = profile.Profile.from_input_profile(profile_data)
                warnings.warn('Encountered null payoff data in profile: {0}'
                              .format(prof))
                continue  # Invalid data, but can continue

            for role, strategy_data in profile_data.items():
                for strategy, count, payoffs in strategy_data:
                    i = game._role_strat_index[(role, strategy)]
                    assert aprofiles[p, i] == 0, (
                        'Duplicate role strategy pair ({}, {})'
                        .format(role, strategy))
                    aprofiles[p, i] = count
                    apayoffs[p, i] = np.average(payoffs)

            p += 1  # profile added

        # The slice at the end truncates any null data
        return Game(game.players, game.strategies, aprofiles[:p, :],
                    apayoffs[:p, :])

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        players, strategies, payoff_data, *length = \
            gameio._game_from_json(json_)
        return Game.from_payoff_format(players, strategies, payoff_data,
                                       *length)

    @staticmethod
    def from_matrix(strategies, matrix):
        """Create an asymmetric game from a payoff matrix

        Parameters
        ----------
        strategies : {role: [strategy]}
            The ordered strategies per role as in the standard input format.
        matrix : ndarray, shape (num_strats_1, ..., num_strats_n, n)
            The payoff matrix, the first n dimensions corespond to the
            strategies of the n players, the last is the payoff for each of the
            n players.
        """
        # Verify everything makes sense
        assert all(len(s) == d for s, d
                   in zip(strategies.values(), matrix.shape)), \
            "number of strategies did not match matrix shape"
        assert matrix.shape[-1] == len(matrix.shape) - 1, \
            "matrix shape is inconsistent with a matrix game {}".format(
                matrix.shape)

        aprofiles = utils.acartesian2(*[np.eye(s, dtype=int)
                                        for s in matrix.shape[:-1]])
        apayoffs = np.zeros(aprofiles.shape)
        apayoffs[aprofiles > 0] = matrix.flat

        return Game({r: 1 for r in strategies}, strategies, aprofiles,
                    apayoffs)

    @staticmethod
    def from_game(game):
        if not isinstance(game, EmptyGame):
            raise ValueError("not a valid game")
        elif not isinstance(game, Game):
            return Game(game.players, game.strategies,
                        np.empty((0, game.num_role_strats), dtype=int),
                        np.empty((0, game. num_role_strats)))
        else:
            return Game(game.players, game.strategies,
                        game.profiles(as_array=True),
                        game.payoffs(as_array=True))

    def _writeable_payoffs(self):
        """Get a writable version of the payoff array

        This makes sure that internal bookkeeping is kept up to date
        """
        self._apayoffs.setflags(write=True)
        view = self._apayoffs.view()
        self._apayoffs.setflags(write=False)
        self._min_payoffs = None
        return view

    @property
    def _dev_reps(self):
        """Get the dev reps

        Lazily computed
        """
        if self._dev_reps_ is None:
            player_factorial = self.role_reduce(
                sps.gammaln(self._aprofiles + 1), axis=1)
            totals = np.exp(np.sum(sps.gammaln(self.aplayers + 1) -
                                   player_factorial, 1))
            self._dev_reps_ = (totals[:, None] * self._aprofiles /
                               self.aplayers.repeat(self.astrategies))
            self._dev_reps_.setflags(write=False)
        return self._dev_reps_

    def min_payoffs(self, as_array=False):
        """Returns the minimum payoff for each role"""
        if self._min_payoffs is None:
            if not len(self):
                self._min_payoffs = np.empty(self.astrategies.shape)
                self._min_payoffs.fill(np.nan)
            else:
                masked = np.ma.masked_array(self._apayoffs,
                                            self._aprofiles == 0)
                self._min_payoffs = np.minimum.reduceat(
                    masked.min(0), self._at_indices).filled(np.nan)
            self._min_payoffs.setflags(write=False)
        if as_array or as_array is None:
            return self._min_payoffs.view()
        else:
            return {r: float(m) for r, m
                    in zip(self.strategies.keys(), self._min_payoffs)}

    def profiles(self, as_array=False):
        """Returns all of the profiles with data"""
        if as_array or as_array is None:
            return self._aprofiles.view()
        else:
            return (self.as_profile(prof, verify=False)
                    for prof in self._aprofiles)

    def payoffs(self, as_array=False):
        if as_array or as_array is None:
            return self._apayoffs.view()
        else:
            return (self._payoff_dict(prof, payoffs)
                    for prof, payoffs
                    in zip(self._aprofiles, self._apayoffs))

    def profile_payoffs(self, as_array=False):
        """Returns tuples of profiles and payoffs"""
        iterable = zip(self._aprofiles, self._apayoffs)
        if as_array or as_array is None:
            return iterable
        else:
            return ((self.as_profile(prof, verify=False),
                     self._payoff_dict(prof, payoffs))
                    for prof, payoffs in iterable)

    def get_payoffs(self, profile, as_array=False):
        """Returns a dictionary mapping roles to strategies to payoff"""
        profile = self.as_profile(profile, as_array=True)
        payoffs = self._aprofile_map[collect.frozenarray(profile)]
        if as_array:
            return payoffs
        else:
            return self._payoff_dict(profile, payoffs)

    def get_payoff(self, profile, role, strategy, default=None):
        """Returns the payoff for a specific profile, role, and strategy

        If there's no data for the profile, and a non None default is
        specified, that is returned instead.
        """
        profile = collect.frozenarray(self.as_profile(profile, as_array=True))
        if default is not None and profile not in self:
            return default
        payoffs = self._aprofile_map[profile]
        return payoffs[self._role_strat_index[(role, strategy)]]

    def get_expected_payoff(self, mix, as_array=False):
        """Returns a dict of the expected payoff of a mixed strategy to each role

        If as_array, then an array in role order is returned.
        """
        mix = self.as_mixture(mix, as_array=True)
        deviations = self.deviation_payoffs(mix, as_array=True)
        deviations[mix == 0] = 0  # Don't care about that missing data
        payoff = self.role_reduce(mix * deviations)

        if as_array:
            return payoff
        else:
            return dict(zip(self.strategies, map(float, payoff)))

    def get_max_social_welfare(self, role=None, as_array=False):
        """Returns the maximum social welfare over the known profiles.

        :param role: If specified, get maximum welfare for that role
        :param as_array: If true, the maximum social welfare profile is
            returned in its array representation

        :returns: Maximum social welfare
        :returns: Profile with the maximum social welfare
        """
        # This should probably stay here, because it can't be moved without
        # exposing underlying structure of _counts and _values or making it
        # less efficient. see mixture_deviation_gains

        # If no data, return none
        if not len(self):
            return np.nan, None

        if role is not None:
            i = self._role_index[role]
            start = self._at_indices[i]
            end = start + self.astrategies[i]
            counts = self._aprofiles[:, start:end]
            values = self._apayoffs[:, start:end]
        else:
            counts = self._aprofiles
            values = self._apayoffs

        welfares = np.sum(values * counts, 1)
        profile_index = welfares.argmax()
        profile = self._aprofiles[profile_index]
        if as_array:
            return welfares[profile_index], profile
        else:
            return (float(welfares[profile_index]),
                    self.as_profile(profile, verify=False))

    def deviation_payoffs(self, mix, verify=True, jacobian=False,
                          as_array=False):
        """Computes the expected value of each pure strategy played against all
        opponents playing mix.

        Parameters
        ----------
        mix : ndarray or Mixture
            The mix all other players are using
        jacobian : bool
            If true, the second returned argument will be the jacobian of the
            deviation payoffs with respect to the mixture. The first axis is
            the deviating strategy, the second axis is the strategy in the mix
            the jacobian is taken with respect to. If True, as_array is also
            True.
        as_array : bool
            If True, returns arguments in array form.
        """
        as_array |= jacobian
        mix = self.as_mixture(mix, verify=verify, as_array=True)
        nan_mask = np.empty_like(mix, dtype=bool)

        # Fill out mask where we don't have data
        if self.is_complete():
            nan_mask.fill(False)
        elif len(self) == 0:
            nan_mask.fill(True)
        else:
            support = mix > 0
            devs = self._aprofiles[:, ~support]
            num_needed = EmptyGame(self.players, self.as_mixture(mix)).size

            nan_mask[support] = np.all(devs == 0, 1).sum() < num_needed
            nan_mask[~support] = devs[devs.sum(1) == 1].sum(0) < num_needed

        # Compute values
        if not nan_mask.all():
            with np.errstate(under='ignore'):  # ignore underflow
                # The first use of 'tiny' makes 0^0=1
                # The use of 'tiny' makes 0/0=0.
                prod = np.prod((mix + _TINY) ** self._aprofiles, 1,
                               keepdims=True)
                weights = (prod * self._dev_reps / (mix + _TINY))
                values = np.sum(self._apayoffs * weights, 0)

                if jacobian:
                    devs = np.eye(self.num_role_strats)
                    dev_profs = np.maximum(self._aprofiles[..., None] - devs,
                                           0)
                    dev_jac = np.sum(self._apayoffs[:, None] *
                                     self._dev_reps[:, None] *
                                     dev_profs *
                                     prod[..., None] /
                                     (mix * mix[:, None] + _TINY), 0)

        else:
            values = np.empty(self.num_role_strats)
            if jacobian:
                dev_jac = np.empty((self.num_role_strats,
                                    self.num_role_strats))

        # Fill in nans
        values[nan_mask] = np.nan
        if jacobian:
            dev_jac[nan_mask, nan_mask] = np.nan

        # Output result
        if as_array:
            result = values
        else:
            result = self.as_dict(values, filter_zeros=False)

        if jacobian:
            return result, dev_jac
        else:
            return result

    def is_complete(self):
        """Returns true if every profile has data"""
        return len(self) == self.size

    def is_constant_sum(self):
        """Returns true if this game is constant sum"""
        profile_sums = np.sum(self._aprofiles * self._apayoffs, 1)
        return np.allclose(profile_sums, profile_sums[0])

    def __contains__(self, profile):
        """Returns true if data for that profile exists"""
        hprof = collect.frozenarray(self.as_profile(profile, as_array=True))
        return hprof in self._aprofile_map

    def __iter__(self):
        """Iterator over data_profiles"""
        return (self.as_profile(prof, verify=False)
                for prof in self._aprofiles)

    def __getitem__(self, profile):
        """Identical to get payoffs, makes game behave like a dictionary of profiles to
        payoffs

        """
        return self.get_payoffs(profile)

    def __len__(self):
        """Number of profiles"""
        return len(self._aprofiles)

    def __repr__(self):
        return '{old}, {data:d} / {total:d}>'.format(
            old=super().__repr__()[:-1],
            data=len(self),
            total=self.size)

    def __str__(self):
        return ('{old}payoff data for {data:d} out of {total:d} '
                'profiles').format(
                    old=super().__str__(),
                    data=len(self),
                    total=self.size)

    def to_json(self):
        """Convert to json according to the egta-online v3 default game spec"""
        json = super().to_json()
        json['profiles'] = [prof.to_input_profile(payoffs)
                            for prof, payoffs in self.profile_payoffs()]
        return json


class SampleGame(Game):
    """A Role Symmetric Game that has multiple samples per observation

    This behaves the same as a normal Game object, except that it has a
    `resample` method, which will resample the used payoffs from the empirical
    distribution of payoffs, allowing bootstrapping over arbitrary statistics.
    """

    def __init__(self, players, strategies, aprofiles, sample_apayoffs,
                 verify=True):
        # In case an empty list is passed
        if sample_apayoffs:
            apayoffs = np.vstack([s.mean(2) for s in sample_apayoffs])
        else:
            apayoffs = np.empty((0,) + aprofiles.shape[1:])

        super().__init__(players, strategies, aprofiles, apayoffs, verify)

        self._sample_apayoffs = sample_apayoffs
        for sapay in sample_apayoffs:
            sapay.setflags(write=False)

        assert len(set(x.shape[1] for x in sample_apayoffs)) <= 1, \
            "Not all sample payoffs shapes compatible"

        lens = np.fromiter(itertools.chain([0], (x.shape[0] for x
                                                 in sample_apayoffs)),
                           int, len(sample_apayoffs) + 1).cumsum()

        if len(sample_apayoffs) == 1:
            assert (sample_apayoffs[0] * (aprofiles[..., None] == 0) == 0).all(), \
                "some sample payoffs were nonzero for invalid payoffs"
        elif len(sample_apayoffs) > 1:
            assert all((samp * (count[..., None] == 0) == 0).all()
                       for count, samp
                       in zip(np.split(aprofiles, lens[1:-1]), sample_apayoffs)), \
                "some sample payoffs were nonzero for invalid payoffs"

        # Lazily computed
        self._sample_aprofile_map_ = None

    @staticmethod
    def from_payoff_format(players, strategies, payoff_data, length=None):
        """Create a sample game from the standard payoff format"""
        # When loaded in this form we sort roles and strategies
        game = EmptyGame.from_payoff_format(players, strategies)

        # {sample_count : ([counts], [sample_values])}
        sample_map = {}

        for profile_data in payoff_data:
            if any(any(p is None or len(p) == 0 for _, __, p in sym_grps)
                   for sym_grps in profile_data.values()):
                prof = profile.Profile.from_input_profile(profile_data)
                warnings.warn('Encountered null payoff data in profile: {0}'
                              .format(prof))
                continue  # Invalid data, but can continue

            num_samples = min(min(len(payoffs) for _, __, payoffs in sym_grps)
                              for sym_grps in profile_data.values())
            aprofile = np.zeros(game.num_role_strats, dtype=int)
            apayoffs = np.zeros((game.num_role_strats, num_samples))

            for role, strategy_data in profile_data.items():
                for strategy, count, payoffs in strategy_data:
                    i = game._role_strat_index[(role, strategy)]
                    assert aprofile[i] == 0, (
                        'Duplicate role strategy pair ({}, {})'
                        .format(role, strategy))
                    if len(payoffs) > num_samples:
                        warnings.warn("Truncating observation data")

                    aprofile[i] = count
                    apayoffs[i] = np.average(payoffs)

            lst_profs, lst_pays = sample_map.setdefault(num_samples, ([], []))
            lst_profs.append(aprofile[None])
            lst_pays.append(apayoffs[None])

        # Join data together
        if sample_map:
            aprofiles = np.vstack(itertools.chain.from_iterable(
                x[0] for x in sample_map.values()))
            sample_apayoffs = [np.vstack(x[1]) for x in sample_map.values()]
        else:  # No data
            aprofiles = np.empty((0, game.num_role_strats), dtype=int)
            sample_apayoffs = []

        return SampleGame(game.players, game.strategies, aprofiles,
                          sample_apayoffs)

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        players, strategies, payoff_data, *length = \
            gameio._game_from_json(json_)
        return SampleGame.from_payoff_format(players, strategies, payoff_data,
                                             *length)

    @staticmethod
    def from_matrix(strategies, matrix):
        """Create a sample game from a payoff matrix

        The payoff matrix has one dimensions for each player, one dimension for
        the payoffs, and one dimensions for the samples.
        """
        # Verify everything makes sense
        assert all(len(s) == d for s, d
                   in zip(strategies.values(), matrix.shape)), \
            "number of strategies did not match matrix shape"
        assert matrix.shape[-2] == len(matrix.shape) - 2, \
            "matrix shape is inconsistent with a matrix game"

        num_samples = matrix.shape[-1]
        aprofiles = utils.acartesian2(*[np.eye(s, dtype=int)
                                        for s in matrix.shape[:-2]])
        apayoffs = np.zeros(aprofiles.shape + (num_samples,))
        # This next set of steps is a hacky way of avoiding duplicating mask by
        # num_samples
        pview = apayoffs.view()
        pview.shape = (-1, num_samples)
        mask = aprofiles > 0
        mask.shape = (-1, 1)
        mask = np.broadcast_to(mask, (mask.size, num_samples))
        np.place(pview, mask, matrix.flat)

        return SampleGame({r: 1 for r in strategies}, strategies, aprofiles,
                          [apayoffs])

    @property
    def _sample_aprofile_map(self):
        """Get the dev reps

        Lazily computed
        """
        if self._sample_aprofile_map_ is None:
            self._sample_aprofile_map_ = (
                {collect.frozenarray(aprof): apay
                 for aprof, apay
                 in self.sample_profile_payoffs(True)})
        return self._sample_aprofile_map_

    def num_samples(self):
        """Get the different sample numbers"""
        if len(self) > 0:
            return {v.shape[2] for v in self._sample_apayoffs}
        else:
            return {0}

    def get_sample_payoffs(self, profile, as_array=False):
        """Returns a dictionary mapping roles to strategies to payoffs"""
        profile = self.as_profile(profile, as_array=True)
        payoffs = self._sample_aprofile_map[collect.frozenarray(profile)]
        if as_array:
            return payoffs
        return self._payoff_dict(profile, payoffs,
                                 lambda l: list(map(float, l)))

    def sample_profile_payoffs(self, as_array=False):
        """Returns a generator of tuples of (profile, sample_payoffs)

        If as_array is True, they are given in their array representation
        """
        iterable = zip(self._aprofiles,
                       itertools.chain.from_iterable(self._sample_apayoffs))
        if as_array or as_array is None:
            return iterable
        else:
            return ((self.as_profile(counts, verify=False),
                     self._payoff_dict(counts, payoffs,
                                       lambda l: list(map(float, l))))
                    for counts, payoffs in iterable)

    def remean(self):
        """Overwrite payoff values with mean payoff"""
        begin = 0
        payoffs = self._writeable_payoffs()
        for obs in self._sample_apayoffs:
            end = begin + obs.shape[0]
            payoffs[begin:end] = obs.mean(2)
            begin = end

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
        begin = 0
        payoffs = self._writeable_payoffs()
        for obs in self._sample_apayoffs:
            num_samples = obs.shape[2]
            num_obs_resamples = (num_samples if num_resamples is None
                                 else num_resamples)
            end = begin + obs.shape[0]
            shape = [dim if switch else 1
                     for dim, switch in zip(obs.shape, switches)]
            sample = np.random.multinomial(
                num_obs_resamples, [1/num_samples]*num_samples, shape)
            payoffs[begin:end] = ((obs * sample).mean(3) *
                                  (num_samples / num_obs_resamples))
            end = begin

    def __repr__(self):
        samples = self.num_samples()
        if len(samples) == 1:
            return '{old}, {samples:d}>'.format(
                old=super().__repr__()[:-1],
                samples=next(iter(samples)))
        else:
            return '{old}, {min_samples:d} - {max_samples:d}>'.format(
                old=super().__repr__()[:-1],
                min_samples=min(samples),
                max_samples=max(samples))

    def __str__(self):
        samples = self.num_samples()
        if len(samples) == 1:
            return '{old}\n{samples:d} observations per profile'.format(
                old=super().__str__(),
                samples=next(iter(samples)))
        else:
            return ('{old}\n{min_samples:d} to {max_samples:d} observations '
                    'per profile').format(
                        old=super().__str__(),
                        min_samples=min(samples),
                        max_samples=max(samples))

    def to_json(self):
        """Convert to json according to the egta-online v3 default game spec"""
        json = EmptyGame.to_json(self)
        json['profiles'] = [prof.to_input_profile(payoffs)
                            for prof, payoffs in self.sample_profile_payoffs()]
        return json
