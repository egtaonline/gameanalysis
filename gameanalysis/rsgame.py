"""Module for Role Symmetric Game data structures"""
import collections
import itertools
import math
import operator
import warnings
from collections import abc

import numpy as np
import scipy.misc as spm

from gameanalysis import collect
from gameanalysis import gameio
from gameanalysis import profile
from gameanalysis import utils


_exact_factorial = np.vectorize(math.factorial, otypes=[object])
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

    Members
    -------
    size : int
        The total number of profiles possible in this game
    players : {role: count}
        An immutable copy of the input players.
    strategies : {role: {strategy}}
        An immutable copy of the input strategies. The iteration order of
        roles, and strategies per role are the cannon iteration order for this
        Game, and the order the strategies and roles will be mapped in the
        array representation.
    """
    def __init__(self, players, strategies):
        assert frozenset(players.keys()) == \
            frozenset(strategies.keys()), \
            "Player roles and strategy roles do not agree"
        assert all(len(strats) == len(frozenset(strats)) for strats
                   in strategies.values()), \
            "Not all strategies are unique: {0}".format(strategies)

        self.strategies = collect.fodict((r, tuple(s))
                                         for r, s in strategies.items())
        # Ensures players and strategies are in the same order
        self.players = collect.fodict((r, players[r]) for r
                                      in self.strategies)

        self.size = utils.prod(utils.game_size(self.players[r], len(strats))
                               for r, strats in self.strategies.items())

        max_strategies = max([len(s) for s in self.strategies.values()])
        # self._mask specifies the valid strategy positions
        self._mask = np.zeros((len(self.strategies), max_strategies),
                              dtype=bool)
        for r, strats in enumerate(self.strategies.values()):
            self._mask[r, :len(strats)] = True

        self._role_index = {r: i for i, r in enumerate(self.strategies.keys())}
        self._strategy_index = {r: {s: i for i, s in enumerate(strats)}
                                for r, strats in self.strategies.items()}

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        players, strategies, *_ = gameio._game_from_json(json_)
        return EmptyGame(players, strategies)

    def all_profiles(self):
        """Returns a generator over all profiles"""
        return map(profile.Profile, itertools.product(*(
            [(role, collections.Counter(comb)) for comb
             in itertools.combinations_with_replacement(
                 strats, self.players[role])]
            for role, strats in self.strategies.items())))

    def _as_dict(self, array, dict_type, data_type):
        """Converts an array profile representation to a dictionary
        representation"""
        if isinstance(array, dict_type):
            return array
        elif isinstance(array, abc.Mapping):
            return dict_type(array)
        else:
            return dict_type({role: {strat: data_type(count) for strat, count
                                     in zip(strats, counts) if count > 0}
                              for counts, (role, strats)
                              in zip(array, self.strategies.items())})

    def as_profile(self, array):
        """Converts an array profile representation into a dictionary representation

        The dictionary representation is a mapping from roles to strategies to
        counts.

        If a profile is passed in, nothing is changed.
        """
        # TODO Assert profile is valid for this game?
        return self._as_dict(array, profile.Profile, int)

    def as_mixture(self, array):
        """Converts an array profile representation into a dictionary representation

        The dictionary representation is a mapping from roles to strategies to
        counts.

        If a profile is passed in, nothing is changed.
        """
        # TODO Assert valid mixture? Maybe in Mixture?
        return self._as_dict(array, profile.Mixture, float)

    def as_array(self, prof, dtype=float):
        """Converts a dictionary profile representation into an array representation

        The array representation is a matrix roles x max_strategies where the
        mapping is defined by the order in the strategies dictionary.

        If an array is passed in, nothing is changed.

        By definition, an invalid entries are zero.

        """
        # TODO Check that profile is a valid profile
        if isinstance(prof, np.ndarray):  # Already an array
            return np.asarray(prof, dtype=dtype)
        array = np.zeros_like(self._mask, dtype=dtype)
        for r, (role, strats) in enumerate(self.strategies.items()):
            for s, strategy in enumerate(strats):
                if strategy in prof[role]:
                    array[r, s] = prof[role][strategy]
        return array

    def uniform_mixture(self, as_array=False):
        """Returns a uniform mixed profile

        Set as_array to True to return the array representation of the profile.

        """
        mix = self._mask / self._mask.sum(1)[:, np.newaxis]
        if as_array:
            return mix
        else:
            return self.as_mixture(mix)

    def random_mixture(self, alpha=1, as_array=False):
        """Return a random mixed profile

        Mixed profiles are sampled from a dirichlet distribution with parameter
        alpha. If alpha = 1 (the default) this is a uniform distribution over
        the simplex for each role. alpha \in (0, 1) is baised towards high
        entropy mixtures, i.e. mixtures where one strategy is played in
        majority. alpha \in (1, oo) is baised towards low entropy (uniform)
        mixtures.

        Set as_array to True to return an array representation of the profile.
        """
        mix = np.random.dirichlet([alpha] * self._mask.shape[1],
                                  self._mask.shape[0]) * self._mask
        mix /= mix.sum(1)[:, np.newaxis]
        if as_array:
            return mix
        else:
            return self.as_mixture(mix)

    def biased_mixtures(self, bias=.9, as_array=False):
        """Generates mixtures for initializing replicator dynamics.

        Gives a generator of all mixtures of the following form: each role has
        one or zero strategies played with probability bias; the reamaining
        1-bias probability is distributed uniformly over the remaining S or S-1
        strategies.
        """
        assert 0 <= bias <= 1, 'probabilities must be between zero and one'
        num_strategies = self._mask.sum(1)

        def possible_strats(num_strat):
            """Returns a generator of all possible biased strategy indices"""
            if num_strat == 1:
                return [None]
            else:
                return itertools.chain([None], range(num_strat))

        for strats in itertools.product(*map(possible_strats, num_strategies)):
            mix = np.array(self._mask, dtype=float)
            for r in range(len(self.strategies)):
                s = strats[r]
                ns = num_strategies[r]
                if s is None:  # uniform
                    mix[r] /= ns
                else:  # biased
                    mix[r, :ns] -= bias
                    mix[r, :ns] /= (ns-1)
                    mix[r, s] = bias
            if as_array:
                yield mix
            else:
                yield self.as_mixture(mix)

    def pure_mixtures(self, as_array=False):
        """Returns a generator over all mixtures where the probability of playing a
        strategy is either 1 or 0

        Set as_array to True to return the mixed profiles in array form.
        """
        mixtures = (profile.Mixture(rs) for rs in itertools.product(
            *([(r, {s: 1}) for s in sorted(ss)] for r, ss
              in self.strategies.items())))
        if as_array:
            return (self.as_array(mix) for mix in mixtures)
        else:
            return mixtures

    def is_symmetric(self):
        """Returns true if this game is symmetric"""
        return len(self.strategies) == 1

    def is_asymmetric(self):
        """Returns true if this game is asymmetric"""
        return all(p == 1 for p in self.players.values())

    def _payoff_dict(self, counts, values, conv=float):
        """Merges a value/payoff array and a counts array into a payoff dict"""
        return {role: {strat: conv(payoff) for strat, count, payoff
                       in zip(strats, s_count, s_value) if count > 0}
                for (role, strats), s_count, s_value
                in zip(self.strategies.items(), counts, values)}

    def to_json(self):
        """Convert to a json serializable format"""
        return {'players': dict(self.players),
                'strategies': dict(self.strategies.items())}

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
                     ', '.join(sorted(self.strategies)),
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

    Members
    -------
    same as EmptyGame
    """
    # There are several private members of a game.
    # _profile_map : maps static profiles to their index in the values array
    # _values : An array of mean payoff data indexed by [profile, role,
    #           strategy] all as indices

    # TODO have escape for error checking
    # TODO better messages for asserts
    def __init__(self, players, strategies, counts, values):
        super().__init__(players, strategies)

        assert counts.shape[1:] == self._mask.shape, \
            "counts must have proper shape : expected {1} but was {0}".format(
                counts.shape[1:], self._mask.shape)
        assert values.shape[1:] == self._mask.shape, \
            "values must have proper shape : expected {0} but was {1}".format(
                self._mask.shape, values.shape[1:])
        assert counts.shape[0] == values.shape[0], \
            "counts and values must match in dim 0 : counts {0}, values {1}"\
            .format(counts.shape, values.shape)
        assert np.issubdtype(counts.dtype, int), \
            "counts must contain integers : dtype {0}".format(counts.dtype)
        assert (counts >= 0).all(), \
            "counts was not non negative"
        role_count = np.fromiter(self.players.values(), int, len(self.players))
        assert (counts.sum(2) == role_count).all(), \
            "not all counts equaled player total"
        assert (counts * ~self._mask == 0).all(), \
            "some invalid counts were nonzero"
        assert (values * ~self._mask == 0).all(), \
            "some invalid values were nonzero"

        self._counts = counts
        self._values = values
        self._profile_map = {self.as_profile(c): i for i, c
                             in enumerate(counts)}

        assert len(self._profile_map) == counts.shape[0], \
            """There was at least one duplicate profile"""

        self._compute_dev_reps()
        self._compute_min_payoffs()

    @staticmethod
    def from_payoff_format(players, strategies, payoff_data, length=None):
        """Create game from generic payoff format

        Strategies and roles are sorted

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
            game more efficient if known, which it usually is.
        """
        # When loaded in this form we sort roles and strategies
        sorted_strategies = collections.OrderedDict(sorted(
            (r, sorted(s)) for r, s in strategies.items()))
        game = EmptyGame(players, sorted_strategies)

        if length is None:
            if not isinstance(payoff_data, abc.Sized):
                warnings.warn('Copying profile data, this usually indicates '
                              'something went wrong')
                payoff_data = list(payoff_data)
            length = len(payoff_data)
        counts = np.zeros((length,) + game._mask.shape, dtype=int)
        values = np.zeros((length,) + game._mask.shape)

        p = 0
        for profile_data in payoff_data:
            prof = profile.Profile.from_input_profile(profile_data)
            if any(any(p is None or len(p) == 0 for _, _, p in dat)
                   for dat in profile_data.values()):
                warnings.warn('Encountered null payoff data in profile: {0}'
                              .format(prof))
                continue  # Invalid data, but can continue

            for r, role in enumerate(game.strategies):
                for strategy, count, payoffs in profile_data[role]:
                    s = game._strategy_index[role][strategy]
                    assert counts[p, r, s] == 0, (
                        'Duplicate role strategy pair ({}, {})'
                        .format(role, strategy))
                    values[p, r, s] = np.average(payoffs)
                    counts[p, r, s] = count

            p += 1  # profile added

        # The slice at the end truncates any null data
        return Game(game.players, game.strategies, counts[:p, ...],
                    values[:p, ...])

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        players, strategies, payoff_data, *length = \
            gameio._game_from_json(json_)
        return Game.from_payoff_format(players, strategies, payoff_data,
                                       *length)

    @staticmethod
    def from_matrix(strategies, matrix):
        """FIXME"""
        # Verify everything makes sense
        assert all(len(s) == d for s, d
                   in zip(strategies.values(), matrix.shape)), \
            "number of strategies did not match matrix shape"
        assert matrix.shape[-1] == len(matrix.shape) - 1, \
            "matrix shape is inconsistent with a matrix game"

        # Calculate necessary information
        num_roles = matrix.shape[-1]
        max_strat = max(matrix.shape[:-1])
        mask_size = max_strat * num_roles
        num_profs = utils.prod(matrix.shape[:-1])

        # These are the indices of non zero payoffs
        indices = np.fromiter(
            itertools.chain.from_iterable(
                (p * mask_size + r * max_strat + s for r, s in enumerate(sts))
                for p, sts in enumerate(itertools.product(
                    *map(range, matrix.shape[:-1])))),
            int, num_profs * num_roles)

        # Use indices to create counts and values
        counts = np.zeros([num_profs, num_roles, max_strat], dtype=int)
        values = np.zeros([num_profs, num_roles, max_strat])
        counts.ravel()[indices] = 1
        values.ravel()[indices] = matrix.ravel()

        return Game({r: 1 for r in strategies}, strategies, counts, values)

    def _compute_dev_reps(self, dtype=float, factorial=spm.factorial,
                          div=operator.truediv):
        """Precompute number of deviations?

        Uses fast floating point math or at least vectorized computation to
        compute devreps"""
        with warnings.catch_warnings():
            if dtype == float:  # approximate setting
                warnings.filterwarnings('error')  # Raise exception on warnings

            try:  # Use approximate unless it overflows
                strat_counts = np.empty(len(self.players), dtype=dtype)
                for i, c in enumerate(self.players.values()):
                    strat_counts[i] = c
                player_factorial = factorial(self._counts).prod(2)
                totals = np.prod(div(factorial(strat_counts),
                                     player_factorial), axis=1)
                self._dev_reps = div(
                    totals[:, np.newaxis, np.newaxis] *
                    self._counts, strat_counts[:, np.newaxis]).astype(float)

            except Exception as e:
                # Tweaks computation to be exact
                if dtype == float:
                    self._compute_dev_reps(object, _exact_factorial,
                                           operator.floordiv)
                else:
                    raise e

    def _compute_min_payoffs(self):
        """Assigns _min_payoffs to the minimum payoff for every role"""
        if (not self._values.size):
            self._min_payoffs = np.empty(self._mask.shape[0])
            self._min_payoffs.fill(np.nan)
        else:
            self._min_payoffs = (np.ma.masked_array(self._values,
                                                    self._counts == 0)
                                 .min((0, 2)).filled(np.nan))

    def min_payoffs(self, as_array=False):
        """Returns the minimum payoff for each role"""
        if as_array:
            view = self._min_payoffs.view()
            view.setflags(write=False)
            return view
        else:
            return {r: float(m) for r, m
                    in zip(self.strategies.keys(), self._min_payoffs)}

    def data_profiles(self):
        """Returns an iterator over all profiles with data

        Note: this returns profiles in a different order than payoffs
        """
        return self._profile_map.keys()

    def get_payoff(self, profile, role, strategy, default=None):
        """Returns the payoff for a specific profile, role, and strategy

        If there's no data for the profile, and a non None default is
        specified, that is returned instead.
        """
        profile = self.as_profile(profile)
        if default is not None and profile not in self:
            return default
        p = self._profile_map[profile]
        r = self._role_index[role]
        s = self._strategy_index[role][strategy]
        return self._values[p, r, s]

    def get_payoffs(self, profile, as_array=False):
        """Returns a dictionary mapping roles to strategies to payoff"""
        index = self._profile_map[self.as_profile(profile)]
        payoffs = self._values[index]
        if as_array:
            return payoffs
        else:
            return self._payoff_dict(self._counts[index], payoffs)

    def payoffs(self, as_array=False):
        """Returns an iterable of tuples of (profile, payoffs)

        If as_array is True, they are given in their array representation

        """
        iterable = zip(self._counts, self._values)
        if as_array:
            return iterable
        else:
            return ((self.as_profile(counts),
                     self._payoff_dict(counts, payoffs))
                    for counts, payoffs in iterable)

    def get_expected_payoff(self, mix, as_array=False):
        """Returns a dict of the expected payoff of a mixed strategy to each role

        If as_array, then an array in role order is returned.
        """
        mix = self.as_array(mix)
        payoff = (mix * self.expected_values(mix, as_array=True)).sum(1)
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
        if not self._values.size:
            return np.nan, None

        if role is not None:
            role_index = self._role_index[role]
            counts = self._counts[:, role_index][..., np.newaxis]
            values = self._values[:, role_index][..., np.newaxis]
        else:
            counts = self._counts
            values = self._values

        welfares = np.sum(values * counts, (1, 2))
        profile_index = welfares.argmax()
        profile = self._counts[profile_index]
        if as_array:
            return welfares[profile_index], profile
        else:
            return float(welfares[profile_index]), self.as_profile(profile)

    def expected_values(self, mix, as_array=False):
        """Computes the expected value of each pure strategy played against all
        opponents playing mix.
        """
        # FIXME returns a number for deviations with unknown data, this should
        # probably be repalced with NaN but that will have repercussions

        # The first use of 'tiny' makes 0^0=1.
        # The second use of 'tiny' makes 0/0=0.

        # Determine if we have all data
        mix = self.as_array(mix)
        num_profiles = (~(self._counts * (mix == 0)).any((1, 2))).sum()
        total_profiles = EmptyGame(self.players, self.as_mixture(mix)).size

        if num_profiles == total_profiles:
            # FIXME put this in with block
            old = np.seterr(under='ignore')  # ignore underflow
            weights = (((mix + _TINY) ** self._counts)
                       .prod((1, 2))[:, None, None]
                       * self._dev_reps / (mix + _TINY))
            values = np.sum(self._values * weights, 0)
            np.seterr(**old)  # Go back to old settings
        else:
            values = np.empty_like(self._mask, dtype=float)
            values.fill(np.nan)
        if as_array:
            return values
        else:
            return {role: {strat: float(value) for strat, value
                           in zip(strats, s_values)}
                    for (role, strats), s_values
                    in zip(self.strategies.items(), values)}

    def is_complete(self):
        """Returns true if every profile has data"""
        return len(self._profile_map) == self.size

    def is_constant_sum(self):
        """Returns true if this game is constant sum"""
        profile_sums = np.sum(self._counts * self._values, (1, 2))
        return np.allclose(profile_sums, np.mean(profile_sums))

    def __contains__(self, profile):
        """Returns true if data for that profile exists"""
        return profile in self._profile_map

    def __iter__(self):
        """Iterator over data_profiles"""
        return iter(self.data_profiles())

    def __getitem__(self, profile):
        """Identical to get payoffs, makes game behave like a dictionary of profiles to
        payoffs

        """
        return self.get_payoffs(profile)

    def __len__(self):
        """Number of profiles"""
        return len(self._profile_map)

    def __repr__(self):
        return '{old}, {data:d} / {total:d}>'.format(
            old=super().__repr__()[:-1],
            data=len(self._profile_map),
            total=self.size)

    def __str__(self):
        return ('{old}payoff data for {data:d} out of {total:d} '
                'profiles').format(
                    old=super().__str__(), data=len(self._profile_map),
                    total=self.size)

    def to_json(self):
        """Convert to json according to the egta-online v3 default game spec"""
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies.items()},
                'profiles': [prof.to_input_profile(payoffs)
                             for prof, payoffs in self.payoffs()]}


# Make sample game take similar arguments to Game
class SampleGame(Game):
    """A Role Symmetric Game that has multiple samples per observation

    TODO
    """

    def __init__(self, players, strategies, counts, sample_values):
        if sample_values:
            values = np.concatenate([x.mean(3) for x in sample_values])
        else:
            values = np.empty((0,) + counts.shape[1:])
        super().__init__(players, strategies, counts, values)

        self._sample_values = sample_values

        assert len(set(x.shape[1:3] for x in sample_values)) <= 1, \
            "Not all sample values shapes compatible"

        lens = np.fromiter(itertools.chain([0], (x.shape[0] for x
                                                 in sample_values)),
                           int, len(sample_values) + 1).cumsum()

        if len(sample_values) == 1:
            assert (sample_values[0] * (counts[..., None] == 0) == 0).all(), \
                "some sample payoffs were nonzero for invalid payoffs"
        elif len(sample_values) > 1:
            assert all((samp * (count[..., None] == 0) == 0).all()
                       for count, samp
                       in zip(np.split(counts, lens[1:-1]), sample_values)), \
                "some sample payoffs were nonzero for invalid payoffs"

        buckets = ((prof, ind, np.searchsorted(lens, ind, side='right') - 1)
                   for prof, ind in self._profile_map.items())
        self._sample_profile_map = {prof: (bucket, index - lens[bucket])
                                    for prof, index, bucket in buckets}

    @staticmethod
    def from_payoff_format(players, strategies, payoff_data, length=None):
        """TODO"""
        # When loaded in this form we sort roles and strategies
        sorted_strategies = collections.OrderedDict(sorted(
            (r, sorted(s)) for r, s in strategies.items()))
        game = EmptyGame(players, sorted_strategies)

        # {sample_count : ([counts], [sample_values])}
        sample_map = {}

        for profile_data in payoff_data:
            num_samples = min(min(len(payoffs) for _, __, payoffs in sym_grps)
                              for sym_grps in profile_data.values())
            counts = np.zeros((1,) + game._mask.shape, dtype=int)
            values = np.zeros((1,) + game._mask.shape + (num_samples,))

            for r, role in enumerate(game.strategies):
                for strategy, count, payoffs in profile_data[role]:
                    if len(payoffs) > num_samples:
                        warnings.warn("Truncating observation data")
                    s = game._strategy_index[role][strategy]
                    counts[0, r, s] = count
                    values[0, r, s] = payoffs[:num_samples]

            count_list, value_list = sample_map.setdefault(num_samples,
                                                           ([], []))
            count_list.append(counts)
            value_list.append(values)

        # Join data together
        if sample_map:
            counts = np.vstack(itertools.chain.from_iterable(
                x[0] for x in sample_map.values()))
            sample_values = [np.vstack(x[1]) for x in sample_map.values()]
        else:  # No data
            counts = np.empty((0,) + game._mask.shape, dtype=int)
            sample_values = np.empty((0,) + game._mask.shape + (0,))

        return SampleGame(game.players, game.strategies, counts, sample_values)

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        players, strategies, payoff_data, *length = \
            gameio._game_from_json(json_)
        return SampleGame.from_payoff_format(players, strategies, payoff_data,
                                             *length)

    @staticmethod
    def from_matrix(strategies, matrix):
        """FIXME"""
        # Verify everything makes sense
        assert all(len(s) == d for s, d
                   in zip(strategies.values(), matrix.shape)), \
            "number of strategies did not match matrix shape"
        assert matrix.shape[-2] == len(matrix.shape) - 2, \
            "matrix shape is inconsistent with a matrix game"

        # Calculate necessary information
        num_samples = matrix.shape[-1]
        num_roles = matrix.shape[-2]
        strat_counts = matrix.shape[:-2]
        max_strat = max(strat_counts)
        mask_size = max_strat * num_roles
        num_profs = utils.prod(strat_counts)

        # These are the indices of non zero payoffs
        indices = np.fromiter(
            itertools.chain.from_iterable(
                (p * mask_size + r * max_strat + s for r, s in enumerate(sts))
                for p, sts in enumerate(itertools.product(
                    *map(range, strat_counts)))),
            int, num_profs * num_roles)

        # Use indices to create counts and values
        counts = np.zeros([num_profs, num_roles, max_strat], dtype=int)
        sample_vals = np.zeros([num_profs, num_roles, max_strat, num_samples])
        counts.ravel()[indices] = 1
        sample_vals.reshape([-1, num_samples])[indices] = \
            matrix.reshape([-1, num_samples])

        return SampleGame({r: 1 for r in strategies}, strategies, counts,
                          [sample_vals])

    def num_samples(self):
        """Get the different sample numbers"""
        if len(self) > 0:
            return {v.shape[3] for v in self._sample_values}
        else:
            return {0}

    def get_sample_payoffs(self, profile, as_array=False):
        """Returns a dictionary mapping roles to strategies to payoffs"""
        prof = self.as_profile(profile)
        bucket, index = self._sample_profile_map[prof]
        payoffs = self._sample_values[bucket][index]
        if as_array:
            return payoffs
        counts = self._counts[self._profile_map[prof]]
        return self._payoff_dict(counts, payoffs,
                                 lambda l: list(map(float, l)))

    def sample_payoffs(self, as_array=False):
        """Returns a generator of tuples of (profile, sample_payoffs)

        If as_array is True, they are given in their array representation
        """
        iterable = zip(self._counts,
                       itertools.chain.from_iterable(self._sample_values))
        if as_array:
            return iterable
        else:
            return ((self.as_profile(counts),
                     self._payoff_dict(counts, payoffs,
                                       lambda l: list(map(float, l))))
                    for counts, payoffs in iterable)

    def remean(self):
        """Overwrite payoff values with mean payoff"""
        offset = 0
        for obs in self._sample_values:
            num_profiles = obs.shape[0]
            self._values[offset:offset + num_profiles] = obs.mean(3)
            offset += num_profiles

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
        offset = 0
        for obs in self._sample_values:
            num_samples = obs.shape[3]
            num_obs_resamples = (num_samples if num_resamples is None
                                 else num_resamples)
            num_profiles = obs.shape[0]
            shape = [dim if switch else 1
                     for dim, switch in zip(obs.shape, switches)]
            sample = np.random.multinomial(
                num_obs_resamples, [1/num_samples]*num_samples, shape)
            self._values[offset:offset + num_profiles] = \
                (obs * sample).mean(3) * (num_samples / num_obs_resamples)
            offset += num_profiles

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
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies.items()},
                'profiles': [prof.to_input_profile(payoffs)
                             for prof, payoffs in self.sample_payoffs()]}
