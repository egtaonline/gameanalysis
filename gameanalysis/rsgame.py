"""This module contains data structures and accompanying methods for working
with role symmetric games"""
import itertools
import math
import collections
import warnings

import numpy as np
import scipy.misc as spm

from gameanalysis import gameio
from gameanalysis import profile
from gameanalysis import utils
from gameanalysis import collect


# Raise an error on any funny business
np.seterr(over='raise')
_exact_factorial = np.vectorize(math.factorial, otypes=[object])
_TINY = np.finfo(float).tiny

# TODO remove reliance on underlying array data structures, and provide array
# access to appropriate efficient parts.

# TODO allow all profile / mix functions to take either style of profile, and
# convert to the appropriate one.


class EmptyGame(object):
    """Role-symmetric game representation

    This object only contains methods and information about definition of the
    game, and does not contain methods to operate on observation data.

    Parameters
    ----------
    players : {role: count}
        Mapping from roles to number of players per role.
    strategies : {role: {strategy}}
        Mapping from roles to per-role strategy sets.

    Members
    -------
    players : {role: count}
        An immutable copy of the input players.
    strategies : {role: {strategy}}
        An immutable copy of the input strategies. The iteration order of
        roles, and strategies per role are the cannon iteration order for this
        Game, and the order the strategies and roles will be mapped in the
        array representation. Note that this can be different than the order of
        the object that is passed in.
    """
    def __init__(self, players, strategies, _=None, __=None):
        self.players = collect.frozendict(players)
        self.strategies = collect.frozendict((r, frozenset(s))
                                             for r, s in strategies.items())

        self._size = utils.prod(utils.game_size(self.players[r], len(strats))
                                for r, strats in self.strategies.items())
        # self._mask specifies the valid strategy positions
        max_strategies = max([len(s) for s in self.strategies.values()])
        self._mask = np.zeros((len(self.strategies), max_strategies),
                              dtype=bool)
        for r, strats in enumerate(self.strategies.values()):
            self._mask[r, :len(strats)] = True

    def all_profiles(self):
        """Returns a generator over all profiles"""
        return map(profile.Profile, itertools.product(*(
            [(role, collections.Counter(comb)) for comb
             in itertools.combinations_with_replacement(
                 strats, self.players[role])]
            for role, strats in self.strategies.items())))

    def _as_dict(self, array):
        """Converts an array profile representation to a dictionary representation

        """
        if isinstance(array, collections.Mapping):
            return array  # Already a profile
        array = np.asarray(array)
        return {role: {strat: count for strat, count
                       in zip(strats, counts) if count > 0}
                for counts, (role, strats)
                in zip(array, self.strategies.items())}

    def as_profile(self, array):
        """Converts an array profile representation into a dictionary representation

        The dictionary representation is a mapping from roles to strategies to
        counts.

        If a profile is passed in, nothing is changed.

        """
        if isinstance(array, profile.Profile):
            return array
        else:
            return profile.Profile(self._as_dict(array))

    def as_mixture(self, array):
        """Converts an array profile representation into a dictionary representation

        The dictionary representation is a mapping from roles to strategies to
        counts.

        If a profile is passed in, nothing is changed.

        """
        if isinstance(array, profile.Mixture):
            return array
        else:
            return profile.Mixture(self._as_dict(array))

    def as_array(self, prof, dtype=float):
        """Converts a dictionary profile representation into an array representation

        The array representation is a matrix roles x max_strategies where the
        mapping is defined by the order in the strategies dictionary.

        If an array is passed in, nothing is changed.

        By definition, an invalid entries are zero.

        """
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
        mix = np.random.gamma(alpha, size=self._mask.shape) * self._mask
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
        strategies."""
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
            for r in range(len(self.players)):
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
        wrap = self.as_array if as_array else lambda x: x
        return (wrap(profile.Mixture(rs)) for rs in itertools.product(
            *([(r, {s: 1}) for s in sorted(ss)] for r, ss
              in self.strategies.items())))

    def is_symmetric(self):
        """Returns true if this game is symmetric"""
        return len(self.strategies) == 1

    def is_asymmetric(self):
        """Returns true if this game is asymmetric"""
        return all(p == 1 for p in self.players.values())

    def to_json(self):
        """Convert to a json serializable format"""
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies.items()}}

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        params = gameio._game_from_json(json_)
        return EmptyGame(*params[:2])

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


# TODO Make this member function with try catch embedded like min payoff
def _compute_dev_reps(counts, players, exact=False):
    """Uses fast floating point math or at least vectorized computation to compute
    devreps

    """
    # Sets up functions to be exact or approximate
    if exact:
        dtype = object
        factorial = _exact_factorial
        div = lambda a, b: a // b
    else:
        dtype = float
        factorial = spm.factorial
        div = lambda a, b: a / b

    # Actual computation
    strat_counts = np.array(list(players.values()), dtype=dtype)
    player_factorial = factorial(counts).prod(2)
    totals = np.prod(div(factorial(strat_counts), player_factorial), 1)
    dev_reps = div(totals[:, np.newaxis, np.newaxis] *
                   counts, strat_counts[:, np.newaxis])
    return dev_reps


class Game(EmptyGame):
    """Role-symmetric game representation

    Parameters
    ----------
    players : {role: count}
        Mapping from roles to number of players per role.
    strategies : {role: {strategy}}
        Mapping from roles to per-role strategy sets.
    payoff_data : [{role: [(strat, count, [payoff])]}]
        Collection of data objects mapping roles to collections of (strategy,
        count, value) tuples.

    Members
    -------
    players : {role: count}
        An immutable copy of the input players.
    strategies : {role: {strategy}}
        An immutable copy of the input strategies. The iteration order of
        roles, and strategies per role are the cannon iteration order for this
        Game, and the order the strategies and roles will be mapped in the
        array representation. Note that this can be different than the order of
        the object that is passed in.
    min_payoffs : ndarray, shape (num_roles,), dtype float
        The minimum payoff a role can ever have.

    """
    # There are several private members of a game.
    # _role_index : maps role string to its index in arrays
    # _strategy_index : maps a role and strategy string to the strategies index
    # _profile_map : maps static profiles to their index in the values array
    # _values : An array of mean payoff data indexed by [profile, role,
    #           strategy] all as indices

    def __init__(self, players, strategies, payoff_data=(), length=None):
        super().__init__(players, strategies)

        self._role_index = {r: i for i, r in enumerate(self.strategies.keys())}
        self._strategy_index = {r: {s: i for i, s in enumerate(strats)}
                                for r, strats in self.strategies.items()}

        if length is None:
            if not hasattr(payoff_data, "__len__"):
                payoff_data = list(payoff_data)
            length = len(payoff_data)
        self._profile_map = {}
        self._values = np.zeros((length,) + self._mask.shape)
        self._counts = np.zeros_like(self._values, dtype=int)

        for p, profile_data in enumerate(payoff_data):
            prof = profile.Profile((role, {s: c for s, c, _ in dats})
                                   for role, dats in profile_data.items())
            assert prof not in self._profile_map, \
                'Duplicate profile {}'.format(prof)
            if any(any(p is None for _, _, p in dat)
                   for dat in profile_data.values()):
                warnings.warn('Encountered null payoff data in profile: {0}'
                              .format(prof))
                continue  # Invalid data, but can continue

            self._profile_map[prof] = p

            for r, role in enumerate(self.strategies):
                for strategy, count, payoffs in profile_data[role]:
                    s = self._strategy_index[role][strategy]
                    assert self._counts[p, r, s] == 0, (
                        'Duplicate role strategy pair ({}, {})'
                        .format(role, strategy))
                    self._values[p, r, s] = np.average(payoffs)
                    self._counts[p, r, s] = count

        try:  # Use approximate unless it overflows
            self._dev_reps = _compute_dev_reps(self._counts, self.players)
        except FloatingPointError:
            self._dev_reps = _compute_dev_reps(self._counts, self.players,
                                               exact=True)
        self._compute_min_payoffs()

    def _compute_min_payoffs(self):
        """Assigns _min_payoffs to the minimum payoff for every role"""
        if (not self._values.size):
            self.min_payoffs = np.empty([self._mask.shape[0]])
            self.min_payoffs.fill(np.nan)
        else:
            self.min_payoffs = (np.ma.masked_array(self._values,
                                                   self._counts == 0)
                                .min((0, 2)).filled(np.nan))

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

    def _payoff_dict(self, counts, values):
        """Merges a value/payoff array and a counts array into a payoff dict"""
        return {role: {strat: payoff for strat, count, payoff
                       in zip(strats, s_count, s_value) if count > 0}
                for (role, strats), s_count, s_value
                in zip(self.strategies.items(), counts, values)}

    def get_payoffs(self, profile, as_array=False):
        """Returns a dictionary mapping roles to strategies to payoff"""
        index = self._profile_map[self.as_profile(profile)]
        payoffs = self._values[index]
        if as_array:
            return payoffs
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
            return dict(zip(self.strategies, payoff))

    def get_max_social_welfare(self, role=None, as_array=False):
        """Returns the maximum social welfare over the known profiles.

        :param role: If specified, get maximum welfare for that role
        :param as_array: If true, the maximum social welfare profile is
            returned in its array representation

        :returns: Maximum social welfare
        :returns: Profile with the maximum social welfare
        """
        # XXX This should probably stay here, because it can't be moved without
        # exposing underlying structure or making it less efficient
        if role is not None:
            role_index = self.role_index[role]
            counts = self._counts[:, role_index][..., np.newaxis]
            values = self._values[:, role_index][..., np.newaxis]
        else:
            counts = self._counts
            values = self._values

        welfares = np.sum(values * counts, (1, 2))
        profile_index = welfares.argmax()
        profile = self._counts[profile_index]
        if as_array:
            profile = self.as_profile(profile)
        return welfares[profile_index], profile

    def expected_values(self, mix, as_array=False):
        """Computes the expected value of each pure strategy played against all
        opponents playing mix.

        """
        # The first use of 'tiny' makes 0^0=1.
        # The second use of 'tiny' makes 0/0=0.
        mix = self.as_array(mix)
        old = np.seterr(under='ignore')  # ignore underflow
        weights = (((mix + _TINY)**self._counts).prod((1, 2))[:, None, None]
                   * self._dev_reps / (mix + _TINY))
        values = np.sum(self._values * weights, 0)
        np.seterr(**old)  # Go back to old settings
        if as_array:
            return values
        else:
            return {role: {strat: value for strat, value
                           in zip(strats, s_values)}
                    for (role, strats), s_values
                    in zip(self.strategies.items(), values)}

    def is_complete(self):
        """Returns true if every profile has data"""
        return len(self._profile_map) == self._size

    def is_constant_sum(self):
        """Returns true if this game is constant sum"""
        profile_sums = np.sum(self._counts * self._values, (1, 2))
        return np.allclose(profile_sums, np.mean(profile_sums))

    def items(self, as_array=False):
        """Identical to payoffs

        Returns an iterable of tuples of (profile, payoffs). This is to make a
        Game behave like a dictionary from profiles to payoff dictionaries.

        If as_array is True, they are given in their array representation

        """
        return self.payoffs(as_array=as_array)

    def __contains__(self, profile):
        """Returns true if data for that profile exists"""
        return profile in self._profile_map

    def __iter__(self):
        """Basically identical to an iterator over data_profiles"""
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
            total=self._size)

    def __str__(self):
        return ('{old}payoff data for {data:d} out of {total:d} '
                'profiles').format(
                    old=super().__str__(), data=len(self._profile_map),
                    total=self._size)

    def to_json(self):
        """Convert to json according to the egta-online v3 default game spec"""
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies.items()},
                'profiles': [
                    {role:
                     [(strat, count, payoffs[role][strat])
                      for strat, count in strats.items()]
                     for role, strats in prof.items()}
                    for prof, payoffs in self.payoffs()]}

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        return Game(*gameio._game_from_json(json_))


class SampleGame(Game):
    """A Role Symmetric Game that has multiple samples per observation"""
    def __init__(self, players, strategies, payoff_data=(), length=None):
        # Copy to list if necessary
        if not hasattr(payoff_data, "__len__"):
            payoff_data = list(payoff_data)
        # Super constructor
        super().__init__(players, strategies, payoff_data, length)

        new_locations = []
        sample_values = {}

        for p, profile_data in enumerate(payoff_data):
            num_samples = min(min(len(payoffs) if hasattr(payoffs, "__len__")
                                  else 1
                                  for _, __, payoffs in sym_grps)
                              for sym_grps in profile_data.values())
            values = np.zeros((1,) + self._mask.shape + (num_samples,))

            for r, role in enumerate(self.strategies):
                for strategy, count, payoffs in profile_data[role]:
                    if not hasattr(payoffs, "__len__"):
                        payoffs = [payoffs]
                    if len(payoffs) > num_samples:
                        warnings.warn("Truncating observation data")
                    s = self._strategy_index[role][strategy]
                    values[0, r, s] = payoffs[:num_samples]

            value_list = sample_values.setdefault(num_samples, [])
            new_locations.append((num_samples, len(value_list)))
            value_list.append(values)

        profiles_before = 0
        sample_to_profiles_before = {}
        sample_to_bucket = {}
        self._sample_values = []
        for i, (samps, values) in enumerate(sorted(sample_values.items())):
            sample_to_bucket[samps] = i
            sample_to_profiles_before[samps] = profiles_before
            self._sample_values.append(np.concatenate(values))
            profiles_before += len(values)

        self._sample_profile_map = {
            prof: (sample_to_bucket[new_locations[index][0]],
                   new_locations[index][1])
            for prof, index in self._profile_map.items()}

        perm = [sample_to_profiles_before[samps] + samps_idx
                for samps, samps_idx in new_locations]

        self._values[perm] = self._values.copy()
        self._counts[perm] = self._counts.copy()
        self._profile_map = {prof: perm[index] for prof, index
                             in self._profile_map.items()}

    def _sample_payoff_dict(self, counts, payoffs):
        """Returns sample payoff array as dict"""
        return {role:
                {strat: list(payoffs) for strat, payoffs in strats.items()}
                for role, strats
                in self._payoff_dict(counts, payoffs).items()}

    def get_sample_payoffs(self, profile, as_array=False):
        """Returns a dictionary mapping roles to strategies to payoffs"""
        prof = self.as_profile(profile)
        bucket, index = self._sample_profile_map[prof]
        payoffs = self._sample_values[bucket][index]
        if as_array:
            return payoffs
        counts = self._counts[self._profile_map[prof]]
        return self._sample_payoff_dict(counts, payoffs)

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
                     self._sample_payoff_dict(counts, payoffs))
                    for counts, payoffs in iterable)

    def remean(self):
        """Overwrite payoff values with mean

        This uses the mean of the stored observations, which may be fewer than
        the original amount. Calling this on a newly created game may change
        the results."""
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

    def single_sample(self, independent_profile=False, independent_role=False,
                      independent_strategy=False):
        """Overwrite payoff values with a single sample

        Keyword arguments function the same as they do for resample.
        """
        self.resample(num_resamples=1, independent_profile=independent_profile,
                      independent_role=independent_role,
                      independent_strategy=independent_strategy)

    def __repr__(self):
        if len(self._sample_values) == 1:
            return '{old}, {samples:d}>'.format(
                old=super().__repr__()[:-1],
                samples=self._sample_values[0].shape[3])
        else:
            return '{old}, {min_samples:d} - {max_samples:d}>'.format(
                old=super().__repr__()[:-1],
                min_samples=self._sample_values[0].shape[3],
                max_samples=self._sample_values[-1].shape[3])

    def __str__(self):
        if len(self._sample_values) == 1:
            return '{old}\n{samples:d} observations per profile'.format(
                old=super().__str__(),
                samples=self._sample_values[0].shape[3])
        else:
            return ('{old}\n{min_samples:d} to {max_samples:d} observations '
                    'per profile').format(
                        old=super().__str__(),
                        min_samples=self._sample_values[0].shape[3],
                        max_samples=self._sample_values[-1].shape[3])

    def to_json(self):
        """Convert to json according to the egta-online v3 default game spec"""
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies.items()},
                'profiles': [
                    {role:
                     [(strat, count, payoffs[role][strat])
                      for strat, count in strats.items()]
                     for role, strats in prof.items()}
                    for prof, payoffs in self.sample_payoffs()]}

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        return SampleGame(*gameio._game_from_json(json_))
