"""This module contains data structures and accompanying methods for working
with role symmetric games"""

import itertools
import math
import collections

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
    def __init__(self, players, strategies, _=None):
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
    def __init__(self, players, strategies, payoff_data):
        super().__init__(players, strategies)

        self._role_index = {r: i for i, r in enumerate(self.strategies.keys())}
        self._strategy_index = {r: {s: i for i, s in enumerate(strats)}
                                for r, strats in self.strategies.items()}

        if not hasattr(payoff_data, "__len__"):
            payoff_data = list(payoff_data)
        self._profile_map = {}
        self._values = np.zeros((len(payoff_data),) + self._mask.shape)
        self._counts = np.zeros_like(self._values, dtype=int)

        for p, profile_data in enumerate(payoff_data):
            prof = profile.Profile((role, {s: c for s, c, _ in dats})
                                   for role, dats in profile_data.items())
            assert prof not in self._profile_map, \
                'Duplicate profile {}'.format(prof)
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
        # TODO Remove filled? There should be no mask
        self.min_payoffs = (np.ma.masked_array(self._values,
                                               self._counts == 0)
                            .min((0, 2)).filled(0))

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
            return dict(zip(payoff, self.strategies))

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
        return '{}, {:d} / {:d}>'.format(
            super().__repr__()[:-1],
            len(self._profile_map),
            self._size)

    def __str__(self):
        return '{}payoff data for {:d} out of {:d} profiles'.format(
            super().__str__(),
            len(self._profile_map),
            self._size)

    def to_json(self):
        """Convert to json according to the egta-online v3 default game spec"""
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies.items()},
                'profiles': [
                    {role:
                     [(strat, count, [self.get_payoff(prof, role, strat)])
                      for strat, count in strats.items()]
                     for role, strats in prof.items()}
                    for prof in self._profile_map]}

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        return Game(*gameio._game_from_json(json_))


# TODO Sample game is not being refactored until Varsha's changes have been
# integrated
#
# TODO make sure sample game has a method to return payoffs to mean of all
# observations, instead of performing a bootstrap sample.
class SampleGame(Game):
    """A Role Symmetric Game that has multiple samples per observation"""
    def __init__(self, players, strategies, payoff_data=()):
        super().__init(players, strategies, payoff_data)
#         self.sample_values = []
#         self.min_samples = np.inf
#         self.max_samples = 0

#     def addProfile(self, role_payoffs):
#         Game.addProfile(self, role_payoffs)
#         self.addSamples(role_payoffs)

#     def addSamples(self, role_payoffs):
#         samples = map(list, self.zeros())
#         for r, role in enumerate(self.roles):
#             played = []
#             for strat, count, values in role_payoffs[role]:
#                 s = self.index(role, strat)
#                 samples[r][s] = values
#                 self.min_samples = min(self.min_samples, len(values))
#                 self.max_samples = max(self.max_samples, len(values))
#                 played.append(strat)
#             for strat in set(self.strategies[role]) - set(played):
#                 s = self.index(role, strat)
#                 p = self.index(role, played[0])
#                 samples[r][s] = [0]*len(samples[r][p])
#             for s in range(self.numStrategies[r], self.maxStrategies):
#                 p = self.index(role, played[0])
#                 samples[r][s] = [0]*len(samples[r][p])
#         self.sample_values.append(np.array(samples))

#     def getPayoffData(self, profile, role, strategy):
#         v = self.sample_values[self[profile]]
#         return v[self.index(role), self.index(role,strategy)]

#     def resample(self, pair="game"):
#         """
#         Overwrites self.values with a bootstrap resample of self.sample_values.

#         pair = payoff: resample all payoff observations independently
#         pair = profile: resample paired profile observations
#         pair = game: resample paired game observations
#         """
#         if pair == "payoff":
#             raise NotImplementedError("TODO")
#         elif pair == "profile":
#             self.values = map(lambda p: np.average(p, 2, weights= \
#                     np.random.multinomial(len(p[0,0]), np.ones( \
#                     len(p[0,0])) / len(p[0,0]))), self.sample_values)
#         elif pair == "game":#TODO: handle ragged arrays
#             if isinstance(self.sample_values, list):
#                 self.sample_values = np.array(self.sample_values, dtype=float)
#             s = self.sample_values.shape[3]
#             self.values = np.average(self.sample_values, 3, weights= \
#                     np.random.multinomial(s, np.ones(s)/s))

#     def singleSample(self):
#         """Makes self.values be a single sample from each sample set."""
#         if self.max_samples == self.min_samples:
#             self.makeArrays()
#             vals = self.sample_values.reshape([prod(self.values.shape), \
#                                                 self.max_samples])
#             self.values = np.array(map(choice, vals)).reshape(self.values.shape)
#         else:
#             self.values = np.array([[[choice(s) for s in r] for r in p] for \
#                                 p in self.sample_values])
#         return self

#     def reset(self):  #TODO: handle ragged arrays
#         self.values = map(lambda p: np.average(p,2), self.sample_values)

#     def toJSON(self):
#         """
#         Convert to JSON according to the EGTA-online v3 default game spec.
#         """
#         game_dict = {}
#         game_dict["players"] = self.players
#         game_dict["strategies"] = self.strategies
#         game_dict["profiles"] = []
#         for prof in self:
#             game_dict["profiles"].append({role:[(strat, prof[role][strat], \
#                     list(self.sample_values[self[prof]][self.index(role), \
#                     self.index(role, strat)])) for strat in prof[role]] for \
#                     role in prof})
#         return game_dict

#     # def to_TB_JSON(self):
#     #     """
#     #     Convert to JSON according to the EGTA-online v3 sample-game spec.
#     #     """
#     #     game_dict = {}
#     #     game_dict["roles"] = [{"name":role, "count":self.players[role], \
#     #                 "strategies": list(self.strategies[role])} for role \
#     #                 in self.roles]
#     #     game_dict["profiles"] = []
#     #     for prof in self:
#     #         p = self[prof]
#     #         obs = {"observations":[]}
#     #         for i in range(self.sample_values[self[prof]].shape[2]):
#     #             sym_groups = []
#     #             for r, role in enumerate(self.roles):
#     #                 for strat in prof[role]:
#     #                     s = self.index(role, strat)
#     #                     sym_groups.append({"role":role, "strategy":strat, \
#     #                             "count":self.counts[p][r,s], \
#     #                             "payoff":float(self.sample_values[p][r,s,i])})
#     #             obs["observations"].append({"symmetry_groups":sym_groups})
#     #         game_dict["profiles"].append(obs)
#     #     return game_dict

#     def __repr__(self):
#         if self.min_samples < self.max_samples:
#             return Game.__repr__(self) + "\n" + str(self.min_samples) + \
#                 "-" + str(self.max_samples) + " samples per profile"
#         return Game.__repr__(self) + "\n" + str(self.max_samples) + \
#             " samples per profile"
