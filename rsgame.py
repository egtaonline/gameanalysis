'''This module contains data structures and accompanying methods for working
with role symmetric games'''
import json
import itertools
import numpy as np
from collections import Counter

import funcs
from hcollections import frozendict


class Profile(frozendict):
    '''A profile is a static assignment of players to roles and strategies

    This is an immutable container that maps roles to strategies to counts

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(((r, frozendict(p)) for r, p
                          in dict(*args, **kwargs).items()))

    def remove(self, role, strategy):
        '''Return a new profile with one less player playing strategy'''
        copy = dict(self)
        role_copy = dict(copy[role])
        copy[role] = role_copy
        role_copy[strategy] -= 1
        if role_copy[strategy] == 0:
            role_copy.pop(strategy)
        return Profile(copy)

    def add(self, role, strategy):
        '''Return a new profile where strategy has one more player'''
        copy = dict(self)
        role_copy = Counter(copy[role])
        copy[role] = role_copy
        role_copy[strategy] += 1
        return Profile(copy)

    def deviate(self, role, strategy, deviation):
        '''Returns a new profile where one player deviated'''
        copy = dict(self)
        role_copy = Counter(copy[role])
        copy[role] = role_copy
        role_copy[strategy] -= 1
        role_copy[deviation] += 1
        if role_copy[strategy] == 0:
            role_copy.pop(strategy)
        return Profile(copy)

    def __str__(self):
        return '; '.join('%s: %s' %
                         (role, ', '.join('%d %s' % (count, strat)
                                          for strat, count in strats.items()))
                         for role, strats in self.items())

    def dump_json(self, file_like, **kwargs):
        '''Dump a profile to a file like in json'''
        json.dump({'type': 'GA_Profile',
                   'data': {r: dict(s) for r, s in self.items()}},
                  file_like, **kwargs)

    def load_json(file_like):
        '''Load a profile from a file like'''
        primitive = json.load(file_like)
        assert primitive['type'] == 'GA_Profile', 'Improper type of profile'
        return Profile(primitive['data'])


class Mixture(frozendict):
    '''A mixture over roles and strategies

    This is an immutable (hashable) data structure that maps roles to
    strategies to probability of play. Only strategies in support will have a
    probability, and the probabilities will sum to 1 for each role.

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(self, ((r, frozendict(p)) for r, p
                                in dict(*args, **kwargs).items()))

    def dump_json(self, file_like, **kwargs):
        '''Dump a mixture to a file like in json'''
        json.dump({'type': 'GA_Mixture',
                   'data': {r: dict(s) for r, s in self.items()}},
                  file_like, **kwargs)

    def load_json(file_like):
        '''Load a mixture from a file like'''
        primitive = json.load(file_like)
        assert primitive['type'] == 'GA_Mixture', 'Improper type of mixture'
        return Profile(primitive['data'])


# TODO Remove
# @functools.total_ordering
class Game(object):
    '''Role-symmetric game representation.

    players:     mapping from roles to number of players per role
    strategies:  mapping from roles to per-role strategy sets
    payoff_data: collection of data objects mapping roles to collections
                 of (strategy, count, value) tuples
    '''
    def __init__(self, players, strategies, payoff_data=()):
        # The number of players in each role
        self.players = frozendict(players)
        # Strategies contains the default ordering of roles and strategies
        self.strategies = frozendict((r, frozenset(s))
                                     for r, s in strategies.items())

        self._max_strategies = max(len(s) for s in self.strategies.values())

        # TODO This is for replicator dynamics, it should be moved there. It
        # should be possible to do with a masked min over the counts array
        # greater than 0
        self._min_payoffs = np.empty((len(self.strategies),
                                      self._max_strategies))
        self._min_payoffs.fill(np.inf)

        # TODO: Generate this better
        # TODO This might not be necessary
        self._mask = np.array([[False]*len(s) + [True]*(self._max_strategies - len(s))
                               for s in self.strategies.values()])
        self._size = funcs.prod(funcs.game_size(self.players[role], len(strats))
                                for role, strats in self.strategies.items())
        self._role_index = {r: i for i, r in enumerate(self.strategies.keys())}
        self._strategy_index = {r: {s: i for i, s in enumerate(strats)}
                                for r, strats in self.strategies.items()}

        payoff_data = list(payoff_data)  # To measure length
        self._profile_map = {}
        self._values = np.zeros((len(payoff_data), 
                                 self._max_strategies))
        self._counts = np.zeros_like(self._values, dtype=int)
        # TODO make uint? May screw up arithmetic
        self._dev_reps = np.zeros_like(self._values, dtype=int)

        for p, profile_data in enumerate(payoff_data):
            prof = Profile((r, {s: c for s, c, _ in dats})
                           for r, dats in profile_data.items())
            self._profile_map[prof] = p
            assert prof not in self._profile_map, 'Duplicate profile %s' % prof

            for r, role in self.strategies.keys():
                for strategy, count, payoffs in profile_data[role]:
                    s = self._strategy_index[role][strategy]
                    assert self._counts[p, r, s] == 0, (
                        'Duplicate role strategy pair (%s, %s)'
                        % (role, strategy))
                    self._values[p, r, s] = np.average(payoffs)
                    self._counts[p, r, s] = count

            # Must be done after counts is complete
            for r, (role, strats) in enumerate(self.strategies.items()):
                for s, strat in enumerate(strats):
                    if self._counts[p, r, s] > 0:
                        opp_prof = np.copy(self._counts[p])
                        opp_prof[r, s] -= 1
                        try:
                            self._dev_reps[p, r, s] = funcs\
                                .profile_repetitions(opp_prof)
                        except OverflowError:
                            self._dev_reps = np.array(self._dev_reps,
                                                      dtype=object)
                            self._dev_reps[p, r, s] = funcs\
                                .profile_repetitions(opp_prof)

    def __hash__(self):
        return hash((self.players, self.strategies))

    def get_payoff(self, profile, role, strategy):
        '''Returns the payoff for a specific profile, role, and strategy'''
        p = self._profile_map[profile]
        r = self._role_index[role]
        s = self._strategy_index[role][strategy]
        return self._values[p, r, s]

    def get_payoffs(self, profile):
        '''Returns a dictionary mapping roles to strategies to payoff'''
        payoffs = self._values[self._profile_map[profile]]
        return {role: dict(zip(strats, strat_payoffs))
                for (role, strats), strat_payoffs
                in zip(self.strategies.items(), payoffs)}

    def getExpectedPayoff(self, mix, role=None):
        if role is None:
            return (mix * self.expectedValues(mix)).sum(1)
        return (mix * self.expectedValues(mix)).sum(1)[self.index(role)]

    def getSocialWelfare(self, profile):
        if is_pure_profile(profile):
            return self.values[self[profile]].sum()
        if is_mixture_array(profile):
            players = np.array([self.players[r] for r in self.roles])
            return (self.getExpectedPayoff(profile) * players).sum()
        if is_profile_array(profile):
            return self.getSocialWelfare(self.toProfile(profile))
        if is_mixed_profile(profile):
            return self.getSocialWelfare(self.toArray(profile))

    def expected_values(self, mix):
        '''Computes the expected value of each pure strategy played against all
        opponents playing mix.

        '''
        # The first use of 'tiny' makes 0^0=1.
        # The second use of 'tiny' makes 0/0=0.
        tiny = np.finfo(float).tiny

        weights = (((mix+tiny)**self._counts).prod((1, 2))[:, None, None]
                   * self._dev_reps / (mix+tiny))
        values = np.sum(self._values * weights, 0)
        return values

    def all_profiles(self):
        '''Returns an generator over all profiles'''
        return map(Profile, itertools.product(*(
            ((role, Counter(comb)) for comb
             in itertools.combinations_with_replacement(
                 strats, self.players[role]))
            for role, strats in self.strategies.items())))

    def is_complete(self):
        '''Returns true if every profile has data'''
        return len(self._profile_map) == self._size

    def uniformMixture(self):
        return np.array(1-self.mask, dtype=float) / \
                (1-self.mask).sum(1).reshape(len(self.roles),1)

    def randomMixture(self):
        # TODO, this should probably be dirichlet, e.g. normalized gammas
        m = np.random.uniform(0, 1, size=self.mask.shape) * (1 - self.mask)
        return m / m.sum(1).reshape(m.shape[0], 1)

    def biasedMixtures(self, role=None, strategy=None, bias=.9):
        '''
        Gives mixtures where the input strategy has bias weight for its role.

        Probability for that role's remaining strategies is distributed
        uniformly, as is probability for all strategies of other roles.

        Returns a list even when a single role & strategy are specified, since
        the main use case is starting replicator dynamics from several mixtures.
        '''
        assert 0 <= bias <= 1, "probabilities must be between zero and one"
        if self.maxStrategies == 1:
            return [self.uniformMixture()]
        if role == None:
            return flatten([self.biasedMixtures(r, strategy, bias) for r \
                    in filter(lambda r: self.numStrategies[self.index(r)] \
                    > 1, self.roles)])
        if strategy == None:
            return flatten([self.biasedMixtures(role, s, bias) for s in \
                    self.strategies[role]])
        i = self.array_index(role, strategy, dtype=float)
        m = 1. - self.mask - i
        m /= m.sum(1).reshape(m.shape[0], 1)
        m[self.index(role)] *= (1. - bias)
        m += i*bias
        return [m]

    def toProfile(self, arr, supp_thresh=1e-3):
        arr = np.array(arr)
        if is_mixture_array(arr):
            arr[arr < supp_thresh] = 0
            sums = arr.sum(1).reshape(arr.shape[0], 1)
            if np.any(sums == 0):
                raise ValueError("no probability greater than threshold.")
            arr /= sums
        p = {}
        for r in self.roles:
            i = self.index(r)
            p[r] = {}
            for s in self.strategies[r]:
                j = self.index(r, s)
                if arr[i,j] > 0:
                    p[r][s] = arr[i,j]
        return Profile(p)

    def toArray(self, prof):
        if is_mixed_profile(prof):
            a = self.zeros(dtype=float)
        elif is_pure_profile(prof):
            a = self.zeros(dtype=int)
        else:
            raise TypeError(funcs.one_line(
                "unrecognized profile type: %s" % prof, 71))
        for role in prof.keys():
            i = self.index(role)
            for strategy in prof[role].keys():
                j = self.index(role, strategy)
                a[i,j] = prof[role][strategy]
        return a

    # TODO Remove
    # def __eq__(self, other):
    #     return (self._players == other._players and
    #             self.strategies == other.strategies and
    #             sorted(self._profile_map.keys()) == sorted(other._profile_map.keys()))

    # def __lt__(self, other):
    #     return (self._players < other._players and
    #             self.strategies < other.strategies and
    #             sorted(self._profile_map.keys()) < sorted(other._profile_map.keys()))

    def is_symmetric(self):
        '''Returns true if this game is symmetric'''
        return len(self.strategies) == 1

    def is_asymmetric(self):
        '''Returns true if this game is asymmetric'''
        return all(p == 1 for p in self.players.values())

    def is_constant_sum(self):
        '''Returns true if this game is constant sum'''
        profile_sums = np.sum(self._counts * self._values, (1, 2))
        return np.allclose(profile_sums, np.mean(profile_sums))

    def __contains__(self, profile):
        '''Returns true if data for that profile exists'''
        return profile in self._profile_map

    def __iter__(self):
        return iter(self._profile_map)

    def __repr__(self):
        return (('%s:\n\tRoles: %s\n\tPlayers:\n\t\t%s\n\tStrategies:\n\t\t%s\n'
                 'payoff data for %d out of %d profiles') % (
                     self.__class__.__name__,
                     ', '.join(sorted(self.strategies)),
                     '\n\t\t'.join('%dx %s' % (count, role)
                                   for role, count
                                   in sorted(self.players.items())),
                     '\n\t\t'.join('%s:\n\t\t\t%s' % (
                         role,
                         '\n\t\t\t'.join(strats))
                                   for role, strats in sorted(self.strategies.items())),
                     len(self._profile_map),
                     self._size)).expandtabs(4)

    # def to_TB_JSON(self):
    #     '''
    #     Convert to JSON according to the EGTA-online v3 default game spec.
    #     '''
    #     game_dict = {}
    #     game_dict["roles"] = [{"name":role, "count":self.players[role], \
    #                 "strategies": list(self.strategies[role])} for role \
    #                 in self.roles]
    #     game_dict["profiles"] = []
    #     for prof in self:
    #         p = self[prof]
    #         sym_groups = []
    #         for r, role in enumerate(self.roles):
    #             for strat in prof[role]:
    #                 s = self.index(role, strat)
    #                 sym_groups.append({"role":role, "strategy":strat, \
    #                         "count":self.counts[p][r,s], \
    #                         "payoff":float(self.values[p][r,s])})
    #         game_dict["profiles"].append({"symmetry_groups":sym_groups})
    #     return game_dict

    def dump_json(self, file_like):
        '''Convert to json according to the egta-online v3 default game spec'''
        json.dump({'players': dict(self.players),
                   'strategies': {r: list(s) for r, s in self.strategies},
                   'profiles': list(itertools.chain.from_iterable(
                       ({role: [(strat, count, self.get_payoff(prof, role, strat))
                                for strat, count in strats.items()]}
                        for role, strats in prof.items())
                       for prof in self._profile_map))}, file_like)


# TODO Sample game is not being refactored until Varsha's changes have been integrated
# class SampleGame(Game):
#     '''A Role Symmetric Game that has multiple samples per observation'''
#     def __init__(self, self, players, strategies, payoff_data=()):
#         super().__init(players, strategies, payoff_data)
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
#         '''
#         Overwrites self.values with a bootstrap resample of self.sample_values.

#         pair = payoff: resample all payoff observations independently
#         pair = profile: resample paired profile observations
#         pair = game: resample paired game observations
#         '''
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
#         '''Makes self.values be a single sample from each sample set.'''
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
#         '''
#         Convert to JSON according to the EGTA-online v3 default game spec.
#         '''
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
#     #     '''
#     #     Convert to JSON according to the EGTA-online v3 sample-game spec.
#     #     '''
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


def is_pure_profile(prof):
    if not isinstance(prof, h_dict):
        return False
    flat = flatten([v.values() for v in prof.values()])
    return all([isinstance(count, int) and count >= 0 for count in flat])


def is_mixed_profile(prof):
    if not isinstance(prof, h_dict):
        return False
    flat = flatten([v.values() for v in prof.values()])
    return all([prob >= 0 for prob in flat]) and \
            np.allclose(sum(flat), len(prof))


def is_profile_array(arr):
    return isinstance(arr, np.ndarray) and np.all(arr >= 0) and \
            arr.dtype == int


def is_mixture_array(arr):
    if not isinstance(arr, np.ndarray):
        return False
    if arr.dtype == "object":
        arr = np.array(arr, dtype=float)
    return np.all(arr >= 0) and np.allclose(arr.sum(1), 1)
