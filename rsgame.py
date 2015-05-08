'''This module contains data structures and accompanying methods for working
with role symmetric games'''
import itertools
import numpy as np
import scipy.misc as spm
from collections import Counter

import funcs
from hcollections import frozendict


class PureProfile(frozendict):
    '''A pure profile is a static assignment of players to roles and strategies

    This is an immutable container that maps roles to strategies to
    counts. Only strategies with at least one player playing them are
    represented.

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
        return PureProfile(copy)

    def add(self, role, strategy):
        '''Return a new profile where strategy has one more player'''
        copy = dict(self)
        role_copy = Counter(copy[role])
        copy[role] = role_copy
        role_copy[strategy] += 1
        return PureProfile(copy)

    def deviate(self, role, strategy, deviation):
        '''Returns a new profile where one player deviated'''
        copy = dict(self)
        role_copy = Counter(copy[role])
        copy[role] = role_copy
        role_copy[strategy] -= 1
        role_copy[deviation] += 1
        if role_copy[strategy] == 0:
            role_copy.pop(strategy)
        return PureProfile(copy)

    def to_json(self):
        '''Return a representation that is json serializable'''
        return {'type': 'GA_PureProfile',
                'data': {r: dict(s) for r, s in self.items()}}

    @staticmethod
    def from_json(json_):
        '''Load a profile from its json representation'''
        assert json_['type'] == 'GA_PureProfile', 'Improper type of profile'
        return PureProfile(json_['data'])

    def __str__(self):
        return '; '.join('%s: %s' %
                         (role, ', '.join('%d %s' % (count, strat)
                                          for strat, count in strats.items()))
                         for role, strats in self.items())

    def __repr__(self):
        return 'PureProfile' + super().__repr__()[10:]


class MixedProfile(frozendict):
    '''A mixed profile is distribution over strategies for each role.

    This is an immutable container that maps roles to strategies to
    probabilities. Only strategies with support are represented.

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(((r, frozendict(p)) for r, p
                          in dict(*args, **kwargs).items()))

    def trim_support(self, supp_thresh=1e-3):
        '''Returns a new mixed profiles without strategies played less than
        supp_thresh

        '''
        def process_roles():
            for role, strats in self.items():
                new_strats = [(strat, prob) for strat, prob in strats.items()
                              if prob >= supp_thresh]
                total_prob = sum(prob for _, prob in new_strats)
                yield role, {strat: p / total_prob for strat, p in new_strats}
        return MixedProfile(process_roles())

    def to_json(self):
        '''Return a representation that is json serializable'''
        return {'type': 'GA_MixedProfile',
                'data': {r: dict(s) for r, s in self.items()}}

    @staticmethod
    def from_json(json_):
        '''Load a profile from its json representation'''
        assert json_['type'] == 'GA_MixedProfile', 'Improper type of profile'
        return PureProfile(json_['data'])

    def __repr__(self):
        return 'MixedProfile' + super().__repr__()[10:]


class EmptyGame(object):
    '''Role symmetric game representation

    This object only contains methods and information about definition of the
    game, and does not contain methods to operate on observation data.

    players:    a mapping from roles to number of players in that role
    strategies: a mapping from roles to strategies

    '''
    def __init__(self, players, strategies, _=None):
        self.players = frozendict(players)
        self.strategies = frozendict((r, frozenset(s))
                                     for r, s in strategies.items())

        self._max_strategies = max(len(s) for s in self.strategies.values())
        self._obs_size = (len(self.strategies), self._max_strategies)

    def all_profiles(self):
        '''Returns a generator over all profiles'''
        return map(PureProfile, itertools.product(*(
            [(role, Counter(comb)) for comb
             in itertools.combinations_with_replacement(
                 strats, self.players[role])]
            for role, strats in self.strategies.items())))

    def to_profile(self, array):
        '''Converts an array profile representation into a dictionary representation

        The dictionary representation is a mapping from roles to strategies to
        either counts or probabilities.

        If a profile is passed in, nothing is changed.

        '''
        if isinstance(array, collections.Mapping):
            return array  # Already a profile
        array = np.asarray(array)
        type_map = {float: MixedProfile, int: PureProfile}
        return type_map[array.dtype]((role, zip(strats, counts))
                                     for counts, (role, strats)
                                     in zip(array, self.strategies.items()))

    def to_array(self, prof):
        '''Converts a dictionary profile representation into an array representation

        The array representation is a matrix roles x max_strategies where the
        mapping is defined by the order in the strategies dictionary.

        If an array is passed in, nothing is changed.

        '''

        if isinstance(prof, np.ndarray):  # Already an array
            return prof
        type_ = type(next(iter(next(iter(prof.values())).values())))
        array = np.zeros(self._obs_size, dtype=type_)
        for r, (role, strats) in enumerate(prof.items()):
            for s, (strategy, payoff) in enumerate(strats.items()):
                array[r, s] = payoff
        return array

    def is_symmetric(self):
        '''Returns true if this game is symmetric'''
        return len(self.strategies) == 1

    def is_asymmetric(self):
        '''Returns true if this game is asymmetric'''
        return all(p == 1 for p in self.players.values())

    def to_json(self):
        '''Convert to a json serializable format'''
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies}}

    @staticmethod
    def from_json(json_):
        '''Load a profile from its json representation'''
        params = gameio._game_from_json(json_)
        return EmptyGame(*params[:2])

    def __repr__(self):
        return (('%s:\n\t'
                 'Roles: %s\n\t'
                 'Players:\n\t\t%s\n\t'
                 'Strategies:\n\t\t%s\n') % (
                     self.__class__.__name__,
                     ', '.join(sorted(self.strategies)),
                     '\n\t\t'.join('%dx %s' % (count, role)
                                   for role, count
                                   in sorted(self.players.items())),
                     '\n\t\t'.join('%s:\n\t\t\t%s' % (
                         role,
                         '\n\t\t\t'.join(strats))
                                   for role, strats
                                   in sorted(self.strategies.items()))
                 )).expandtabs(4)


class Game(EmptyGame):
    '''Role-symmetric game representation.

    players:     mapping from roles to number of players per role
    strategies:  mapping from roles to per-role strategy sets
    payoff_data: collection of data objects mapping roles to collections
                 of (strategy, count, value) tuples
    '''
    def __init__(self, players, strategies, payoff_data=()):
        super().__init__(players, strategies)

        # TODO This is for replicator dynamics, it should be moved there. It
        # should be possible to do with a masked min over the counts array
        # greater than 0
        self._min_payoffs = np.empty(self._obs_size)
        self._min_payoffs.fill(np.inf)

        # TODO: Generate this better
        # TODO This might not be necessary
        # self._mask = np.array([[False]*len(s) + [True]*(self._max_strategies - len(s))
        #                        for s in self.strategies.values()])
        self._size = funcs.prod(funcs.game_size(self.players[role], len(strats))
                                for role, strats in self.strategies.items())
        self._role_index = {r: i for i, r in enumerate(self.strategies.keys())}
        self._strategy_index = {r: {s: i for i, s in enumerate(strats)}
                                for r, strats in self.strategies.items()}

        payoff_data = list(payoff_data)  # To measure length
        self._profile_map = {}
        self._values = np.zeros((len(payoff_data),
                                 len(self.strategies),
                                 self._max_strategies))
        self._counts = np.zeros_like(self._values, dtype=int)

        for p, profile_data in enumerate(payoff_data):
            prof = PureProfile((role, {s: c for s, c, _ in dats})
                               for role, dats in profile_data.items())
            assert prof not in self._profile_map, 'Duplicate profile %s' % prof
            self._profile_map[prof] = p

            for r, role in enumerate(self.strategies):
                for strategy, count, payoffs in profile_data[role]:
                    s = self._strategy_index[role][strategy]
                    assert self._counts[p, r, s] == 0, (
                        'Duplicate role strategy pair (%s, %s)'
                        % (role, strategy))
                    self._values[p, r, s] = np.average(payoffs)
                    self._counts[p, r, s] = count

        # Approximate dev_reps
        strat_counts = np.array(list(self.players.values()))
        player_factorial = spm.factorial(self._counts).prod(2)
        totals = np.prod(spm.factorial(strat_counts) / player_factorial, 1)
        self._dev_reps = (totals[:, np.newaxis, np.newaxis] *
                          self._counts / strat_counts[:, np.newaxis])

        # self._dev_reps = np.zeros_like(self._values, dtype=int)
            # # TODO move to another scan? only dependent on counts
            # # Must be done after counts is complete
            # for r, (role, strats) in enumerate(self.strategies.items()):
            #     for s, strat in enumerate(strats):
            #         if self._counts[p, r, s] > 0:
            #             opp_prof = np.copy(self._counts[p])
            #             opp_prof[r, s] -= 1
            #             try:
            #                 self._dev_reps[p, r, s] = funcs\
            #                     .profile_repetitions(opp_prof)
            #             except OverflowError:
            #                 self._dev_reps = np.array(self._dev_reps,
            #                                           dtype=object)
            #                 self._dev_reps[p, r, s] = funcs\
            #                     .profile_repetitions(opp_prof)

        self._compute_min_payoffs()

    def __hash__(self):
        return hash((self.players, self.strategies))

    def _compute_min_payoffs(self):
        '''Assigns _min_payoffs to the minimum payoff for every role strategy

        '''
        self._min_payoffs = (np.ma.masked_array(self._values, self._counts == 0)
                             .min(0).filled(0))

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

    # def getExpectedPayoff(self, mix, role=None):
    #     if role is None:
    #         return (mix * self.expectedValues(mix)).sum(1)
    #     return (mix * self.expectedValues(mix)).sum(1)[self.index(role)]

    # def getSocialWelfare(self, profile):
    #     if is_pure_profile(profile):
    #         return self.values[self[profile]].sum()
    #     if is_mixture_array(profile):
    #         players = np.array([self.players[r] for r in self.roles])
    #         return (self.getExpectedPayoff(profile) * players).sum()
    #     if is_profile_array(profile):
    #         return self.getSocialWelfare(self.toProfile(profile))
    #     if is_mixed_profile(profile):
    #         return self.getSocialWelfare(self.toArray(profile))

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

    def is_complete(self):
        '''Returns true if every profile has data'''
        return len(self._profile_map) == self._size

    # def uniformMixture(self):
    #     return np.array(1-self.mask, dtype=float) / \
    #             (1-self.mask).sum(1).reshape(len(self.roles),1)

    # def randomMixture(self):
    #     # TODO, this should probably be dirichlet, e.g. normalized gammas
    #     m = np.random.uniform(0, 1, size=self.mask.shape) * (1 - self.mask)
    #     return m / m.sum(1).reshape(m.shape[0], 1)

    # def biasedMixtures(self, role=None, strategy=None, bias=.9):
    #     '''
    #     Gives mixtures where the input strategy has bias weight for its role.

    #     Probability for that role's remaining strategies is distributed
    #     uniformly, as is probability for all strategies of other roles.

    #     Returns a list even when a single role & strategy are specified, since
    #     the main use case is starting replicator dynamics from several mixtures.
    #     '''
    #     assert 0 <= bias <= 1, "probabilities must be between zero and one"
    #     if self.maxStrategies == 1:
    #         return [self.uniformMixture()]
    #     if role == None:
    #         return flatten([self.biasedMixtures(r, strategy, bias) for r \
    #                 in filter(lambda r: self.numStrategies[self.index(r)] \
    #                 > 1, self.roles)])
    #     if strategy == None:
    #         return flatten([self.biasedMixtures(role, s, bias) for s in \
    #                 self.strategies[role]])
    #     i = self.array_index(role, strategy, dtype=float)
    #     m = 1. - self.mask - i
    #     m /= m.sum(1).reshape(m.shape[0], 1)
    #     m[self.index(role)] *= (1. - bias)
    #     m += i*bias
    #     return [m]

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
        return '%spayoff data for %d out of %d profiles' % (
            super().__repr__(),
            len(self._profile_map),
            self._size)

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

    def to_json(self):
        '''Convert to json according to the egta-online v3 default game spec'''
        return {'players': dict(self.players),
                'strategies': {r: list(s) for r, s in self.strategies},
                'profiles': list(itertools.chain.from_iterable(
                    ({role: [(strat, count, self.get_payoff(prof, role, strat))
                             for strat, count in strats.items()]}
                     for role, strats in prof.items())
                    for prof in self._profile_map))}

    @staticmethod
    def from_json(json_):
        '''Load a profile from its json representation'''
        return Game(*gameio._game_from_json(json_))


# TODO Sample game is not being refactored until Varsha's changes have been integrated
class SampleGame(Game):
    '''A Role Symmetric Game that has multiple samples per observation'''
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
