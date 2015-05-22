'''Methods for interacting with mappings from role to strategy to value'''
from gameanalysis.collect import frozendict


class Support(frozendict):
    '''A static assignment of roles to strategies'''
    pass


class PureProfile(frozendict):
    '''A static assignment of players to roles and strategies

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

    def to_input_profile(self, payoff_map):
        '''Given a payoff map, which maps role to strategy to payoffs, return an input
        profile for game construction

        This requires that the payoff map contains data for every role and
        strategy in the profile. An input profile looks like {role: [(strat,
        count, payoffs)]}, and is necessary to construct a game object.

        '''
        return {role: {(strat, counts, payoff_map[role][strat])
                       for strat, counts in strats.items()}
                for role, strats in self.items()}

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
        return 'PureProfile' + super().__repr__()[12:]


class MixedProfile(frozendict):
    '''A mixed profile is distribution over strategies for each role.

    This is an immutable container that maps roles to strategies to
    probabilities. Only strategies with support are represented.

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(((r, frozendict(p)) for r, p
                          in dict(*args, **kwargs).items()))

    def support(self):
        '''Returns the support of this mixed profile

        The support is a dict mapping roles to strategies.

        '''
        return {role: set(strats) for role, strats in self.items()}

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
        return MixedProfile(json_['data'])

    def __repr__(self):
        return 'MixedProfile' + super().__repr__()[12:]
