"""Methods for interacting with mappings from role to strategy to value"""
import collections
import itertools
import numpy as np

from gameanalysis import collect


class Support(collect.frozendict):
    """A static assignment of roles to strategies"""
    # TODO implement
    pass


class _RoleStratMap(collect.frozendict):
    """An abstract class that provides common methods between Profiles and Mixtures

    """
    def __init__(self, *args, **kwargs):
        super().__init__(((r, collect.frozendict(p)) for r, p
                          in dict(*args, **kwargs).items()))

    def support(self):
        """Returns the support of this mixed profile

        The support is a dict mapping roles to strategies.

        """
        # TODO make this a support object
        return {role: set(strats) for role, strats in self.items()}

    def to_json(self):
        """Return a representation that is json serializable"""
        return {r: dict(s) for r, s in self.items()}


class Profile(_RoleStratMap):
    """A static assignment of players to roles and strategies

    This is an immutable container that maps roles to strategies to
    counts. Only strategies with at least one player playing them are
    represented.

    """

    def remove(self, role, strategy):
        """Return a new profile with one less player playing strategy"""
        copy = dict(self)
        role_copy = dict(copy[role])
        copy[role] = role_copy
        role_copy[strategy] -= 1
        if role_copy[strategy] == 0:
            role_copy.pop(strategy)
        return Profile(copy)

    def add(self, role, strategy):
        """Return a new profile where strategy has one more player"""
        copy = dict(self)
        role_copy = collections.Counter(copy[role])
        copy[role] = role_copy
        role_copy[strategy] += 1
        return Profile(copy)

    def deviate(self, role, strategy, deviation):
        """Returns a new profile where one player deviated"""
        copy = dict(self)
        role_copy = collections.Counter(copy[role])
        copy[role] = role_copy
        role_copy[strategy] -= 1
        role_copy[deviation] += 1
        if role_copy[strategy] == 0:
            role_copy.pop(strategy)
        return Profile(copy)

    def to_input_profile(self, payoff_map):
        """Given a payoff map, which maps role to strategy to payoffs, return an input
        profile for game construction

        This requires that the payoff map contains data for every role and
        strategy in the profile. An input profile looks like {role: [(strat,
        count, payoffs)]}, and is necessary to construct a game object."""
        # TODO It's unclear if the dictionary should map to a set or not, or
        # how deduping should be done. It is clear that that shouldn't be two
        # entries with the same strategy.
        return {role: [(strat, counts, payoff_map[role][strat])
                       for strat, counts in strats.items()]
                for role, strats in self.items()}

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        return Profile(json_)

    @staticmethod
    def from_symmetry_groups(sym_groups):
        """Load a profile from a symmetry group representation

        A symmetry group representation is:
        [{role: <role>, strategy: <strategy>, count: <count>}].
        """
        base_dict = {}
        for sym_group in sym_groups:
            strats = base_dict.setdefault(sym_group['role'], {})
            strats[sym_group['strategy']] = sym_group['count']
        return Profile(base_dict)

    def to_symmetry_groups(self):
        """Convert profile to symmetry groups representation"""
        return list(itertools.chain.from_iterable(
            ({'role': r, 'strategy': s, 'count': c} for s, c in cs.iteritems())
            for r, cs in self.iteritems()))

    def __str__(self):
        return '; '.join('{0}: {1}'.format
                         (role, ', '.join('{0:d} {1}'.format(count, strat)
                                          for strat, count in strats.items()))
                         for role, strats in self.items())


class Mixture(_RoleStratMap):
    """A mixed profile is distribution over strategies for each role.

    This is an immutable container that maps roles to strategies to
    probabilities. Only strategies with support are represented.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(((r, collect.frozendict(p)) for r, p
                          in dict(*args, **kwargs).items()))

    def trim_support(self, supp_thresh=1e-3):
        """Returns a new mixed profiles without strategies played less than
        supp_thresh

        """
        def process_roles():
            for role, strats in self.items():
                new_strats = [(strat, prob) for strat, prob in strats.items()
                              if prob >= supp_thresh]
                total_prob = sum(prob for _, prob in new_strats)
                yield role, {strat: p / total_prob for strat, p in new_strats}
        return Mixture(process_roles())

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        return Mixture(json_)


def trim_mixture_array_support(mixture, supp_thresh=1e-3):
    """Trims strategies played less than supp_thresh from the support"""
    mixture *= mixture >= supp_thresh
    mixture /= mixture.sum(1)[:, np.newaxis]
    return mixture
