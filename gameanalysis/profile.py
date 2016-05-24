"""Methods for interacting with mappings from role to strategy to value"""
import collections
import itertools
from collections import abc

import numpy as np

from gameanalysis import collect


class _RoleStratMap(collect.frozendict):
    """An abstract class that provides common methods between Profiles and Mixtures

    """
    def __init__(self, *args, **kwargs):
        super().__init__(((r, collect.frozendict(p)) for r, p
                          in dict(*args, **kwargs).items()))

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

    def is_valid(self, players, strategies):
        """Determines if a profile is valid

        Arguments
        ---------
        players : {role: int}
            Mapping of role to count playing role.
        strategies : {role: [strat]}
            Mapping of role to valid strategies.
        """
        for role, strat_counts in self.items():
            if players[role] != sum(strat_counts.values()):
                return False
            if not strat_counts.keys() <= set(strategies[role]):
                return False
        return True

    def to_input_profile(self, payoff_map):
        """Given a payoff map, which maps role to strategy to payoffs, return an input
        profile for game construction

        This requires that the payoff map contains data for every role and
        strategy in the profile. An input profile looks like {role: [(strat,
        count, payoffs)]}, and is necessary to construct a game object."""
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

    @staticmethod
    def from_input_profile(input_prof):
        return Profile((role, {s: c for s, c, _ in dats})
                       for role, dats in input_prof.items())

    @staticmethod
    def from_profile_string(prof_str):
        """Load a profile from a profile string representation"""
        return Profile(((role,
                         ((strat, int(count)) for count, strat
                          in (y.split(' ', 1) for y in rest.split(', '))))  # noqa
                        for role, rest
                        in (x.split(': ', 1) for x in prof_str.split('; '))))

    def to_symmetry_groups(self):
        """Convert profile to symmetry groups representation"""
        return list(itertools.chain.from_iterable(
            ({'role': r, 'strategy': s, 'count': c} for s, c in cs.items())
            for r, cs in self.items()))

    def to_profile_string(self):
        """Convert a profile to its egta string representation"""
        return '; '.join('{0}: {1}'.format
                         (role, ', '.join('{0:d} {1}'.format(count, strat)
                                          for strat, count in strats.items()))
                         for role, strats in self.items())

    def __str__(self):
        return self.to_profile_string()


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

    def is_valid(self, strategies, tolerance=1e-3):
        """Determines if a profile is valid

        Arguments
        ---------
        strategies : {role: [strat]}
            Mapping of role to valid strategies.
        tolerance : float
            Tolerance for considering a valid mixure. The sum of all
            probabilities for a role must be within `tolerance` of 1.
        """
        for role, strat_counts in self.items():
            if abs(1 - sum(strat_counts.values())) > tolerance:
                return False
            if not strat_counts.keys() <= set(strategies[role]):
                return False
        return True

    @staticmethod
    def from_json(json_):
        """Load a profile from its json representation"""
        return Mixture(json_)


def simplex_project(game, mixture):
    """Project an invalid mixture array onto the simplex"""
    # This is a variant on the algorithm in utils, but works for the staggered
    # mixture arrays that we use.
    sort = -mixture
    maxes = game.role_reduce(sort, ufunc=np.maximum)
    mins = game.role_reduce(sort, ufunc=np.minimum)
    offsets = np.concatenate(([0], np.cumsum(maxes[:-1] - mins[1:]))).repeat(
        game.astrategies)
    sort += offsets
    sort.sort()
    sort -= offsets
    sort *= -1

    inds = game.astrategies.cumsum()
    cum = sort.copy()
    cum[inds[:-1]] -= game.role_reduce(sort)[:-1]
    ones = np.ones_like(cum)
    ones[inds[:-1]] -= game.astrategies[:-1]
    rho = (1 - cum.cumsum()) / ones.cumsum()

    max_inds = (np.diff(np.insert(rho + sort > 0, inds,
                                  False)).nonzero()[0][::2] -
                np.arange(game.astrategies.size))
    lam = rho[max_inds].repeat(game.astrategies)
    return np.maximum(mixture + lam, 0)


def support_set(strategies):
    """Takes a support like object and returns a set representing the support

    A support like object is a dict mapping role to an iterable of
    strategies. This includes cases when the iterable of strategies is another
    mapping type to any sort of information, e.g. a profile. The support set is
    simply a set of (role, strategy) that have support in this profile type.

    """
    return frozenset(itertools.chain.from_iterable(
        ((r, s) for s in ses) for r, ses in strategies.items()))
