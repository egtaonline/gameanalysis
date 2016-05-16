"""Module for performing actions on subgames

A subgame is a game with a restricted set of strategies that usually make
analysis tractable.

"""
import bisect
import collections
import itertools

import numpy as np

from gameanalysis import collect
from gameanalysis import rsgame
from gameanalysis import utils


def pure_subgames(game):
    """Returns a generator of every pure subgame in a game

    A pure subgame is a subgame where each role only has one strategy. This
    returns the pure subgames in sorted order based off of role and strategy
    names.

    """
    return (EmptySubgame(game, dict(rs)) for rs in itertools.product(
        *([(r, {s}) for s in sorted(ss)] for r, ss
          in game.strategies.items())))


def support_set(strategies):
    """Takes a support like object and returns a set representing the support

    A support like object is a dict mapping role to an iterable of
    strategies. This includes cases when the iterable of strategies is another
    mapping type to any sort of information, e.g. a profile. The support set is
    simply a set of (role, strategy) that have support in this profile type.

    """
    return frozenset(itertools.chain.from_iterable(
        ((r, s) for s in ses) for r, ses in strategies.items()))


@utils.compare_by_key(lambda sub: sub._key())
class EmptySubgame(rsgame.EmptyGame):
    """A subgame corresponding to an empty game

    empty_game is the full game that this is a subgame of. strategies is a
    reduced role-strategy mapping for the subgame.

    This class provides methods that don't require payoff data.

    These are order

    """
    def __init__(self, game, strategies):
        super().__init__(game.players,
                         collections.OrderedDict((r, strategies[r])
                                                 for r in game.players))
        self.full_game = game
        self._support_set = None

    def deviation_profiles(self):
        """Return a generator of every deviation profile, the role, and deviation

        """
        for role, strats in self.strategies.items():
            nd_players = dict(self.players)
            nd_players[role] -= 1
            nd_game = rsgame.EmptyGame(nd_players, self.strategies)
            for prof in nd_game.all_profiles():
                for dev in set(self.full_game.strategies[role]) - set(strats):
                    yield prof.add(role, dev), role, dev

    def additional_strategy_profiles(self, role, strat):
        """Returns a generator of all additional profiles that exist in the subgame
        with strat

        """
        # This uses the observation that the added profiles are all of the
        # profiles of the new subgame with one less player in role, and then
        # where that last player always plays strat
        new_players = dict(self.players)
        new_players[role] -= 1
        new_strats = self.add_strategy(role, strat).strategies
        new_game = rsgame.EmptyGame(new_players, new_strats)
        return (p.add(role, strat) for p in new_game.all_profiles())

    def add_strategy(self, role, strategy):
        """Returns a subgame with the additional strategy

        If strategy is already in subgame, returns the same subgame
        """
        if strategy in self.strategies[role]:
            return self
        else:
            strats = dict(self.strategies)
            strats[role] = tuple(strats[role]) + (strategy,)
            return EmptySubgame(self.full_game, strats)

    def support_set(self):
        if self._support_set is None:
            self._support_set = support_set(self.strategies)
        return self._support_set

    def _key(self):
        """Function that defines how to compare subgames"""
        return id(self.full_game), self.support_set()


def subgame(game, strategies):
    """Returns a new game that only has data for profiles in strategies"""
    # TODO Add support for sample game (shouldn't be hard)
    strat_set = {role: {strat: True for strat in strats}
                 for role, strats in strategies.items()}
    new_strats = collect.fodict(
        (role, tuple(strat for strat in strats if strat in strat_set[role]))
        for role, strats in game.strategies.items())

    strat_mask = game.as_array(strat_set, bool)
    prof_mask = ~np.any(game.profiles(True) * ~strat_mask, 1)

    new_counts = game.profiles(True)[prof_mask][:, strat_mask]
    new_values = game.payoffs(True)[prof_mask][:, strat_mask]

    return rsgame.Game(game.players, new_strats, new_counts, new_values)


# def translate(arr, source_game, target_game):
#     """
#     Translates a mixture, profile, count, or payoff array between related
#     games based on role/strategy indices.

#     Useful for testing full-game regret of subgame equilibria.
#     """
#     a = target_game.zeros()
#     for role in target_game.roles:
#         for strategy in source_game.strategies[role]:
#             a[target_game.index(role), target_game.index(role, strategy)] = \
#                     arr[source_game.index(role), source_game.index(role, \
#                     strategy)]
#     return a


# def subgame(game, strategies={}, require_all=False):
#     """
#     Creates a game with a subset each role's strategies.

#     default settings result in a subgame with no strategies
#     """
#     if not strategies:
#         strategies = {r:[] for r in game.roles}
#     sg = type(game)(game.roles, game.players, strategies)
#     if sg.size <= len(game):
#         for p in sg.allProfiles():
#             if p in game:
#                 add_subgame_profile(game, sg, p)
#             elif require_all:
#                 raise KeyError('Profile missing')
#     elif require_all:
#         raise KeyError('Profile missing')
#     else:
#         for p in game:
#             if is_valid_profile(sg, p):
#                 add_subgame_profile(game, sg, p)
#     return sg


# def add_subgame_profile(game, subgame, prof):
#     if isinstance(game, SampleGame):
#         subgame.addProfile({role:[PayoffData(strat, prof[role][strat], \
#                 game.sample_values[game[prof]][game.index(role), game.index( \ # noqa
#                 role, strat)]) for strat in prof[role]] for role in prof})
#     else:
#         subgame.addProfile({role:[PayoffData(strat, prof[role][strat], \
#                 game.getPayoff(prof, role, strat)) for strat in prof[role]] \
#                 for role in prof})


# def is_valid_profile(game, prof):
#     if set(prof.keys()) != set(game.roles):
#         return False
#     for role in prof:
#         for strat in prof[role]:
#             if strat not in game.strategies[role]:
#                 return False
#     return True


# def is_subgame(small_game, big_game):
#     if any((r not in big_game.roles for r in small_game.roles)):
#         return False
#     if any((small_game.players[r] != big_game.players[r] for r \
#             in small_game.roles)):
#         return False
#     for r in small_game.roles:
#         if any((s not in big_game.strategies[r] for s in \
#                 small_game.strategies[r])):
#             return False
#     return True

def maximal_subgames(game):
    """Returns a generator over all maximally complete subgames

    The subgames returned are empty subgames.

    """
    # invariant that we have data for every subgame in queue
    queue = [sub for sub in pure_subgames(game)
             if all(p in game for p in sub.all_profiles())]
    # Bisect strategies
    bsts = {role: tuple(sorted(ses)) for role, ses in game.strategies.items()}
    maximals = []
    while queue:
        sub = queue.pop()
        maximal = True
        for role, sts in sub.strategies.items():
            for dev in bsts[role][:bisect.bisect_left(bsts[role], min(sts))]:
                if all(p in game for p
                       in sub.additional_strategy_profiles(role, dev)):
                    maximal = False
                    queue.append(sub.add_strategy(role, dev))

        # This checks that no duplicates are emitted.  This algorithm will
        # always find the largest subset first, but subsequent 'maximal'
        # subsets may actually be subsets of previous maximal subsets.
        if maximal:
            as_set = support_set(sub.strategies)
            if not any(as_set.issubset(max_sub) for max_sub in maximals):
                maximals.append(as_set)
                yield sub
