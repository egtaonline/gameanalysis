import argparse
import bisect
import itertools

from gameanalysis import rsgame


def pure_subgames(game):
    '''Returns a generator of every pure subgame in a game

    A pure subgame is a subgame where each role only has one strategy. This
    returns the pure subgames in sorted order based off of role and strategy
    names.

    '''
    return (EmptySubgame(game, dict(rs)) for rs in itertools.product(
        *([(r, {s}) for s in sorted(ss)] for r, ss
          in game.strategies.items())))


def _strategies_to_set(strategies):
    '''Converts a strategies dict to a set of role strategy tuples'''
    return frozenset(itertools.chain.from_iterable(
        ((r, s) for s in ses) for r, ses in strategies.items()))


def _extract_profiles(game, strategies):
    '''Given a game and a reduced set of strategies, create a general profile
    structure to feed into a game initialization

    '''
    subgame_set = _strategies_to_set(strategies)
    for prof in game:
        if not _strategies_to_set(prof).issubset(subgame_set):
            continue  # Profile not in subgame
        payoffs = game.get_payoffs(prof)
        yield {role: [(strat, prof[role][strat], (payoff,))
                      for strat, payoff in strat_payoffs.items()]
               for role, strat_payoffs in payoffs.items()}


class EmptySubgame(rsgame.EmptyGame):
    '''A subgame corresponding to an empty game

    empty_game is the full game that this is a subgame of. strategies is a
    reduced role-strategy mapping for the subgame.

    This class provides methods that don't require payoff data.

    '''
    def __init__(self, game, strategies):
        super().__init__(game.players, strategies)
        self.full_game = game

    def deviation_profiles(self):
        '''Return a generator of every deviation profile, the role, and deviation

        '''
        for role, strats in self.strategies.items():
            nd_players = dict(self.players)
            nd_players[role] -= 1
            nd_game = rsgame.EmptyGame(nd_players, self.strategies)
            for prof in nd_game.all_profiles():
                for dev in self.full_game.strategies[role] - strats:
                    yield prof.add(role, dev), role, dev

    def additional_strategy_profiles(self, role, strat):
        '''Returns a generator of all additional profiles that exist in the subgame
        with strat

        '''
        # This uses the observation that the added profiles are all of the
        # profiles of the new subgame with one less player in role, and then
        # where that last player always plays strat
        new_players = dict(self.players)
        new_players[role] -= 1
        new_strats = self.add_strategy(role, strat).strategies
        new_game = rsgame.EmptyGame(new_players, new_strats)
        return (p.add(role, strat) for p in new_game.all_profiles())

    def add_strategy(self, role, strategy):
        '''Returns a subgame with the additional strategy'''
        strats = dict(self.strategies)
        strats[role] = list(strats[role]) + [strategy]
        return EmptySubgame(self.full_game, strats)

    def create_game(self):
        '''Returns a new game that only has data for profiles in this subgame

        '''
        return rsgame.Game(self.players, self.strategies,
                           _extract_profiles(self.full_game, self.strategies))

# XXX I wanted to make a multiple inheritance subgame object that behaved like
# a game, but also had data. I couldn't get this to work with pythons multiple
# inheritance. It would have been a diamond pattern, which isn't great. I'm not
# sure what the best approach is, but given that the data basically has to be
# copied for efficiency, I just made an explicit call in EmptySubgame.

# def translate(arr, source_game, target_game):
#     '''
#     Translates a mixture, profile, count, or payoff array between related
#     games based on role/strategy indices.

#     Useful for testing full-game regret of subgame equilibria.
#     '''
#     a = target_game.zeros()
#     for role in target_game.roles:
#         for strategy in source_game.strategies[role]:
#             a[target_game.index(role), target_game.index(role, strategy)] = \
#                     arr[source_game.index(role), source_game.index(role, \
#                     strategy)]
#     return a


# def subgame(game, strategies={}, require_all=False):
#     '''
#     Creates a game with a subset each role's strategies.

#     default settings result in a subgame with no strategies
#     '''
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
#                 game.sample_values[game[prof]][game.index(role), game.index( \
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
    '''Returns a generator over all maximally complete subgames

    The subgames returned are empty subgames.

    '''
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
            as_set = _strategies_to_set(sub.strategies)
            if not any(as_set.issubset(max_sub) for max_sub in maximals):
                maximals.append(as_set)
                yield sub

###################
# Subgame Command #
###################

_PARSER = argparse.ArgumentParser(description='''Detect all complete subgames
in a partial game or extract specific subgames.''')
_PARSER.add_argument('mode', choices=('detect', 'extract'), help='''If mode is
set to detect, all complete subgames will be found, and the output will be a
JSON list of role:[strategies] maps enumerating the complete subgames. If mode
is set to extract, then the output will be a JSON representation of a game or a
list of games with the specified subsets of strategies.''')
_PARSER.add_argument('--full', action='store_true', help='''In 'detect' mode:
setting this flag causes the script to output games instead of role:strategy
maps.''')
_PARSER.add_argument('-f', metavar='strategies file', type=str, default='',
                     help='''In 'extract' mode: JSON file with
                     role:[strategies] map(s) of subgame(s) to extract. The
                     file should have the same format as the output of detect
                     mode (or to extract just one subgame, a single map instead
                     of a list of them).''')
_PARSER.add_argument('-s', type=int, nargs='*', default=[], help='''In
'extract' mode: a list of strategy indices to extract. A strategy is specified
by its zero-indexed position in a list of all strategies sorted alphabetically
by role and sub-sorted alphabetically by strategy name. For example if role r1
has strategies s1,s2,s2 and role r2 has strategies s1,s2, then the subgame with
all but the last strategy for each role is extracted by './Subgames.py extract
-s 0 1 3'. Ignored if -f is also specified.''')


def command(args, prog, print_help=False):
    _PARSER.prog = '%s %s' % (_PARSER.prog, prog)
    args = _PARSER.parse_args(args)

    if args.mode == 'detect':
        if args.k != '':
            known = read(args.k)
        else:
            known = []
        subgames = cliques(game, known)
        if args.full:
            subgames = [subgame(game,s) for s in subgames]
    else:
        if args.f != '':
            strategies = read(args.f)
        elif len(args.s) > 0:
            strategies = {r:[] for r in game.roles}
            l = 0
            i = 0
            for r in game.roles:
                while i < len(args.s) and args.s[i] < l + \
                                    len(game.strategies[r]):
                    strategies[r].append(game.strategies[r][args.s[i]-l])
                    i += 1
                l += len(game.strategies[r])
            strategies = [strategies]
        else:
            raise IOError('Please specify either -f or -s for extract mode.')
        subgames = [subgame(game, s) for s in strategies]
        if len(subgames) == 1:
            subgames = subgames[0]

    #print to_JSON_str(subgames)
