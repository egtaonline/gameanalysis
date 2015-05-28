'''Module for performing actions on subgames

A subgame is a game with a restricted set of strategies that usually make
analysis tractable.

'''
import sys
import argparse
import bisect
import json
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


# TODO This returns a set over the support of a profile that can be useful for
# making comparisons between supports. This feels like it belongs in a class
# instead of as a function, but the class would basically have this one
# function. It could be tied in with EmptySubgame, but that seems verbose and
# somewhat over designed.
def support_set(strategies):
    '''Takes a support like object and returns a set representing the support

    A support like object is a dict mapping role to an iterable of
    strategies. This includes cases when the iterable of strategies is another
    mapping type to any sort of information, e.g. a profile. The support set is
    simply a set of (role, strategy) that have support in this profile type.

    '''
    return frozenset(itertools.chain.from_iterable(
        ((r, s) for s in ses) for r, ses in strategies.items()))


def _extract_profiles(game, strategies):
    '''Given a game and a reduced set of strategies, create a general profile
    structure to feed into a game initialization

    '''
    subgame_set = support_set(strategies)
    for prof in game:
        if not support_set(prof).issubset(subgame_set):
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


def subgame(game, strategies):
    '''Returns a new game that only has data for profiles in strategies'''
    return rsgame.Game(game.players, strategies,
                       _extract_profiles(game, strategies))


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
            as_set = support_set(sub.strategies)
            if not any(as_set.issubset(max_sub) for max_sub in maximals):
                maximals.append(as_set)
                yield sub

###################
# Subgame Command #
###################


def _parse_text_spec(game, spec):
    subgame = {}
    current_role = '<undefined role>'
    for role_strat in spec:
        if role_strat in game.strategies:
            current_role = role_strat
        elif role_strat in game.strategies[current_role]:
            subgame.setdefault(current_role, set()).add(role_strat)
        else:
            raise ValueError('{} was not a role or a strategy in role {}'
                             .format(role_strat, current_role))
    return EmptySubgame(game, subgame)


def _parse_index_spec(game, spec):
    index_list = sorted(support_set(game.strategies))
    subgame = {}
    for index in spec:
        role, strat = index_list[index]
        subgame.setdefault(role, set()).add(strat)
    return EmptySubgame(game, subgame)


_PARSER = argparse.ArgumentParser(add_help=False, description='''Extract
subgames and optionally detects all complete subgames. All subgame
specifications will be concatentated.''')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='subgame-file',
                     default=sys.stdout, type=argparse.FileType('w'),
                     help='''Output subgame file. This file will contain a json
                     list of subgames. (default: stdout)''')
_PARSER.add_argument('--no-extract', '-n', action='store_true', help='''Don't
extract full subgame data, just print the specifications of the subgames. This
is mainly only useful with the detect option.''')

_SUB_GROUP = _PARSER.add_argument_group(title='subgame specifications',
                                        description='''These are all of the
                                        ways to specify subgames to
                                        extract. All of these specifications
                                        are concatentated together before being
                                        output.''')
_SUB_GROUP.add_argument('--detect', '-d', action='store_true', help='''Run
clique finding to detect maximally complete subgames.''')
_SUB_GROUP.add_argument('--subgame-file', '-f', metavar='file', default=[],
                        type=argparse.FileType('r'), action='append', help='''A
                        file that contains a list of subgames. The same format
                        that can be output by this script with the no-extract
                        option. This can be specified multiple times.''')
_SUB_GROUP.add_argument('--text-spec', '-t', nargs='+', metavar='role-strat',
                        default=[], action='append', help='''Specify a subgame
                        as a list of roles and strategies. To specify the
                        subgame where role0 has strategies strat0 and strat2
                        and role1 has strategy strat1 enter "role0 strat0
                        strat2 role1 strat1". This option is ambiguous if
                        strategies share names with roles. For unambiguous
                        specification, use index specification. This can be
                        entered several times in order to specify several
                        subgames.''')
_SUB_GROUP.add_argument('--index-spec', '-s', type=int, nargs='+', default=[],
                        metavar='strat-index', action='append', help='''Specify
                        a subgame with a list of strategy indices. A strategy
                        is specified by its zero-indexed position in a list of
                        all strategies sorted alphabetically by role and
                        sub-sorted alphabetically by strategy name. For example
                        if role1 has strategies s1, s2, and s3 and role2 has
                        strategies s4 and s5, then the subgame with all but the
                        last strategy for each role is extracted by "0 1
                        3". This can be specified multiple times for several
                        subgames.''')


def command(args, prog, print_help=False):
    _PARSER.prog = '{} {}'.format(_PARSER.prog, prog)
    if print_help:
        _PARSER.print_help()
        return
    args = _PARSER.parse_args(args)
    game = rsgame.Game.from_json(json.load(args.input))

    # Collect all subgames
    subgames = []
    if args.detect:
        subgames.extend(maximal_subgames(game))
    for sub_file in args.subgame_file:
        # This actually adds EmptyGames instead of EmptySubgames, but for our
        # use they'll function the same.
        subgames.extend(rsgame.EmptyGame.from_json(sub)
                        for sub in json.load(sub_file))
    subgames.extend(_parse_text_spec(game, spec) for spec in args.text_spec)
    subgames.extend(_parse_index_spec(game, spec) for spec in args.index_spec)

    if not args.no_extract:
        subgames = [subgame(game, sub.strategies) for sub in subgames]

    json.dump(subgames, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
