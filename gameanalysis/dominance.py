import sys
import argparse
import json
import itertools
import numpy as np

from gameanalysis import rsgame, regret

# from math import isinf

# from RoleSymmetricGame import Profile
# from Regret import regret
# from Subgames import subgame


def iterated_elimination(game, criterion, *args, **kwargs):
    '''Iterated elimination of dominated strategies

    input:
    criterion = function to find dominated strategies

    '''
    reduced_game = eliminate_strategies(game, criterion, *args, **kwargs)
    while len(reduced_game) < len(game):
        game = reduced_game
        reduced_game = eliminate_strategies(game, criterion, *args, **kwargs)
    return game


def eliminate_strategies(game, criterion, *args, **kwargs):
    eliminated = criterion(game, *args, **kwargs)
    return subgame(game, {r : set(game.strategies[r]) - eliminated[r] \
            for r in game.roles})


def best_responses(game, prof, role=None, strategy=None):
    '''If role is unspecified, bestResponses returns a dict mapping each role all
    of its strategy-level results. If strategy is unspecified, best_responses
    returns a dict mapping strategies to the set of best responses to the
    opponent-profile without that strategy.

    '''
    if role == None:
        return {r: best_responses(game, prof, r, strategy) for r \
                in game.roles}
    if strategy == None and isinstance(prof, Profile):
        return {s: best_responses(game, prof, role, s) for s in \
                prof[role]}
    best_deviations = set()
    biggest_gain = -np.inf
    unknown = set()
    for dev in game.strategies[role]:
        reg = regret(game, prof, role, strategy, dev)
        if isinf(reg):
            unknown.add(dev)
        elif reg > biggest_gain:
            best_deviations = {dev}
            biggest_gain = reg
        elif reg == biggest_gain:
            best_deviations.add(dev)
    return best_deviations, unknown


def never_best_response(game, conditional=True):
    '''Never-a-weak-best-response criterion for IEDS

    This criterion is very strong: it can eliminate strict Nash equilibria.

    '''
    non_best_responses = {role: set(strats) for role, strats
                          in game.strategies.items()}
    for prof in game:
        for r in game.roles:
            for s in prof[r]:
                br, unknown = best_responses(game, prof, r, s)
                non_best_responses[r] -= set(br)
                if conditional:
                    non_best_responses[r] -= unknown
    return non_best_responses


def pure_strategy_dominance(game, conditional=1, weak=False):
    '''Pure-strategy dominance criterion for IEDS

    conditional:
        0: unconditional dominance
        1: conditional dominance
        2: extra-conservative conditional dominance

    '''
    dominated_strategies = {r: set() for r in game.strategies}
    for role, strats in game.strategies.items():
        for dominant, dominated in itertools.product(strats, repeat=2):
            if dominant == dominated or \
               dominated in dominated_strategies[role] or \
               dominant in dominated_strategies[role]:
                continue
            if dominates(game, role, dominant, dominated, conditional, weak):
                dominated_strategies[role].add(dominated)
    return dominated_strategies


def dominates(game, role, dominant, dominated, conditional=True, weak=False):
    dominance_observed = False
    for prof in game:
        if dominated in prof[role]:
            reg = regret(game, prof, role, dominated, dominant)
            if reg > 0 and not isinf(reg):
                dominance_observed = True
            elif (reg < 0) or (reg == 0 and not weak) or \
                    (isinf(reg) and conditional):
                return False
        elif conditional > 1 and dominant in prof[role] and \
                (prof.deviate(role, dominant, dominated) not in game):
                return False
    return dominance_observed


_CRITERIA = {
        'psd': pure_strategy_dominance,
        'nbr': never_best_response}

_MISSING = {
        'uncond': 0,
        'cond': 1,
        'conservative': 2}

_PARSER = argparse.ArgumentParser(add_help=False, description='')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='output-file',
                     default=sys.stdout, type=argparse.FileType('w'),
                     help='''Output dominance file. The contents depend on the
                     format specified.  (default: stdout)''')
_PARSER.add_argument('--format', '-f', choices=('game', 'strategies'),
                     default='game', help='''Output formats: game = outputs a
                     JSON representation of the game after IEDS; strategies =
                     outputs a mapping of roles to eliminated strategies.
                     (default: %(default)s)''')
_PARSER.add_argument('--criterion', '-c', metavar='criterion', default='psd',
                     choices=_CRITERIA, help='''Dominance criterion: psd =
                     pure-strategy dominance; nbr =
                     never-best-response. (default: %(default)s)''')
_PARSER.add_argument('--missing', '-m', metavar='missing', choices=_MISSING,
                     default=1, help='''Method to handle missing data: uncond =
                     unconditional dominance; cond = conditional dominance;
                     conservative = conservative. (default: %(default)s)''')
_PARSER.add_argument('--weak', '-w', action='store_true', help='''If set,
strategies are eliminated even if they are only weakly dominated.''')


def command(args, prog, print_help=False):
    _PARSER.prog = '%s %s' % (_PARSER.prog, prog)
    if print_help:
        _PARSER.print_help()
        return
    args = _PARSER.parse_args(args)
    game = rsgame.Game.from_json(json.load(args.input))
    subgame = iterated_elimination(game, _CRITERIA[args.criterion],
                                   conditional=_MISSING[args.missing])

    if args.format == 'strategies':
        eliminated = {role: sorted(strats.difference(subgame.strategies[role]))
                      for role, strats in game.strategies.items()}
        json.dump(eliminated, args.output)
    else:
        json.dump(subgame, args.output, default=lambda x: x.to_json())
