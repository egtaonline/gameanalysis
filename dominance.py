#!/usr/bin/env python3
"""Module for finding dominated strategies"""
import sys

if not hasattr(sys, 'real_prefix'):
    sys.stderr.write('Could not detect virtualenv. Make sure that you\'ve '
                     'activated the virtual env\n(`. bin/activate`).\n')
    sys.exit(1)

import argparse
import json

from gameanalysis import rsgame, dominance

_CRITERIA = {
    'psd': dominance.pure_strategy_dominance,
    'nbr': dominance.never_best_response}

_MISSING = {
    'uncond': 0,
    'cond': 1,
    'conservative': 2}

_PARSER = argparse.ArgumentParser(description='''Compute dominated strategies,
or subgames''')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='file', default=sys.stdout,
                     type=argparse.FileType('w'), help='''Output dominance
                     file. The contents depend on the format specified.
                     (default: stdout)''')
_PARSER.add_argument('--format', '-f', choices=('game', 'strategies'),
                     default='game', help='''Output formats: game = outputs a
                     JSON representation of the game after IEDS; strategies =
                     outputs a mapping of roles to eliminated strategies.
                     (default: %(default)s)''')
_PARSER.add_argument('--criterion', '-c', default='psd', choices=_CRITERIA,
                     help='''Dominance criterion: psd = pure-strategy
                     dominance; nbr = never-best-response. (default:
                     %(default)s)''')
_PARSER.add_argument('--missing', '-m', choices=_MISSING, default='cond',
                     help='''Method to handle missing data: uncond =
                     unconditional dominance; cond = conditional dominance;
                     conservative = conservative. (default: %(default)s)''')
_PARSER.add_argument('--weak', '-w', action='store_true', help='''If set,
strategies are eliminated even if they are only weakly dominated.''')


def main():
    args = _PARSER.parse_args()
    game = rsgame.Game.from_json(json.load(args.input))
    sub = dominance.iterated_elimination(game, _CRITERIA[args.criterion],
                                         conditional=_MISSING[args.missing])

    if args.format == 'strategies':
        subgame = {role: sorted(strats.difference(sub.strategies[role]))
                   for role, strats in game.strategies.items()}

    json.dump(subgame, args.output, default=lambda x: x.to_json())
    args.output.write('\n')


if __name__ == '__main__':
    main()
else:
    raise ImportError('This module should not be imported')
