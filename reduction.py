#!/usr/bin/env python3
"""Module for doing player reduction on games"""
import sys

if not hasattr(sys, 'real_prefix'):
    sys.stderr.write('Could not detect virtualenv. Make sure that you\'ve '
                     'activated the virtual env\n(`. bin/activate`).\n')
    sys.exit(1)

import argparse
import json

from gameanalysis import reduction, rsgame


def _parse_sorted(players, game):
    """Parser reduction input for roles in sorted order"""
    assert len(players) == len(game.strategies), \
        'Must input a reduced count for every role'
    return dict(zip(game.strategies, map(int, players)))


def _parse_inorder(players, game):
    """Parser input for role number pairs"""
    assert len(players) == 2 * len(game.strategies), \
        'Must input a reduced count for every role'
    parsed = {}
    for i in range(0, len(players), 2):
        assert players[i] in game.strategies, \
            'role "{}" not found in game'.format(players[i])
        parsed[players[i]] = int(players[i + 1])
    return parsed


_PLAYERS = {
    True: _parse_sorted,
    False: _parse_inorder}

_REDUCTIONS = {
    'dpr': reduction.deviation_preserving_reduction,
    'hr': reduction.hierarchical_reduction,
    'tr': reduction.twins_reduction}

_PARSER = argparse.ArgumentParser(description='Create reduced games.')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='reduced-file',
                     default=sys.stdout, type=argparse.FileType('w'),
                     help='''Output equilibria file. This file will contain a
                     json list of mixed profiles. (default: stdout)''')
_PARSER.add_argument('--type', '-t', choices=_REDUCTIONS, default='dpr',
                     help='''Type of reduction to perform. (default:
                     %(default)s)''')
_PARSER.add_argument('--sorted-roles', '-s', action='store_true', help='''If
set, players should be a list of reduced counts for the role names in sorted
order.''')
_PARSER.add_argument('players', nargs='*', help='''Number of players in each
reduced-game role. This should be a list of role then counts e.g. "role1 4
role2 2"''')


def main():
    args = _PARSER.parse_args()
    game = rsgame.Game.from_json(json.load(args.input))
    players = _PLAYERS[args.sorted_roles](args.players, game)

    reduced = _REDUCTIONS[args.type](game, players)

    json.dump(reduced, args.output, default=lambda x: x.to_json())
    args.output.write('\n')


if __name__ == '__main__':
    main()
else:
    raise ImportError('This module should not be imported')
