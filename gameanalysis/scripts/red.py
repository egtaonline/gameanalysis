"""perform player reduction on games"""
import argparse
import json
import sys
from os import path

import numpy as np

from gameanalysis import gameio
from gameanalysis import reduction


def parse_sorted(players, serial):
    """Parser reduction input for roles in sorted order"""
    assert len(players) == serial.num_roles, \
        'Must input a reduced count for every role'
    return players


def parse_inorder(players, serial):
    """Parser input for role number pairs"""
    assert len(players) == 2 * serial.num_roles, \
        'Must input a reduced count for every role'
    red_players = np.zeros(serial.num_roles, int)
    for s, n in zip(players[::2], map(int, players[1::2])):
        red_players[serial.role_index(s)] = n
    return red_players


PLAYERS = {
    True: parse_sorted,
    False: parse_inorder,
}

REDUCTIONS = {
    'dpr': reduction.DeviationPreserving,
    'hr': reduction.Hierarchical,
    'tr': lambda s, f, r: reduction.Twins(s, f),
    'idr': lambda s, f, r: reduction.Identity(s, f),
}

PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Create
                                 reduced game files from input game files.""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input file for script.
                    (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('--type', '-t', choices=REDUCTIONS, default='dpr',
                    help="""Type of reduction to perform. (default:
                    %(default)s)""")
PARSER.add_argument('--sample-game', '-m', action='store_true', help="""If set,
                    interprets the game as a sample game instead of a normal
                    game.""")
PARSER.add_argument('--sorted-roles', '-s', action='store_true', help="""If
                    set, players should be a list of reduced counts for the
                    role names in sorted order.""")
PARSER.add_argument('players', nargs='*', metavar='<role-or-count>',
                    help="""Number of players in each reduced-game role.  This
                    should be a list of role then counts e.g. 'role1 4 role2
                    2'""")


def main():
    args = PARSER.parse_args()
    read = gameio.read_sample_game if args.sample_game else gameio.read_game
    game, serial = read(json.load(args.input))
    reduced_players = (
        None if not args.players
        else PLAYERS[args.sorted_roles](args.players, serial))

    reduced = REDUCTIONS[args.type](game.num_strategies, game.num_players,
                                    reduced_players).reduce_game(game)

    json.dump(serial.to_json(reduced), args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
