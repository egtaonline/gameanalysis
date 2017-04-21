"""perform player reduction on games"""
import argparse
import json
import sys

import numpy as np

from gameanalysis import gameio
from gameanalysis import reduction


def parse_sorted(red, serial):
    """Parser reduction input for roles in sorted order"""
    players = red.split(',')
    assert len(players) == serial.num_roles, \
        'Must input a reduced count for every role'
    return np.fromiter(map(int, players), int, len(players))


def parse_inorder(red, serial):
    """Parser input for role number pairs"""
    players = red.split(',')
    assert len(players) == serial.num_roles, \
        'Must input a reduced count for every role'
    red_players = np.zeros(serial.num_roles, int)
    for p in players:
        s, c = p.split(':')
        red_players[serial.role_index(s)] = int(c)
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


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'reduce', aliases=['red'], help="""Reduce games""",
        description="""Create reduced game files from input game files.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        '--type', '-t', choices=REDUCTIONS, default='dpr', help="""Type of
        reduction to perform. `dpr` - deviation preserving. `hr` -
        hierarchical. `tr` - twins. `idr` - identity. (default:
        %(default)s)""")
    parser.add_argument(
        '--sample-game', '-m', action='store_true', help="""If set, interprets
        the game as a sample game instead of a normal game.""")
    parser.add_argument(
        '--sorted-roles', '-s', action='store_true', help="""If set, reduction
        should be a comma separated list of reduced counts for the role names
        in sorted order.""")
    parser.add_argument(
        '--allow-incomplete', '-a', action='store_true', help="""If set,
        incomplete profiles will be kept in the reduced game. Currently this is
        only relevant to DPR.""")
    parser.add_argument(
        'reduction', nargs='?', metavar='<role>:<count>[,<role>:<count>]...',
        help="""Number of players in each reduced-game role.  This is a string
        e.g. 'role1:4,role2:2'""")
    return parser


def main(args):
    read = gameio.read_samplegame if args.sample_game else gameio.read_game
    game, serial = read(json.load(args.input))
    reduced_players = (
        None if not args.reduction
        else PLAYERS[args.sorted_roles](args.reduction, serial))

    reduced = REDUCTIONS[args.type](
        game.num_strategies, game.num_players,
        reduced_players).reduce_game(game, args.allow_incomplete)

    json.dump(serial.to_game_json(reduced), args.output)
    args.output.write('\n')
