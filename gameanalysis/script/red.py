"""perform player reduction on games"""
import argparse
import json
import sys

import numpy as np

from gameanalysis import gamereader
from gameanalysis.reduction import deviation_preserving as dpr
from gameanalysis.reduction import hierarchical as hr
from gameanalysis.reduction import identity as idr
from gameanalysis.reduction import twins as tr


def parse_sorted(red, game):
    """Parser reduction input for roles in sorted order"""
    players = red.split(',')
    assert len(players) == game.num_roles, \
        'Must input a reduced count for every role'
    return np.fromiter(map(int, players), int, len(players))


def parse_inorder(red, game):
    """Parser input for role number pairs"""
    players = red.split(',')
    assert len(players) == game.num_roles, \
        'Must input a reduced count for every role'
    red_players = np.zeros(game.num_roles, int)
    for p in players:
        s, c = p.split(':')
        red_players[game.role_index(s)] = int(c)
    return red_players


PLAYERS = {
    True: parse_sorted,
    False: parse_inorder,
}

REDUCTIONS = {
    'dpr': dpr,
    'hr': hr,
    'tr': tr,
    'idr': idr,
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
        '--sorted-roles', '-s', action='store_true', help="""If set, reduction
        should be a comma separated list of reduced counts for the role names
        in sorted order.""")
    # TODO Want metavar to be '<role>:<count>[,<role>:<count>]...', but
    # argparse doesn't allow [] in metavars
    parser.add_argument(
        'reduction', nargs='?', metavar='<role>:<count>,...',
        help="""Number of players in each reduced-game role.  This is a string
        e.g. 'role1:4,role2:2'""")
    return parser


def main(args):
    game = gamereader.read(json.load(args.input))
    reduced_players = (
        None if not args.reduction
        else PLAYERS[args.sorted_roles](args.reduction, game))

    reduced = REDUCTIONS[args.type].reduce_game(game, reduced_players)
    json.dump(reduced.to_json(), args.output)
    args.output.write('\n')
