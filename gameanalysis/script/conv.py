"""convert between game types"""
import argparse
import json
import sys

from gameanalysis import gamereader
from gameanalysis import matgame
from gameanalysis import rsgame
from gameanalysis import serialize


def emptygame_json(game, serial):
    return serialize.gameserializer_copy(serial).to_json(
        rsgame.emptygame_copy(game))


def game_json(game, serial):
    return serialize.gameserializer_copy(serial).to_json(
        rsgame.game_copy(game))


def samplegame_json(game, serial):
    return serialize.samplegameserializer_copy(serial).to_json(
        rsgame.samplegame_copy(game))


def matgame_json(game, serial):
    mserial = matgame.matgameserializer_copy(serial, game.num_role_players)
    return mserial.to_json(matgame.matgame_copy(game))


_TYPES = {
    'emptygame': emptygame_json,
    'game': game_json,
    'samplegame': samplegame_json,
    'matgame': matgame_json,
}


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'convert', aliases=['conv'], help="""Convert between supported game
        types""", description="""Convert one game representation into another
        using defined conversion routines.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        '--type', '-t', default='game', choices=_TYPES, help="""Game type to
        convert to. (default: %(default)s)""")
    return parser


def main(args):
    game, serial = gamereader.read(json.load(args.input))
    json.dump(_TYPES[args.type](game, serial), args.output)
    args.output.write('\n')
