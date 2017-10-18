"""convert between game types"""
import argparse
import json
import sys

from gameanalysis import gamereader
from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import rsgame


_TYPES = {
    'emptygame': rsgame.emptygame_copy,
    'game': paygame.game_copy,
    'samplegame': paygame.samplegame_copy,
    'matgame': matgame.matgame_copy,
    'norm': lambda game: game.normalize(),
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
    game = gamereader.read(json.load(args.input))
    json.dump(_TYPES[args.type](game).to_json(), args.output)
    args.output.write('\n')
