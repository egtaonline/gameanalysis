"""convert old game data into current format"""
import argparse
import json
import sys

from gameanalysis import gameio
from gameanalysis import rsgame


OUTPUT_TYPE = {
    'json': lambda game, ser, out: json.dump(game.to_json(ser), out)
}

GAME_TYPE = {
    'identity': lambda g: g,
    'base': rsgame.BaseGame,
    'game': rsgame.Game,
    'sample': rsgame.SampleGame,
}


def add_parser(subparsers):
    parser = subparsers.add_parser('convert', aliases=['conv'], help="""Convert
                                   between game types""",
                                   description="""Converts between game data
                                   formats. Currently this is only useful for
                                   modernizing old game json formats.""")
    parser.add_argument('--input', '-i', metavar='<input-file>',
                        default=sys.stdin, type=argparse.FileType('r'),
                        help="""Input file for script.  (default: stdin)""")
    parser.add_argument('--output', '-o', metavar='<output-file>',
                        default=sys.stdout, type=argparse.FileType('w'),
                        help="""Output file for script. (default: stdout)""")
    parser.add_argument('--type', '-t', choices=GAME_TYPE, default='identity',
                        help="""Output game type. (default: %(default)s)""")
    parser.add_argument('--format', '-f', choices=OUTPUT_TYPE, default='json',
                        help="""Output format. (default: %(default)s)""")
    return parser


def main(args):
    game, serial = gameio.read_game(json.load(args.input))
    game = GAME_TYPE[args.type](game)
    OUTPUT_TYPE[args.format](game, serial, args.output)
    args.output.write('\n')
