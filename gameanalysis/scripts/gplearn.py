"""Module for constructing game models by Gaussian process regression."""
import argparse
import json
import pickle
import sys

from gameanalysis import gameio
from gameanalysis import gpgame


def add_parser(subparsers):
    parser = subparsers.add_parser('gplearn', help="""Learn a gp model""",
                                   description="""Learn a gp model""")
    parser.add_argument('--input', '-i', metavar='<input-file>',
                        default=sys.stdin, type=argparse.FileType('r'),
                        help="""Input file for script.  (default: stdin)""")
    parser.add_argument('--output', '-o', metavar='<output-file>',
                        default=sys.stdout.buffer,
                        type=argparse.FileType('wb'), help="""Output file for
                        script. (default: stdout)""")
    return parser


def main(args):
    data, serial = gameio.read_game(json.load(args.input))
    game = gpgame.BaseGPGame(data)
    pickle.dump(game, args.output)
    pickle.dump(serial, args.output)
