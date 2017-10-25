"""sample profiles from a mixture"""
import argparse
import json
import sys

from gameanalysis import gamereader
from gameanalysis import rsgame


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'sample', aliases=['samp'], help="""Sample profiles from a mixture""",
        description="""Sample profiles from a mixture.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Game file to draw samples from.
        (default: stdin)""")
    parser.add_argument(
        '--mix', '-m', metavar='<mixture-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Mixture to sample profiles from.
        (default: stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""File to write stream of profiles
        too. (default: stdout)""")
    parser.add_argument(
        '--num', '-n', metavar='<num-samples>', default=1, type=int,
        help="""The number of samples to gather.  (default: %(default)d)""")
    # TODO add an option for classic profile?
    return parser


def main(args):
    game = rsgame.emptygame_copy(gamereader.read(json.load(args.input)))
    mix = game.from_mix_json(json.load(args.mix))

    for prof in game.random_profiles(args.num, mix):
        json.dump(game.to_prof_json(prof), args.output)
        args.output.write('\n')
