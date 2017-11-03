"""sample profiles from a mixture"""
import argparse
import json
import sys

import numpy.random as rand

from gameanalysis import gamereader
from gameanalysis import rsgame


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'sample', aliases=['samp'], help="""Sample profiles from a mixture.
        This returns each profile on a new line, allowing streaming simulation
        of each profile.""",
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
    parser.add_argument(
        '--seed', metavar='<int>', type=int, help="""Set the seed of the random
        number generator to get consistent output""")
    return parser


def main(args):
    game = rsgame.emptygame_copy(gamereader.read(json.load(args.input)))
    mix = game.from_mix_json(json.load(args.mix))

    if args.seed:
        rand.seed(args.seed)

    for prof in game.random_profiles(args.num, mix):
        # We sort the keys when a seed is set to guarantee identical output.
        # This technically shouldn't be necessary, but on the off chance that a
        # simulator depends on the order, we want to make sure we produce
        # identical results.
        json.dump(game.to_prof_json(prof), args.output,
                  sort_keys=args.seed is not None)
        args.output.write('\n')
