"""Module for constructing game models by Gaussian process regression."""
import argparse
import json
import pickle
import sys

from gameanalysis import gpgame
from gameanalysis import nash


def add_parser(subparsers):
    parser = subparsers.add_parser('gpnash', help="""Compute nash of gp
                                   games""", description="""Find nash of a gp
                                   model""")
    parser.add_argument('--input', '-i', metavar='<input-file>',
                        default=sys.stdin.buffer, type=argparse.FileType('rb'),
                        help="""Input file for script.  (default: stdin)""")
    parser.add_argument('--output', '-o', metavar='<output-file>',
                        default=sys.stdout, type=argparse.FileType('w'),
                        help="""Output file for script. (default: stdout)""")
    return parser


def main(args):
    game = gpgame.PointGPGame(pickle.load(args.input))
    serial = pickle.load(args.input)
    equilibria = nash.mixed_nash(game, replicator=None)
    json.dump([serial.to_prof_json(eqm) for eqm in equilibria], args.output)
    args.output.write('\n')
