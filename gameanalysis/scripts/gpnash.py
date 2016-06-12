"""Module for constructing game models by Gaussian process regression."""
import argparse
import json
import pickle
import sys
from os import path

from gameanalysis import gpgame
from gameanalysis import nash


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Find nash
                                 of a gp model""")
PARSER.add_argument('--input', '-i', metavar='<input-file>',
                    default=sys.stdin.buffer, type=argparse.FileType('rb'),
                    help="""Input file for script.  (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")


def main():
    args = PARSER.parse_args()
    game = gpgame.PointGPGame(pickle.load(args.input))
    serial = pickle.load(args.input)
    equilibria = nash.mixed_nash(game, replicator=None)
    json.dump([serial.to_prof_json(eqm) for eqm in equilibria], args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
