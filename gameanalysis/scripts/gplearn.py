"""Module for constructing game models by Gaussian process regression."""
import argparse
import json
import pickle
import sys
from os import path

from gameanalysis import gameio
from gameanalysis import gpgame


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Learn a
                                 gp model""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input file for script.
                    (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout.buffer, type=argparse.FileType('wb'),
                    help="""Output file for script. (default: stdout)""")


def main():
    args = PARSER.parse_args()
    data, serial = gameio.read_game(json.load(args.input))
    game = gpgame.BaseGPGame(data)
    pickle.dump(game, args.output)
    pickle.dump(serial, args.output)


if __name__ == '__main__':
    main()
