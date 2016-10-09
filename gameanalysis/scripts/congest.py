"""create a congestion game"""
import argparse
import json
import sys
from os import path

from gameanalysis import gamegen


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Create a
                                 compact congestion game.""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('num_players', metavar='<num-players>', type=int,
                    help="""The number of players in the congestion game.""")
PARSER.add_argument('num_required', metavar='<num-required>', type=int,
                    help="""The number of facilities a player has to occupy in
                    the congestion game.""")
PARSER.add_argument('num_facilities', metavar='<num-facilities>', type=int,
                    help="""The number of facilities in the congestion
                    game.""")


def main():
    args = PARSER.parse_args()
    game = gamegen.congestion_game(args.num_players, args.num_required,
                                   args.num_facilities)
    json.dump(game.to_json(), args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
