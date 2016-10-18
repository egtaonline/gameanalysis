"""create a congestion game"""
import argparse
import json
import sys

from gameanalysis import gamegen


def add_parser(subparsers):
    parser = subparsers.add_parser('congest', aliases=['cgst'], help="""Create
                                   congestion games""", description="""Create a
                                   compact congestion game.""")
    parser.add_argument('--output', '-o', metavar='<output-file>',
                        default=sys.stdout, type=argparse.FileType('w'),
                        help="""Output file for script. (default: stdout)""")
    parser.add_argument('num_players', metavar='<num-players>', type=int,
                        help="""The number of players in the congestion
                        game.""")
    parser.add_argument('num_facilities', metavar='<num-facilities>', type=int,
                        help="""The number of facilities in the congestion
                        game.""")
    parser.add_argument('num_required', metavar='<num-required>', type=int,
                        help="""The number of facilities a player has to occupy
                        in the congestion game.""")
    return parser


def main(args):
    game = gamegen.congestion_game(args.num_players, args.num_facilities,
                                   args.num_required)
    json.dump(game.to_json(), args.output)
    args.output.write('\n')
