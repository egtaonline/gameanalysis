"""find dominated strategies"""
import argparse
import json
import sys
from os import path

from gameanalysis import dominance
from gameanalysis import gameio
from gameanalysis import subgame


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Compute
                                 dominated strategies, or subgames with only
                                 undominated strategies.""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input file for script.
                    (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('--strategies', '-s', action='store_true',
                    help="""Output the remaining strategies instead of the
                    subgame after removing appropriate strategies. (default:
                    %(default)s)""")
PARSER.add_argument('--criterion', '-c', default='strictdom',
                    choices=['weakdom', 'strictdom', 'neverbr'],
                    help="""Dominance criterion: strictdom = strict
                    pure-strategy dominance; weakdom = weak pure-strategy
                    dominance; neverbr = never-best-response. (default:
                    %(default)s)""")
PARSER.add_argument('--unconditional', '-u', action='store_false', help="""If
                    specified use unconditional dominance, instead of
                    conditional dominance.""")


def main():
    args = PARSER.parse_args()
    game, serial = gameio.read_game(json.load(args.input))
    sub_mask = dominance.iterated_elimination(game, args.criterion,
                                              args.unconditional)
    if args.strategies:
        res = {r: list(s) for r, s in serial.to_prof_json(sub_mask).items()}
        json.dump(res, args.output)
    else:
        sub_game = subgame.subgame(game, sub_mask)
        sub_serial = subgame.subserializer(serial, sub_mask)
        json.dump(sub_game.to_json(sub_serial), args.output)

    args.output.write('\n')


if __name__ == '__main__':
    main()
