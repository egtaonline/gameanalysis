"""convert old game data into current format"""
import argparse
import json
import sys
from os import path

from gameanalysis import gameio


OUTPUT_TYPE = {
    'json': lambda game, ser, out: json.dump(ser.to_json(game), out)
}

PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Converts
                                 between game data formats. Currently this is
                                 only useful for modernizing old game json
                                 formats.""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input file for script.
                    (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('--format', '-f', choices=OUTPUT_TYPE, default='json',
                    help="""Output format. (default: %(default)s)""")


def main():
    args = PARSER.parse_args()
    game, serial = gameio.read_game(json.load(args.input))
    OUTPUT_TYPE[args.format](game, serial, args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()