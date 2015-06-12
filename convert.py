#!/usr/bin/env python3
"""Module for converting old data into new format"""
import sys

if not hasattr(sys, 'real_prefix'):
    sys.stderr.write('Could not detect virtualenv. Make sure that you\'ve '
                     'activated the virtual env\n(`. bin/activate`).\n')
    sys.exit(1)

import argparse
import json

from gameanalysis import rsgame


_OUTPUT_TYPE = {
    'json': lambda game, out: json.dump(game.to_json(), out)
}

_PARSER = argparse.ArgumentParser(description='''Converts between game data
formats. Currently this is only useful for modernizing old game json
formats.''')
_PARSER.add_argument('--input', '-i', type=argparse.FileType('r'),
                     metavar='file', default=sys.stdin, help='''Input game
                     file. (default: stdin)''')
_PARSER.add_argument('--output', '-o', type=argparse.FileType('w'),
                     metavar='file', default=sys.stdout, help='''Output game
                     file. (default: stdout)''')
_PARSER.add_argument('--format', '-f', choices=_OUTPUT_TYPE, default='json',
                     help='''Output format. (default: %(default)s)''')


def main():
    args = _PARSER.parse_args()
    game = rsgame.Game.from_json(json.load(args.input))
    _OUTPUT_TYPE[args.format](game, args.output)


if __name__ == '__main__':
    main()
else:
    raise ImportError('This module should not be imported')
