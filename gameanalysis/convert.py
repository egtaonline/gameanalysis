import sys
import argparse
import json

from gameanalysis import rsgame


_OUTPUT_TYPE = {
    'json': lambda game, out: json.dump(game.to_json(), out)
}

_PARSER = argparse.ArgumentParser(add_help=False, description='''Converts
between game data formats. Currently this is only useful for modernizing old
game json formats.''')
_PARSER.add_argument('--input', '-i', type=argparse.FileType('r'),
                     metavar='file', default=sys.stdin, help='''Input game
                     file. (default: stdin)''')
_PARSER.add_argument('--output', '-o', type=argparse.FileType('w'),
                     metavar='file', default=sys.stdout, help='''Output game
                     file. (default: stdout)''')
_PARSER.add_argument('--format', '-f', choices=_OUTPUT_TYPE, default='json',
                     help='''Output format. (default: %(default)s)''')


def command(args, prog, print_help=False):
    _PARSER.prog = '%s %s' % (_PARSER.prog, prog)
    if print_help:
        _PARSER.print_help()
        return
    args = _PARSER.parse_args(args)
    game = rsgame.Game.from_json(json.load(args.input))
    _OUTPUT_TYPE[args.format](game, args.output)
