"""Module for converting old data into new format"""
import json

from gameanalysis import rsgame


_OUTPUT_TYPE = {
    'json': lambda game, out: json.dump(game.to_json(), out)
}


def update_parser(parser):
    parser.description = """Converts between game data formats. Currently this
    is only useful for modernizing old game json formats."""
    parser.add_argument('--format', '-f', choices=_OUTPUT_TYPE, default='json',
            help="""Output format. (default: %(default)s)""")


def main(args):
    game = rsgame.Game.from_json(json.load(args.input))
    _OUTPUT_TYPE[args.format](game, args.output)
