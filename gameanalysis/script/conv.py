"""convert between game types"""
import argparse
import itertools
import json
import sys

from gameanalysis import gambit
from gameanalysis import gamereader
from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import rsgame


_TYPES = {
    'emptygame': (
        ['empty'], 'Strip payoff data', """Strip all payoff data from a game
        and return only its base structure---role and strategy names and player
        counts.""",
        lambda game, out: json.dump(
            rsgame.empty_copy(game).to_json(), out)),
    'game': (
        [], 'Sparse payoff format', """Convert a game to a sparse mapping of
        profiles to their corresponding payoff data.""",
        lambda game, out: json.dump(
            paygame.game_copy(game).to_json(), out)),
    'samplegame': (
        ['samp'], 'Multiple payoffs per profile', """Convert a game to a format
        with a sparse mapping of profiles to potentially several samples of
        payoff data. There should be little reason to convert a non-samplegame
        to a samplegame as all profiles will have exactly one sample.""",
        lambda game, out: json.dump(
            paygame.samplegame_copy(game).to_json(), out)),
    'matgame': (
        ['mat'], 'Asymmetric format', """Convert a game to a compact
        representation for asymmetric games. If the input game is not
        asymmetric, role names will be duplicated and modified to allow for the
        conversion. This will only work if the input game is complete.""",
        lambda game, out: json.dump(
            matgame.matgame_copy(game).to_json(), out)),
    'norm': (
        [], 'Normalize payoffs to [0, 1]', """Modify the input game by scaling
        all payoffs to the range [0, 1]. In the rare event that a single role
        always gets the same payoffs, it's payoffs will be made zero. The game
        type is unchanged. This doesn't guarantee that the maximum and minimum
        payoffs are one and zero respectively, only that they're in the
        range.""",
        lambda game, out: json.dump(game.normalize().to_json(), out)),
    'string': (
        ['str'], 'String representation', """Convert a game to its string
        representation. This form is more human readable than the full
        serialization formats and may be more useful for 'inspecting' a
        game.""",
        lambda game, out: out.write(str(game))),
    'gambit': (
        [], 'Gambit format', """Convert a game to gambit 'nfg' format.
        Internally this first converts the game to a 'matgame', and as a
        result, the same caveats for converting to a matrix game apply
        here.""",
        gambit.dump),
}


def add_parser(subparsers):
    """Add conversion parser"""
    parser = subparsers.add_parser(
        'convert', aliases=['conv'], help="""Convert between supported game
        types""", description="""Convert one game representation into another
        using defined conversion routines.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    types = parser.add_subparsers(
        title='types', dest='type', metavar='<type>', help="""The type to
        convert input to. Available commands are:""")
    for name, (aliases, help_text, description, _) in _TYPES.items():
        types.add_parser(name, aliases=aliases, help=help_text,
                         description=description)
    return parser


def main(args):
    """Entry point for conversion"""
    game = gamereader.load(args.input)

    lookup = {}
    for key, (aliases, _, _, func) in _TYPES.items():
        for alias in itertools.chain([key], aliases):
            lookup[alias] = func

    lookup[args.type](game, args.output)
    args.output.write('\n')
