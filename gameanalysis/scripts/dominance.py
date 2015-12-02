"""Module for finding dominated strategies"""
import json

from gameanalysis import rsgame
from gameanalysis import dominance


_CRITERIA = {
    'psd': dominance.pure_strategy_dominance,
    'nbr': dominance.never_best_response
}

_MISSING = {
    'uncond': 0,
    'cond': 1,
    'conservative': 2
}


def update_parser(parser):
    parser.description="""Compute dominated strategies, or subgames with only
    undominated strategies."""
    parser.add_argument('--format', '-f', choices=('game', 'strategies'),
            default='game', help="""Output formats: game = outputs a JSON
            representation of the game after IEDS; strategies = outputs a
            mapping of roles to eliminated strategies.  (default:
            %(default)s)""")
    parser.add_argument('--criterion', '-c', default='psd', choices=_CRITERIA,
            help="""Dominance criterion: psd = pure-strategy dominance; nbr =
            never-best-response. (default: %(default)s)""")
    parser.add_argument('--missing', '-m', choices=_MISSING, default='cond',
            help="""Method to handle missing data: uncond = unconditional
            dominance; cond = conditional dominance; conservative =
            conservative. (default: %(default)s)""")
    parser.add_argument('--weak', '-w', action='store_true', help="""If set,
    strategies are eliminated even if they are only weakly dominated.""")


def main(args):
    game = rsgame.Game.from_json(json.load(args.input))
    sub = dominance.iterated_elimination(game, _CRITERIA[args.criterion],
            conditional=_MISSING[args.missing])

    if args.format == 'strategies':
        sub = {role: sorted(strats.difference(sub.strategies[role]))
                   for role, strats in game.strategies.items()}

    json.dump(sub, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
