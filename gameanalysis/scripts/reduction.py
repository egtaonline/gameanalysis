"""Module for doing player reduction on games"""
import json

from gameanalysis import reduction
from gameanalysis import rsgame


def _parse_sorted(players, game):
    """Parser reduction input for roles in sorted order"""
    assert len(players) == len(game.strategies), \
        'Must input a reduced count for every role'
    return dict(zip(game.strategies, map(int, players)))


def _parse_inorder(players, game):
    """Parser input for role number pairs"""
    assert len(players) == 2 * len(game.strategies), \
        'Must input a reduced count for every role'
    parsed = {}
    for i in range(0, len(players), 2):
        assert players[i] in game.strategies, \
            'role "{}" not found in game'.format(players[i])
        parsed[players[i]] = int(players[i + 1])
    return parsed


_PLAYERS = {
    True: _parse_sorted,
    False: _parse_inorder
}

_REDUCTIONS = {
    'dpr': reduction.DeviationPreserving,
    'hr': reduction.Hierarchical,
    'tr': lambda f, r: reduction.Twins(f),
    'id': lambda f, r: reduction.Identity()
}


def update_parser(parser, base):
    parser.description = """Create reduced game files from input game files."""
    parser.add_argument('--type', '-t', choices=_REDUCTIONS, default='dpr',
                        help="""Type of reduction to perform. (default:
                        %(default)s)""")
    parser.add_argument('--sorted-roles', '-s', action='store_true', help="""If
                        set, players should be a list of reduced counts for the
                        role names in sorted order.""")
    parser.add_argument('players', nargs='*', metavar='<role-or-count>',
                        help="""Number of players in each reduced-game role.
                        This should be a list of role then counts e.g. 'role1 4
                        role2 2'""")


def main(args):
    game = rsgame.Game.from_json(json.load(args.input))
    reduced_players = (
        None if not args.players
        else _PLAYERS[args.sorted_roles](args.players, game))

    reduced = _REDUCTIONS[args.type](game.players, reduced_players)\
        .reduce_game(game)

    json.dump(reduced, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
