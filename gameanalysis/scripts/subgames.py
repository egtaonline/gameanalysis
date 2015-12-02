"""Script for extracting and finding subgames"""
import argparse
import json

from gameanalysis import rsgame
from gameanalysis import subgame


def _parse_text_spec(game, spec):
    subgame = {}
    current_role = '<undefined role>'
    for role_strat in spec:
        if role_strat in game.strategies:
            current_role = role_strat
        elif role_strat in game.strategies[current_role]:
            subgame.setdefault(current_role, set()).add(role_strat)
        else:
            raise ValueError('{0} was not a role or a strategy in role {1}'
                             .format(role_strat, current_role))
    return subgame.EmptySubgame(game, subgame)


def _parse_index_spec(game, spec):
    index_list = sorted(subgame.support_set(game.strategies))
    subg = {}
    for index in spec:
        role, strat = index_list[index]
        subg.setdefault(role, set()).add(strat)
    return subgame.EmptySubgame(game, subg)


def update_parser(parser):
    parser.description = """Extract subgames and optionally detects all
    complete subgames. All subgame specifications will be concatentated,
    resulting in a list of subgames."""
    parser.add_argument('--no-extract', '-n', action='store_true', help="""Don't
    extract full subgame data, just print the specifications of the subgames. This
    is mainly only useful with the detect option.""")

    sub_group = parser.add_argument_group(title='subgame specifications',
            description="""These are all of the ways to specify subgames to
            extract. All of these specifications are concatentated together
            before being output.""")
    sub_group.add_argument('--detect', '-d', action='store_true', help="""Run
    clique finding to detect maximally complete subgames.""")
    sub_group.add_argument('--subgame-file', '-f', metavar='<file>',
            default=[], type=argparse.FileType('r'), action='append', help="""A
            file that contains a list of subgames. The same format that can be
            output by this script with the no-extract option. This can be
            specified multiple times.""")
    sub_group.add_argument('--text-spec', '-t', nargs='+',
            metavar='<role-strat>', default=[], action='append',
            help="""Specify a subgame as a list of roles and strategies. To
            specify the subgame where role0 has strategies strat0 and strat2
            and role1 has strategy strat1 enter "role0 strat0 strat2 role1
            strat1". This option is ambiguous if strategies share names with
            roles. For unambiguous specification, use index specification. This
            can be entered several times in order to specify several
            subgames.""")
    sub_group.add_argument('--index-spec', '-s', type=int, nargs='+',
            default=[], metavar='<strat-index>', action='append',
            help="""Specify a subgame with a list of strategy indices. A
            strategy is specified by its zero-indexed position in a list of all
            strategies sorted alphabetically by role and sub-sorted
            alphabetically by strategy name. For example if role1 has
            strategies s1, s2, and s3 and role2 has strategies s4 and s5, then
            the subgame with all but the last strategy for each role is
            extracted by "0 1 3". This can be specified multiple times for
            several subgames.""")


def main(args):
    game = rsgame.Game.from_json(json.load(args.input))

    # Collect all subgames
    subgames = []
    if args.detect:
        subgames.extend(subgame.maximal_subgames(game))
    for sub_file in args.subgame_file:
        # This actually adds EmptyGames instead of EmptySubgames, but for our
        # use they'll function the same.
        subgames.extend(rsgame.EmptyGame.from_json(sub)
                        for sub in json.load(sub_file))
    subgames.extend(_parse_text_spec(game, spec) for spec in args.text_spec)
    subgames.extend(_parse_index_spec(game, spec) for spec in args.index_spec)

    if not args.no_extract:
        subgames = [subgame.subgame(game, sub.strategies) for sub in subgames]

    json.dump(subgames, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
