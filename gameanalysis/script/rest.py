"""extract and find restrictions"""
import argparse
import json
import sys

import numpy as np

from gameanalysis import gamereader
from gameanalysis import restrict
from gameanalysis import utils


def add_parser(subparsers):
    """Add restriction parser"""
    parser = subparsers.add_parser(
        'restriction', aliases=['rest'], help="""Compute and select
        restrictions""", description="""Extract restricted game and optionally
        detects all complete restrictions.  All restriction specifications will
        be concatenated, resulting in a list of restrictions.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        '--no-extract', '-n', action='store_true', help="""Don't extract
        restricted games, just print the specifications of the restricted
        strategy set. This is mainly only useful with the detect option.""")

    sub_group = parser.add_argument_group(
        title='restriction specifications', description="""These are all of the
        ways to specify restricted games to extract.  All of these
        specifications are concatenated together before being output.""")
    sub_group.add_argument(
        '--detect', '-d', action='store_true', help="""Run clique finding to
        detect maximally complete restrictions.""")
    sub_group.add_argument(
        '--restriction-file', '-f', metavar='<file>', default=[],
        type=argparse.FileType('r'), action='append', help="""A file that
        contains a list of restrictions. A restriction is simply a mapping of
        roles to strategies i.e. "{r: ["s1", "s2"]}". This is the same format
        that can be output by this script with the no-extract option. This can
        be specified multiple times.""")
    sub_group.add_argument(
        '--text-spec', '-t', metavar='<role:strat,...;...>', action='append',
        default=[], help="""Specify a restrictions as a string. To specify the
        restriction where role0 has strategies strat0 and strat2 and role1 has
        strategy strat1 enter "role0:strat0,strat2;role1:strat1".""")
    sub_group.add_argument(
        '--index-spec', '-s', metavar='<i,j,...>', action='append', default=[],
        help="""Specify a restriction with a list of strategy indices. A
        strategy is specified by its zero-indexed position in a list of all
        strategies sorted alphabetically by role and sub-sorted alphabetically
        by strategy name.  For example if role1 has strategies s1, s2, and s3
        and role2 has strategies s4 and s5, then the restriction with all but
        the last strategy for each role is extracted by "0,1,3". This can be
        specified multiple times for several restrictions.""")
    return parser


def parse_index_spec(game, spec):
    """Parse restriction index specification"""
    rest = np.zeros(game.num_strats, bool)
    rest[list(map(int, spec.split(',')))] = True
    utils.check(
        game.is_restriction(rest), '"{}" does not define a valid restriction',
        spec)
    return rest


def main(args):
    """Entry point for restriction cli"""
    game = gamereader.load(args.input)

    # Collect all restrictions
    restrictions = []
    if args.detect:
        restrictions.extend(restrict.maximal_restrictions(game))

    for rest_file in args.restriction_file:
        restrictions.extend(game.restriction_from_json(spec)
                            for spec in json.load(rest_file))

    restrictions.extend(game.restriction_from_repr(spec)
                        for spec in args.text_spec)
    restrictions.extend(parse_index_spec(game, spec)
                        for spec in args.index_spec)

    if args.no_extract:
        json.dump([game.restriction_to_json(rest) for rest in restrictions],
                  args.output)
    else:
        json.dump([game.restrict(rest).to_json() for rest in restrictions],
                  args.output)
    args.output.write('\n')
