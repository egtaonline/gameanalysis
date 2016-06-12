"""extract and find subgames"""
import argparse
import json
import sys
from os import path

import numpy as np

from gameanalysis import gameio
from gameanalysis import subgame


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Extract
                                 subgames and optionally detects all complete
                                 subgames. All subgame specifications will be
                                 concatentated, resulting in a list of
                                 subgames.""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input file for script.
                    (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('--no-extract', '-n', action='store_true', help="""Don't
                    extract full subgame data, just print the specifications of
                    the subgames. This is mainly only useful with the detect
                    option.""")

SUB_GROUP = PARSER.add_argument_group(title='subgame specifications',
                                      description="""These are all of the ways
                                      to specify subgames to extract.  All of
                                      these specifications are concatentated
                                      together before being output.""")
SUB_GROUP.add_argument('--detect', '-d', action='store_true', help="""Run
                       clique finding to detect maximally complete
                       subgames.""")
SUB_GROUP.add_argument('--subgame-file', '-f', metavar='<file>', default=[],
                       type=argparse.FileType('r'), action='append', help="""A
file that contains a list of subgames. A subgame is simply a mapping of roles
to strategies i.e. "{r: ["s1", "s2"]}". This is the same format that can be
                       output by this script with the no-extract option. This
                       can be specified multiple times.""")
SUB_GROUP.add_argument('--text-spec', '-t', nargs='+', metavar='<role-strat>',
                       default=[], action='append', help="""Specify a subgame
as a list of roles and strategies. To specify the subgame where role0 has
strategies strat0 and strat2 and role1 has strategy strat1 enter "role0 strat0
strat2 role1 strat1". This option is ambiguous if strategies share names with
roles. For unambiguous specification, use index specification. This can be
entered several times in order to specify several subgames.""")
SUB_GROUP.add_argument('--index-spec', '-s', type=int, nargs='+', default=[],
                       metavar='<strat-index>', action='append',
                       help="""Specify a subgame with a list of strategy
indices. A strategy is specified by its zero-indexed position in a list of all
strategies sorted alphabetically by role and sub-sorted alphabetically by
strategy name. For example if role1 has strategies s1, s2, and s3 and role2 has
strategies s4 and s5, then the subgame with all but the last strategy for each
                       role is extracted by "0 1 3". This can be specified
                       multiple times for several subgames.""")


def parse_text_spec(serial, spec):
    current_role = '<undefined role>'
    subg = np.zeros(serial.num_role_strats, bool)
    roles = set(serial.role_names)
    for role_strat in spec:
        if role_strat in roles:
            current_role = role_strat
        else:
            subg[serial.role_strat_index(current_role, role_strat)] = True
    return subg


def parse_index_spec(serial, spec):
    subg = np.zeros(serial.num_role_strats, bool)
    subg[spec] = True
    return subg


def parse_json_spec(serial, spec):
    return serial.from_prof_json({r: {s: True for s in ses}
                                  for r, ses in spec.items()})


def main():
    args = PARSER.parse_args()
    game, serial = gameio.read_game(json.load(args.input))

    # Collect all subgames
    subgames = []
    if args.detect:
        subgames.extend(subgame.maximal_subgames(game))

    for sub_file in args.subgame_file:
        subgames.extend(parse_json_spec(serial, spec)
                        for spec in json.load(sub_file))

    subgames.extend(parse_text_spec(serial, spec) for spec in args.text_spec)
    subgames.extend(parse_index_spec(serial, spec) for spec in args.index_spec)

    if args.no_extract:
        json.dump([{r: list(s) for r, s in serial.to_prof_json(sub).items()}
                   for sub in subgames], args.output)
    else:
        json.dump([
            subgame.subserializer(serial, sub).to_json(
                subgame.subgame(game, sub)) for sub in subgames], args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
