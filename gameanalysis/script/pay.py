"""calculate payoffs and social welfare"""
import argparse
import json
import sys

import numpy as np

from gameanalysis import gamereader
from gameanalysis import regret
from gameanalysis import scriptutils


def is_pure_profile(game, prof):
    """Returns true of the profile is pure"""
    # For an asymmetric game, this will always return false, but then it
    # shouldn't be an issue, because pure strategy regret will be more
    # informative.
    pure = np.any(np.add.reduceat(prof, game.role_starts) > 1.5)
    assert (game.is_profile(np.asarray(prof, int)) if pure else
            game.is_mixture(prof))
    return pure


def payoffs(game, prof):
    """get payoffs to every agent or role"""
    if is_pure_profile(game, prof):
        prof = np.asarray(prof, int)
        return game.to_payoff_json(game.get_payoffs(prof), prof)
    else:
        return game.to_role_json(game.expected_payoffs(prof))


def welfare(game, prof):
    """get the welfare of a profile or mixture"""
    if is_pure_profile(game, prof):
        return regret.pure_social_welfare(game, np.asarray(prof, int)).item()
    else:
        return regret.mixed_social_welfare(game, prof).item()


TYPE = {
    'payoffs': payoffs,
    'welfare': welfare,
}
TYPE_HELP = ' '.join('`{}` - {}.'.format(s, f.__doc__)
                     for s, f in TYPE.items())


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'payoffs', aliases=['pay'], help="""Compute payoffs""",
        description="""Compute payoff relative information in input game of
        specified profiles.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        'profiles', metavar='<profile>', nargs='+', help="""File or string with
        json profiles from input games for which payoffs should be calculated.
        This file can be to be a list or a single profile""")
    parser.add_argument(
        '-t', '--type', metavar='type', default='payoffs', choices=TYPE,
        help="""What to return: {} (default: %(default)s)""".format(TYPE_HELP))
    return parser


def main(args):
    game = gamereader.read(json.load(args.input))
    prof_func = TYPE[args.type]
    payoffs = [prof_func(game, game.from_mix_json(prof, verify=False))
               for prof in scriptutils.load_profiles(args.profiles)]
    json.dump(payoffs, args.output)
    args.output.write('\n')
