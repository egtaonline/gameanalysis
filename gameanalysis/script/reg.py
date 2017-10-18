"""calculate regrets and deviations gains"""
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


def reg(game, prof):
    """the regret of the profile"""
    if is_pure_profile(game, prof):
        return regret.pure_strategy_regret(game, np.asarray(prof, int)).item()
    else:
        return regret.mixture_regret(game, prof).item()


def gains(game, prof):
    """the gains from deviating from profile"""
    if is_pure_profile(game, prof):
        prof = np.asarray(prof, int)
        gains = regret.pure_strategy_deviation_gains(game, prof)
        return game.to_dev_payoff_json(gains, prof)
    else:
        gains = regret.mixture_deviation_gains(game, prof)
        return game.to_payoff_json(gains)


TYPE = {
    'regret': reg,
    'gains': gains,
    'ne': gains,
}
TYPE_HELP = ' '.join('`{}` - {}.'.format(s, f.__doc__)
                     for s, f in TYPE.items())


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'regret', aliases=['reg'], help="""Compute regret""",
        description="""Compute regret in input game of specified profiles.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input file for script.  (default:
        stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        'profiles', metavar='<profile>', nargs='+', help="""File with profiles
        or raw strings of profiles from the input. The input can be a json list
        of profiles or an individual profile.""")
    parser.add_argument(
        '-t', '--type', default='regret', choices=TYPE, help="""What to return:
        {} (default: %(default)s)""".format(TYPE_HELP))
    return parser


def main(args):
    game = gamereader.read(json.load(args.input))
    prof_func = TYPE[args.type]
    regrets = [prof_func(game, game.from_mix_json(prof, verify=False))
               for prof in scriptutils.load_profiles(args.profiles)]
    json.dump(regrets, args.output)
    args.output.write('\n')
