"""calculate regrets and deviations gains"""
import argparse
import json
import sys

import numpy as np

from gameanalysis import gameio
from gameanalysis import regret
from gameanalysis import scriptutils


def is_pure_profile(game, prof):
    """Returns true of the profile is pure"""
    # For an asymmetric game, this will always return false, but then it
    # shouldn't be an issue, because pure strategy regret will be more
    # informative.
    return np.any(game.role_reduce(prof) > 1.5)


def reg(game, serial, prof):
    """the regret of the profile"""
    if is_pure_profile(game, prof):
        return regret.pure_strategy_regret(game, prof).item()
    else:
        return regret.mixture_regret(game, prof).item()


def gains(game, serial, prof):
    """the gains from deviating from profile"""
    if is_pure_profile(game, prof):
        gains = regret.pure_strategy_deviation_gains(game, prof)
        return serial.to_deviation_payoff_json(prof, gains)
    else:
        gains = regret.mixture_deviation_gains(game, prof)
        return serial.to_payoff_json(np.ones(game.num_role_strats, bool),
                                     gains)


TYPE = {
    'regret': reg,
    'gains': gains,
    'ne': gains,
}
TYPE_HELP = ' '.join('`{}` - {}.'.format(s, f.__doc__)
                     for s, f in TYPE.items())


def add_parser(subparsers):
    parser = subparsers.add_parser('regret', aliases=['reg'], help="""Compute
                                   regret""", description="""Compute regret in
                                   input game of specified profiles.""")
    parser.add_argument('--input', '-i', metavar='<input-file>',
                        default=sys.stdin, type=argparse.FileType('r'),
                        help="""Input file for script.  (default: stdin)""")
    parser.add_argument('--output', '-o', metavar='<output-file>',
                        default=sys.stdout, type=argparse.FileType('w'),
                        help="""Output file for script. (default: stdout)""")
    parser.add_argument('profiles', metavar='<profile>',
                        nargs='+', help="""File with profiles or raw strings of
                        profiles from the input. The input can be a json list
                        of profiles or an individual profile.""")
    parser.add_argument('-t', '--type', default='regret', choices=TYPE,
                        help="""What to return: {} (default:
                        %(default)s)""".format(TYPE_HELP))
    return parser


def main(args):
    game, serial = gameio.read_game(json.load(args.input))
    prof_func = TYPE[args.type]
    regrets = [prof_func(game, serial, serial.from_prof_json(prof))
               for prof in scriptutils.load_profiles(args.profiles)]

    json.dump(regrets, args.output)
    args.output.write('\n')
