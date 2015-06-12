#!/usr/bin/env python3
"""Script for calculating regrets, deviations gains, and social welfare"""
import sys

if not hasattr(sys, 'real_prefix'):
    sys.stderr.write('Could not detect virtualenv. Make sure that you\'ve '
                     'activated the virtual env\n(`. bin/activate`).\n')
    sys.exit(1)

import argparse
import json

from gameanalyis import regret, rsgame


def _is_pure_profile(prof):
    """Returns true of the profile is pure"""
    # For an asymmetric game, this will always return false, but then it
    # shouldn't be an issue, because pure strategy regret will be more
    # informative.
    return any(sum(strats.values()) > 1.5 for strats in prof.values())


_TYPE = {
    'regret': lambda prof: (regret.pure_strategy_regret(prof)
                            if _is_pure_profile(prof)
                            else regret.mixture_regret(prof)),
    'gains': lambda prof: (regret.pure_strategy_deviation_gains(prof)
                           if _is_pure_profile(prof)
                           else regret.mixture_deviation_gains(prof)),
    'ne': lambda prof: (regret.pure_strategy_deviation_gains(prof)
                        if _is_pure_profile(prof)
                        else regret.mixture_deviation_gains(prof)),
    'welfare': lambda prof: (regret.pure_social_welfare(prof)
                             if _is_pure_profile(prof)
                             else regret.mixed_social_welfare(prof))
}

_PARSER = argparse.ArgumentParser(description='''Compute regret in input game
of specified profiles.''')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='file', default=sys.stdout,
                     type=argparse.FileType('w'), help='''Output dominance
                     file. The contents depend on the format specified.
                     (default: stdout)''')
_PARSER.add_argument('profiles', type=argparse.FileType('r'), help='''File with
profiles from input games for which regrets should be calculated. This file
needs to be a json list. of profiles''')
_PARSER.add_argument('-t', '--type', metavar='type', default='regret',
                     choices=_TYPE, help='''What to return. regret: returns the
                     the regret of the profile; gains: returns a json object of
                     the deviators gains for every deviation; ne: return the
                     "nash equilibrium regrets", these are identical to gains;
                     welfare: returns the social welfare of the
                     profile. (default: %(default)s)''')
# _PARSER.add_argument('-m', '--max-welfare', action='store_true', help='''Ignore
# all other options, and instead return the maximum social welfare''')


def main():
    args = _PARSER.parse_args()
    game = rsgame.Game.from_json(json.load(args.input))
    profiles = json.load(args.profiles)

    # FIXME Need to differentiate between mixed and pure
    regrets = [regret.mixture_regret(game, prof) for prof in profiles]

    json.dump(regrets, args.output, defaults=lambda x: x.to_json())
    args.output.write('\n')


if __name__ == '__main__':
    main()
else:
    raise ImportError('This module should not be imported')
