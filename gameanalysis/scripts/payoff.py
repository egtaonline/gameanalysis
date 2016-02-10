"""Script for calculating regrets, deviations gains, and social welfare"""
import argparse
import json

from gameanalysis import regret
from gameanalysis import rsgame


def _is_pure_profile(prof):
    """Returns true of the profile is pure"""
    # For an asymmetric game, this will always return false, but then it
    # shouldn't be an issue, because pure strategy regret will be more
    # informative.
    return any(sum(strats.values()) > 1.5 for strats in prof.values())


_TYPE = {
    'payoffs': lambda game, prof: (
        game.get_payoffs(prof)
        if _is_pure_profile(prof)
        else game.get_expected_payoff(prof)),
    'welfare': lambda game, prof: (
        regret.pure_social_welfare(game, prof)
        if _is_pure_profile(prof)
        else regret.mixed_social_welfare(game, prof)),
}


def update_parser(parser):
    parser.description = """Compute payoff relative information in input game
of specified profiles."""
    parser.add_argument('profiles', metavar='<profile-file>',
                        type=argparse.FileType('r'), help="""File with profiles
                        from input games for which payoffs should be
                        calculated. This file needs to be a json list of
                        profiles""")
    parser.add_argument('-t', '--type', metavar='type', default='payoffs',
                        choices=_TYPE, help="""What to return. payoffs: returns
                        the payoffs of every role, and for pure profiles
                        strategy, for each profile; welfare: returns the social
                        welfare of the profile.  (default: %(default)s)""")


def main(args):
    game = rsgame.Game.from_json(json.load(args.input))
    profiles = json.load(args.profiles)
    prof_func = _TYPE[args.type]

    regrets = [prof_func(game, prof) for prof in profiles]

    json.dump(regrets, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
