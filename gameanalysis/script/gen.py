"""create random games"""
import argparse
import json
import sys

from gameanalysis import gamegen
from gameanalysis import gamereader


class ZeroSum(object):
    """two player zero sum game"""

    @staticmethod
    def add_parser(subparsers, base):
        parser = subparsers.add_parser(
            'uzs', aliases=['zs'], parents=[base], help="""Two-player
            Zero-sum""", description="""Construct a two-player zero-sum game
            with uniform payoffs.""")
        parser.add_argument(
            'num_strats', metavar='<num-strategies>', type=int, help="""Number
            of strategies to use in generating the zero sum game""")
        return parser

    @staticmethod
    def create(args):
        return gamegen.two_player_zero_sum_game(args.num_strats)


class RoleSymmetric(object):
    """independent uniform role symmetric game"""

    @staticmethod
    def add_parser(subparsers, base):
        parser = subparsers.add_parser(
            'ursym', aliases=['rs'], parents=[base], description="""Construct a
            role symmetric game.""", help="""Role Symmetric""")
        parser.add_argument(
            'pands', nargs='+', metavar='<player> <strat>', type=int,
            help="""The number of players and strategies for a role, specified
            as many times as there are roles. e.g. "1 4 3 2" will be a two role
            game, the first role has 1 player and 4 strategies, the second role
            has 3 players and 2 strategies.""")
        return parser

    @staticmethod
    def create(args):
        assert len(args.pands) % 2 == 0, \
            'Must specify matching sets of players and strategies'
        return gamegen.role_symmetric_game(args.pands[::2], args.pands[1::2])


class Noise(object):
    """add experimental noise to a given game"""

    distributions = {
        'uniform': gamegen.width_uniform,
        'gaussian': gamegen.width_gaussian,
        'bimodal': gamegen.width_bimodal,
        'gumbel': gamegen.width_gumbel,
    }

    @staticmethod
    def add_parser(subparsers, base):
        parser = subparsers.add_parser(
            'noise', parents=[base], help="""Add Noise""", description="""Add
            noise to an existing game.""")
        parser.add_argument(
            '--input', '-i', metavar='<input-file>', default=sys.stdin,
            type=argparse.FileType('r'), help="""Input file for script.
            (default: stdin)""")
        parser.add_argument(
            'distribution', choices=Noise.distributions, help="""The
            distribution to sample from. uniform: the width corresponds to the
            half width of the sampled value. gaussian: the width corresponds to
            the standard deviation. bimodal: gaussian mixture where the means
            are N(0, max-width) apart.  gumbel: the width corresponds to the
            shape parameter.""")
        parser.add_argument(
            'max_width', metavar='<max-width>', type=float, help="""The max
            width for the distribution. For each payoff the width is drawn
            uniformly from [0, max-width], and then the noise is drawn from
            distribution parameterized by width.""")
        parser.add_argument(
            'num_samples', metavar='<num-samples>', type=int, help="""The
            number of samples to draw for every payoff.""")
        return parser

    @staticmethod
    def create(args):
        game = gamereader.read(json.load(args.input))
        dist = Noise.distributions[args.distribution]
        return gamegen.add_noise_width(
            game, args.num_samples, args.max_width, dist)


_TYPES = {}


def add_parser(subparsers):
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    base.add_argument(
        '--normalize', '-n', action='store_true', help="""Normalize the game
        payoffs so that the minimum payoff is 0 and the maximum payoff is 1""")
    parser = subparsers.add_parser(
        'generate', aliases=['gen'], parents=[base], help="""Generate random
        games""", description="""Generate random games.  Input is unused""")
    subparsers = parser.add_subparsers(
        title='game generator types', dest='type', metavar='<game-type>',
        help="""The game generation function to use. Allowed values:""")
    for gentype in [RoleSymmetric, ZeroSum, Noise]:
        subparser = gentype.add_parser(subparsers, base)
        subparser.create = gentype.create

    _TYPES.update(subparsers.choices)
    return parser


def main(args):
    json.dump(_TYPES[args.type].create(args).to_json(), args.output)
    args.output.write('\n')
