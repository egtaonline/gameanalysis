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
        """Create zero sum parser"""
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
        """Create zero sum game"""
        return gamegen.two_player_zero_sum_game(args.num_strats)


class RoleSymmetric(object):
    """independent uniform role symmetric game"""

    @staticmethod
    def add_parser(subparsers, base):
        """Add parser for role symmetric game"""
        parser = subparsers.add_parser(
            'ursym', aliases=['rs'], parents=[base], description="""Construct a
            role symmetric game.""", help="""Role Symmetric""")
        parser.add_argument(
            'pands', metavar='players:strats,...',
            help="""The number of players and strategies for a role, specified
            as many times as there are roles. e.g. "1:4,3:2" will be a two role
            game, the first role has 1 player and 4 strategies, the second role
            has 3 players and 2 strategies.""")
        return parser

    @staticmethod
    def create(args):
        """Create role symmetric game"""
        players, strats = zip(*(map(int, ps.split(':')) for ps
                                in args.pands.split(',')))
        return gamegen.game(players, strats)


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
        """Create parser for adding noise to game"""
        parser = subparsers.add_parser(
            'noise', parents=[base], help="""Add Noise""", description="""Add
            noise to an existing game.""")
        parser.add_argument(
            '--input', '-i', metavar='<input-file>', default=sys.stdin,
            type=argparse.FileType('r'), help="""Input file for script.
            (default: stdin)""")
        parser.add_argument(
            '--distribution', '-d', choices=Noise.distributions,
            default='gaussian', help="""The distribution to sample from.
            (default: %(default)s)""")
        parser.add_argument(
            '--min-width', metavar='<min-width>', default=0, type=float,
            help="""The minimum width for each distribution. See max-width.
            (default: %(default)g)""")
        parser.add_argument(
            '--max-width', '-w', metavar='<max-width>', default=1, type=float,
            help="""The max width for each distribution. For each payoff the
            width is drawn uniformly from [min-width, max-width], and then the
            noise is drawn from distributions whith zero mean and width
            standard deviation. (default: %(default)g)""")
        parser.add_argument(
            '--min-samples', '-s', default=1, metavar='<min-samples>',
            type=int, help="""The minimum number of samples to draw for every
            payoff, before potentially sampling more with prob. (default:
            %(default)d)""")
        parser.add_argument(
            '--prob', '-p', metavar='<prob>', default=0, type=float,
            help="""The probability of sampling successively more profiles
            after the minimum number have been sampled. Profiles will keep
            being sampled until a failure is reached. Setting this to 0.5 will
            add one extra sample in expectation to every profile. (default:
            %(default)g)""")
        return parser

    @staticmethod
    def create(args):
        """Add noise to game"""
        game = gamereader.load(args.input)
        dist = Noise.distributions[args.distribution]
        return gamegen.gen_noise(
            game, args.prob, args.min_samples, args.min_width, args.max_width,
            noise_distribution=dist)


_TYPES = {}


def add_parser(subparsers):
    """Add parser for game gen"""
    base = argparse.ArgumentParser(add_help=False)
    group = base.add_argument_group('generate arguments')
    group.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    group.add_argument(
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
    """Entry point for game gen"""
    json.dump(_TYPES[args.type].create(args).to_json(), args.output)
    args.output.write('\n')
