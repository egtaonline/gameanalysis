"""Module for creating random games"""
import argparse
import collections
import json
import sys

import numpy.random as rand

from gameanalysis import gamegen
from gameanalysis import rsgame


class ZeroSum(object):
    @staticmethod
    def update_parser(parser):
        parser.description = """A two player zero sum game with uniform
        values"""
        parser.add_argument('num_strats', metavar='<num-strategies>', type=int,
                            help="""Number of strategies to use in generating
                            the zero sum game""")

    @staticmethod
    def create(args):
        return gamegen.zero_sum_game(args.num_strats, cool=args.cool)


class Symmetric(object):
    @staticmethod
    def update_parser(parser):
        parser.description = """A uniform symmetric game"""
        parser.add_argument('num_players', metavar='<num-players>', type=int,
                            help="""The number of players""")
        parser.add_argument('num_strats', metavar='<num-strategies>', type=int,
                            help="""The number of strategies""")

    @staticmethod
    def create(args):
        return gamegen.symmetric_game(args.num_players, args.num_strats,
                                      cool=args.cool)


class RoleSymmetric(object):
    @staticmethod
    def update_parser(parser):
        parser.description = """A role symmetric game"""
        parser.add_argument('pands', nargs='+', metavar='<player-strat-pair>',
                            type=int, help="""The number of players and
strategies for a role, specified as many times as there are roles. e.g. "1 4 3
2" will be a two role game, the first role has 1 player and 4 strategies, the
                            second role has 3 players and 2 strategies.""")

    @staticmethod
    def create(args):
        assert len(args.pands) % 2 == 0, \
            'Must specify matching sets of players and strategies'
        return gamegen.role_symmetric_game(len(args.pands) // 2,
                                           args.pands[::2],
                                           args.pands[1::2],
                                           cool=args.cool)


class ExperimentNoise(object):

    def uniform(max_width):
        def actual(shape):
            num_payoffs, num_samples = shape
            width = rand.uniform(0, max_width, num_payoffs)
            return rand.uniform(-width, width, (num_samples, num_payoffs)).T
        return actual

    def gaussian(max_width):
        def actual(shape):
            num_payoffs, num_samples = shape
            width = rand.uniform(0, max_width, num_payoffs)
            return rand.normal(0, width, (num_samples, num_payoffs)).T
        return actual

    def bimodal_gaussian(max_width):
        def actual(shape):
            num_payoffs, num_samples = shape
            width = rand.uniform(0, max_width, num_payoffs)
            spread = rand.normal(0, max_width, num_payoffs)
            means = ((rand.rand(num_payoffs) < .5) * 2 - 1) * spread
            return rand.normal(means, width, (num_samples, num_payoffs)).T
        return actual

    def gumbel(max_width):
        def actual(shape):
            num_payoffs, num_samples = shape
            width = rand.uniform(0, max_width, num_payoffs)
            return rand.gumbel(0, width, (num_samples, num_payoffs)).T
        return actual

    distributions = {
        'uniform': uniform,
        'gaussian': gaussian,
        'bimodal': bimodal_gaussian,
        'gumbel': gumbel,
    }

    @staticmethod
    def update_parser(parser):
        parser.description = """Add noise to an existing game"""
        parser.add_argument('distribution',
                            choices=ExperimentNoise.distributions, help="""The
distribution to sample from. uniform: the width corresponds to the half width
of the sampled value. gaussian: the width corresponds to the standard
                            deviation. bimodal: gaussian mixture where the
                            means are N(0, max-width) apart.  gumbel: the width
                            corresponds to the shape parameter.""")
        parser.add_argument('max_width', metavar='<max-width>', type=float,
                            help="""The max width for the distribution. For
                            each payoff the width is drawn uniformly from [0,
                            max-width], and then the noise is drawn from
                            distribution parameterized by width.""")
        parser.add_argument('num_samples', metavar='<num-samples>', type=int,
                            help="""The number of samples to draw for every
                            payoff.""")

    @staticmethod
    def create(args):
        game = rsgame.Game.from_json(json.load(args.input))
        dist = ExperimentNoise.distributions[args.distribution](args.max_width)
        return gamegen.add_noise(game, args.num_samples, dist)


_GAME_TYPES = collections.OrderedDict([
    ('uzs', ZeroSum),
    ('usym', Symmetric),
    ('ursym', RoleSymmetric),
    ('noise', ExperimentNoise),
])


def update_parser(parser, base):
    sub_base = argparse.ArgumentParser(add_help=False, parents=[base])
    base_group = sub_base.add_argument_group('game gen arguments')
    base_group.add_argument('--cool', '-c', action='store_true', help="""Use
                            role and strategy names that come from a text file
                            instead of indexed names.  This produces more "fun"
                            games as thy have more interesting names, but it is
                            harder to use in an automated sense because the
                            names can't be predicted.""")
    base_group.add_argument('--normalize', '-n', action='store_true',
                            help="""Normalize the game payoffs so that the
                            minimum payoff is 0 and the maximum payoff is 1""")

    parser.description = """Generate random games. Input is unused"""

    subcommands = parser.add_subparsers(title='Generator types', dest='type',
                                        help="""The game generation function to
                                        use.""")
    subcommands.required = True

    class Help(object):

        @staticmethod
        def update_parser(parser):
            parser.add_argument('gametype', metavar='game-type', nargs='?',
                                help="""Game type to get help on""")

        @staticmethod
        def create(args):
            if args.gametype is None:
                game = parser
            else:
                game = subcommands.choices[args.gametype]
            game.print_help()
            sys.exit(0)

    _GAME_TYPES['help'] = Help

    for name, cls in _GAME_TYPES.items():
        sub_parser = subcommands.add_parser(name, parents=[sub_base])
        cls.update_parser(sub_parser)


def main(args):
    game = _GAME_TYPES[args.type].create(args)

    if args.normalize:
        game = game.normalize()

    # elif args.type == 'cs':
    #     game_func = congestion_game
    #     assert len(args.game_args) == 3, 'game_args must specify player, '+\
    #                                 'facility, and required facility counts'
    # elif args.type == 'LEG':
    #     game_func = local_effect_game
    #     assert len(args.game_args) == 2, 'game_args must specify player and '+\ # noqa
    #                                 'strategy counts'
    # elif args.type == 'PMX':
    #     game_func = polymatrix_game
    #     assert len(args.game_args) == 2, 'game_args must specify player and '+\ # noqa
    #                                 'strategy counts'
    # elif args.type == 'ind':
    #     game_func = polymatrix_game
    #     assert len(args.game_args) == 2, 'game_args must specify player and '+\ # noqa
    #                                 'strategy counts'

    # if args.noise == 'normal':
    #     assert len(args.noise_args) == 2, 'noise_args must specify stdev '+\
    #                                         'and sample count'
    #     noise_args = [float(args.noise_args[0]), int(args.noise_args[1])]
    #     games = map(lambda g: normal_noise(g, *noise_args), games)
    # elif args.noise == 'gauss_mix':
    #     assert len(args.noise_args) == 3, 'noise_args must specify max '+\
    #                                 'stdev, sample count, and number of modes' # noqa
    #     noise_args = [float(args.noise_args[0]), int(args.noise_args[1]), \
    #                     int(args.noise_args[2])]
    #     games = map(lambda g: gaussian_mixture_noise(g, *noise_args), games)

    json.dump(game, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
