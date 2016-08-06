"""create random games"""
import argparse
import collections
import json
import sys
from os import path

import numpy.random as rand

from gameanalysis import gameio
from gameanalysis import gamegen


class ZeroSum(object):
    """two player zero sum game"""

    @staticmethod
    def update_parser(parser):
        parser.description = """A two player zero sum game with uniform
        values"""
        parser.add_argument('num_strats', metavar='<num-strategies>', type=int,
                            help="""Number of strategies to use in generating
                            the zero sum game""")

    @staticmethod
    def create(args):
        game = gamegen.two_player_zero_sum_game(args.num_strats)
        serial = gamegen.game_serializer(game)
        return game, serial


class RoleSymmetric(object):
    """independent uniform role symmetric game"""

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
        game = gamegen.role_symmetric_game(args.pands[::2],
                                           args.pands[1::2])
        serial = gamegen.game_serializer(game)
        return game, serial


class Congestion(object):
    """congestion game"""

    @staticmethod
    def update_parser(parser):
        parser.description = """A congestion game"""
        parser.add_argument('num_players', metavar='<num-players>', type=int,
                            help="""The number of players in the congestion
                            game.""")
        parser.add_argument('num_facilities', metavar='<num-facilities>',
                            type=int, help="""The number of facilities in the
                            congestion game.""")
        parser.add_argument('num_required', metavar='<num-required>', type=int,
                            help="""The number of facilities a player has to
                            occupy in the congestion game.""")

    @staticmethod
    def create(args):
        game, serial = gamegen.congestion_game(
            args.num_players, args.num_facilities, args.num_required, True)
        return game, serial


class ExperimentNoise(object):
    """add experimental noise to a given game"""

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
        parser.add_argument('--input', '-i', metavar='<input-file>',
                            default=sys.stdin, type=argparse.FileType('r'),
                            help="""Input file for script. (default: stdin)""")
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
        game, serial = gameio.read_game(json.load(args.input))
        dist = ExperimentNoise.distributions[args.distribution](args.max_width)
        return gamegen.add_noise(game, args.num_samples, noise=dist), serial


class Help(object):
    """get help on a generation function"""

    @staticmethod
    def update_parser(parser):
        parser.add_argument('gametype', metavar='game-type', nargs='?',
                            help="""Game type to get help on""")

    @staticmethod
    def create(args):
        if args.gametype is None:
            game = PARSER
        else:
            game = SUBCOMMANDS.choices[args.gametype]
        game.print_help()
        sys.exit(0)


GAME_TYPES = collections.OrderedDict([
    ('uzs', ZeroSum),
    ('ursym', RoleSymmetric),
    ('congest', Congestion),
    ('noise', ExperimentNoise),
    ('help', Help),
])


TYPE_HELP = ' '.join('`{}` - {}.'.format(t, m.__doc__)
                     for t, m in GAME_TYPES.items())
PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
BASE = argparse.ArgumentParser(prog='ga ' + PACKAGE, add_help=False)
BASE.add_argument('--output', '-o', metavar='<output-file>',
                  default=sys.stdout, type=argparse.FileType('w'),
                  help="""Output file for script. (default: stdout)""")
BASE.add_argument('--normalize', '-n', action='store_true', help="""Normalize
                  the game payoffs so that the minimum payoff is 0 and the
                  maximum payoff is 1""")
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, parents=[BASE],
                                 description="""Generate random games. Input is
                                 unused""")
SUBCOMMANDS = PARSER.add_subparsers(title='Generator types', dest='type',
                                    metavar='<game-type>', help="""The game
                                    generation function to use. Options are:
                                    """ + TYPE_HELP)
SUBCOMMANDS.required = True

for name, cls in GAME_TYPES.items():
    SUB_PARSER = SUBCOMMANDS.add_parser(name, parents=[BASE])
    cls.update_parser(SUB_PARSER)


def main():
    args = PARSER.parse_args()
    game, serial = GAME_TYPES[args.type].create(args)

    if args.normalize:
        game = gamegen.normalize(game)

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

    json.dump(serial.to_json(game), args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
