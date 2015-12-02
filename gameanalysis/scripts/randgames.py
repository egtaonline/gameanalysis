"""Module for creating random games"""
import sys
import argparse
import json

from gameanalysis import randgames


class ZeroSum(object):
    @staticmethod
    def update_parser(parser):
        parser.description = """A two player zero sum game with uniform values"""
        parser.add_argument('num_strats', metavar='<num-strategies>', type=int,
                help="""Number of strategies to use in generating the zero sum
                game""")

    @staticmethod
    def create(args):
        return randgames.zero_sum_game(args.num_strats, cool=args.cool)


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
        return randgames.symmetric_game(args.num_players, args.num_strats,
                cool=args.cool)


class RoleSymmetric(object):
    @staticmethod
    def update_parser(parser):
        parser.description = """A role symmetric game"""
        parser.add_argument('pands', nargs='+', metavar='<player-strat-pair>',
                type=int, help="""The number of players and strategies for a
                role, specified as many times as there are roles. e.g. "1 4 3
                2" will be a two role game, the first role has 1 player and 4
                strategies, the second role has 3 players and 2 strategies.""")

    @staticmethod
    def create(args):
        assert len(args.pands) % 2 == 0, \
                'Must specify matching sets of players and strategies'
        return randgames.role_symmetric_game( len(args.pands) // 2,
                args.pands[::2], args.pands[1::2], cool=args.cool)


_GAME_TYPES = {
    'uzs': ZeroSum,
    'usym': Symmetric,
    'ursym': RoleSymmetric
}


def update_parser(parser):
    parser.description="""Generate random games. Input is unused"""
    parser.add_argument('--noise', choices=['none', 'normal', 'gauss_mix'],
            default='none', help="""Noise function. (default: %(default)s)""")
    parser.add_argument('--noise_args', nargs='+', default=[], metavar='<arg>',
                         help="""Arguments to be passed to the noise function.""")
    parser.add_argument('--cool', '-c', action='store_true', help="""Use role and
    strategy names that come from a text file instead of indexed names. This
    produces more "fun" games as thy have more interesting names, but it is harder
    to use in an automated sense because the names can't be predicted.""")

    subcommands = parser.add_subparsers(title='Generator types', dest='type',
            help="""The game generation function to use.""")
    subcommands.required = True

    for name, cls in _GAME_TYPES.items():
        sub_parser = subcommands.add_parser(name)
        cls.update_parser(sub_parser)

def main(args):
    game = _GAME_TYPES[args.type].create(args)

    # elif args.type == 'cs':
    #     game_func = congestion_game
    #     assert len(args.game_args) == 3, 'game_args must specify player, '+\
    #                                 'facility, and required facility counts'
    # elif args.type == 'LEG':
    #     game_func = local_effect_game
    #     assert len(args.game_args) == 2, 'game_args must specify player and '+\
    #                                 'strategy counts'
    # elif args.type == 'PMX':
    #     game_func = polymatrix_game
    #     assert len(args.game_args) == 2, 'game_args must specify player and '+\
    #                                 'strategy counts'
    # elif args.type == 'ind':
    #     game_func = polymatrix_game
    #     assert len(args.game_args) == 2, 'game_args must specify player and '+\
    #                                 'strategy counts'

    # if args.noise == 'normal':
    #     assert len(args.noise_args) == 2, 'noise_args must specify stdev '+\
    #                                         'and sample count'
    #     noise_args = [float(args.noise_args[0]), int(args.noise_args[1])]
    #     games = map(lambda g: normal_noise(g, *noise_args), games)
    # elif args.noise == 'gauss_mix':
    #     assert len(args.noise_args) == 3, 'noise_args must specify max '+\
    #                                 'stdev, sample count, and number of modes'
    #     noise_args = [float(args.noise_args[0]), int(args.noise_args[1]), \
    #                     int(args.noise_args[2])]
    #     games = map(lambda g: gaussian_mixture_noise(g, *noise_args), games)

    json.dump(game, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
