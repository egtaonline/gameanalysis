#!/usr/bin/env python3
"""Module for creating random games"""
import sys

if not hasattr(sys, 'real_prefix'):
    sys.stderr.write('Could not detect virtualenv. Make sure that you\'ve '
                     'activated the virtual env\n(`. bin/activate`).\n')
    sys.exit(1)

import argparse
import json

from gameanalysis import randgames


def _zero_sum_parse(zs_args, **kwargs):
    assert len(zs_args.arg) == 1, \
        'Must specify strategy count for uniform zero sum'
    return randgames.zero_sum_game(*map(int, zs_args), **kwargs)


def _symmetric_parse(sym_args, **kwargs):
    assert len(sym_args.arg) == 2, \
        'Must specify player and strategy counts for uniform symmetric'
    return randgames.symmetric_game(*map(int, sym_args), **kwargs),


def _role_symmetric_parse(rs_args, **kwargs):
    int_args = list(map(int, rs_args))
    num_roles = int_args[0]
    assert len(int_args) == 2 * num_roles + 1, \
        'Must specify a number of players and strategies for each role'
    num_players = int_args[1:1+num_roles]
    num_strats = int_args[-num_roles:]
    return randgames.role_symmetric_game(num_roles, num_players, num_strats,
                                         **kwargs)


_MAPPING = {
    'uzs': _zero_sum_parse,
    'usym': _symmetric_parse,
    'ursym': _role_symmetric_parse
}

_PARSER = argparse.ArgumentParser(description='Generate random games')
_PARSER.add_argument('type', choices=_MAPPING, help='''Type of random game to
generate. uzs = uniform zero sum.  usym = uniform symmetric. cg = congestion
game. pmx = polymatrix game. ind = independent game.''')
_PARSER.add_argument('arg', nargs='*', default=[],
                     help='Additional arguments for game generator function.')
_PARSER.add_argument('--noise', choices=['none', 'normal', 'gauss_mix'],
                     default='none', help='Noise function.')
_PARSER.add_argument('--noise_args', nargs='+', default=[],
                     help='Arguments to be passed to the noise function.')
_PARSER.add_argument('--output', '-o', metavar='output', default=sys.stdout,
                     type=argparse.FileType('w'),
                     help='Output destination; defaults to standard out')
_PARSER.add_argument('--indent', '-i', metavar='indent', type=int,
                     default=None,
                     help='Indent for json output; default = None')
_PARSER.add_argument('--cool', '-c', action='store_true', help='''Use role and
strategy names that come from a text file instead of indexed names. This
produces more "fun" games as thy have more interesting names, but it is harder
to use in an automated sense because the names can't be predicted.''')


def main():
    args = _PARSER.parse_args()

    game = _MAPPING[args.type](args.arg, cool=args.cool)
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

    json.dump(game, args.output, default=lambda x: x.to_json(),
              indent=args.indent)
    args.output.write('\n')


if __name__ == '__main__':
    main()
else:
    raise ImportError('This module should not be imported')
