"""sample profiles from a mixture"""
import argparse
import hashlib
import json
import sys

import numpy as np

from gameanalysis import gamereader
from gameanalysis import rsgame


# TODO There should be an easier way to handle subparsers that is robust to
# errors and concise

rests = ['restriction', 'rest']
mixes = ['mixture', 'mix']
profs = ['profile', 'prof']


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'sample', aliases=['samp'], help="""Sample objects from a game.  This
        returns each object on a new line, allowing streaming to some other
        utility.""", description="""Sample profiles from a mixture.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Game file to draw samples from.
        (default: stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""File to write stream of objects
        to. (default: stdout)""")
    parser.add_argument(
        '--num', '-n', metavar='<num-samples>', default=1, type=int,
        help="""The number of samples to gather.  (default: %(default)d)""")
    parser.add_argument(
        '--seed', metavar='<string>', help="""Set the seed of the random number
        generator to get consistent output. The seed is set to a hash of string
        passed in.""")

    types = parser.add_subparsers(
        title='types', dest='types', metavar='<type>', help="""The type of game
        object to sample from the input. Available commands are:""")
    types.required = True

    rest = types.add_parser(
        rests[0], aliases=rests[1:], help='restrictions', description="""Sample
        random restrictions""")
    rest.add_argument(
        '--prob', '-p', type=float, metavar='<prob>', help="""Probability that
        an particular strategy occurs in the restriction. By default this is
        set so all restrictions are equally likely.""")
    rest.add_argument(
        '--unnormalize', '-u', action='store_false', help="""Don't normalize
        probabilities so they are reflected in the final distributions.""")

    mix = types.add_parser(
        mixes[0], aliases=mixes[1:], help='mixtures', description="""Sample
        random mixtures""")
    mix.add_argument(
        '--alpha', '-a', type=float, default=1.0, metavar='<alpha>',
        help="""Alpha argument for random mixtures. One represents uniform
        probability. (default: %(default)s)""")
    mix.add_argument(
        '--sparse', '-s', default=_nosparse, type=float, nargs='?',
        metavar='prob', help="""Generate sparse mixtures with probability `prob` of
        being nonsparse.""")

    prof = types.add_parser(
        profs[0], aliases=profs[1:], help='profiles', description="""Sample
        random profiles""")
    excl = prof.add_mutually_exclusive_group()
    excl.add_argument(
        '--mix', '-m', metavar='<mixture-file>', type=argparse.FileType('r'),
        help="""A file with the mixture to sample profiles from.""")
    excl.add_argument(
        '--alpha', '-a', metavar='<alpha>', type=float, default=1.0,
        help="""Alpha to use to draw mixtures that will get rounded to
        profiles.""")

    return parser


def main(args):
    game = rsgame.emptygame_copy(gamereader.load(args.input))
    if args.seed is not None:
        # Python hash is randomly salted, so we use this to guarantee
        # determinism
        np.random.seed(int(
            hashlib.sha256(args.seed.encode('utf8')).hexdigest()[:8], 16))

    if args.types in rests:
        objs = (game.restriction_to_json(rest) for rest
                in game.random_restrictions(
                    args.num, strat_prob=args.prob,
                    normalize=args.unnormalize))

    elif args.types in mixes:
        if args.sparse is _nosparse:
            mix = game.random_mixtures(args.num, alpha=args.alpha)
        else:
            mix = game.random_sparse_mixtures(args.num, alpha=args.alpha,
                                              support_prob=args.sparse)
        objs = (game.mixture_to_json(m) for m in mix)

    elif args.types in profs:  # pragma: no branch
        if args.mix is None:
            prof = game.round_mixture_to_profile(
                game.random_mixtures(args.num, alpha=args.alpha))
        else:
            mix = game.mixture_from_json(json.load(args.mix))
            prof = game.random_profiles(args.num, mix)
        objs = (game.profile_to_json(p) for p in prof)

    # We sort the keys when a seed is set to guarantee identical output.  This
    # technically shouldn't be necessary, but on the off chance that a
    # simulator depends on the order, we want to make sure we produce identical
    # results.
    for obj in objs:
        json.dump(obj, args.output, sort_keys=args.seed is not None)
        args.output.write('\n')


class _NoSparseClass(object):
    pass


_nosparse = _NoSparseClass()
