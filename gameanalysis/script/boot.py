"""calculate bootstrap bounds on a sample game"""
import argparse
import json
import sys

import numpy as np

from gameanalysis import bootstrap
from gameanalysis import gamereader
from gameanalysis import regret
from gameanalysis import scriptutils


CHOICES = {
    'regret': (bootstrap.mixture_regret, regret.mixture_regret),
    'surplus': (bootstrap.mixture_welfare, regret.mixed_social_welfare),
}


def add_parser(subparsers):
    parser = subparsers.add_parser(
        'bootstrap', aliases=['boot'], help="""Bootstrap on sample games""",
        description="""Compute bootstrap statistics using a sample game with
        data for every profile in the support of the subgame and potentially
        deviations. The return value is a list with an entry for each mixture
        in order. Each element is a dictionary mapping percentile to value,
        plus 'mean' to the mean.""")
    parser.add_argument(
        '--input', '-i', metavar='<input-file>', default=sys.stdin,
        type=argparse.FileType('r'), help="""Input sample game to run bootstrap
        on.  (default: stdin)""")
    parser.add_argument(
        '--output', '-o', metavar='<output-file>', default=sys.stdout,
        type=argparse.FileType('w'), help="""Output file for script. (default:
        stdout)""")
    parser.add_argument(
        'profiles', metavar='<profile>', nargs='+', help="""File or string with
        profiles from input game for which regrets should be calculated.  This
        file can be a list or a single profile""")
    parser.add_argument(
        '-t', '--type', default='regret', choices=CHOICES, help="""What to
        return. regret - returns the regret of each profile. surplus - returns
        the bootstrap surplus of every profile.  (default: %(default)s)""")
    parser.add_argument(
        '--processes', metavar='num-processes', type=int, help="""The number of
        processes when constructing bootstrap samples. Default will use all the
        cores available.""")
    parser.add_argument(
        '--percentiles', '-p', metavar='percentile', type=float, nargs='+',
        help="""Percentiles to return in [0, 100]. By default all bootstrap
        values will be returned sorted.""")
    parser.add_argument(
        '--num-bootstraps', '-n', metavar='num-bootstraps', default=101,
        type=int, help="""The number of bootstrap samples to acquire. More
        samples takes longer, but in general the percentiles requested should
        be a multiple of this number minus 1, otherwise there will be some
        error due to linear interpolation between points.  (default:
        %(default)s)""")
    return parser


def main(args):
    # TODO Profiles that aren't in support of mixtures or single deviations
    # could be safely pruned.
    game = gamereader.read(json.load(args.input))
    profiles = np.concatenate([game.from_prof_json(p)[None] for p
                               in scriptutils.load_profiles(args.profiles)])
    bootf, meanf = CHOICES[args.type]
    results = bootf(game, profiles, args.num_bootstraps,
                    percentiles=args.percentiles, processes=args.processes)
    if args.percentiles is None:
        args.percentiles = np.linspace(0, 100, args.num_bootstraps)
    percentile_strings = [str(p).rstrip('0').rstrip('.')
                          for p in args.percentiles]
    jresults = [{p: v.item() for p, v in zip(percentile_strings, boots)}
                for boots in results]
    for jres, mix in zip(jresults, profiles):
        jres['mean'] = meanf(game, mix)

    json.dump(jresults, args.output)
    args.output.write('\n')
