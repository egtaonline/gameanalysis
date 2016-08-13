"""calculate bootstrap bounds on a sample game"""
import argparse
import json
import sys
from os import path

import numpy as np

from gameanalysis import bootstrap
from gameanalysis import gameio


CHOICES = {
    'regret': bootstrap.mixture_regret,
    'surplus': bootstrap.mixture_welfare,
}

PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Compute
                                 bootstrap statistics using a sample game with
                                 data for every profile in the support of the
                                 subgame and potentially deviations.""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input sample game to
                    run bootstrap on.  (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('profiles', metavar='<profile-file>',
                    type=argparse.FileType('r'), help="""File with profiles
                    from input game for which regrets should be calculated.
                    This file needs to be a json list of profiles""")
PARSER.add_argument('-t', '--type', default='regret', choices=CHOICES,
                    help="""What to return. regret - returns the regret of each
                    profile. surplus - returns the bootstrap surplus of every
                    profile.  (default: %(default)s)""")
PARSER.add_argument('--processes', metavar='num-processes', type=int,
                    help="""The number of processes when constructing
                    bootstrap samples. Default will use all the cores
                    available.""")
PARSER.add_argument('--percentiles', '-p', metavar='percentile', type=float,
                    nargs='+', help="""Percentiles to return in [0, 100]. By
                    default all bootstrap values will be returned sorted.""")
PARSER.add_argument('--num-bootstraps', '-n', metavar='num-bootstraps',
                    default=101, type=int, help="""The number of bootstrap
samples to acquire. More samples takes longer, but in general the percentiles
requested should be a multiple of this number minus 1, otherwise there will be
                    some error due to linear interpolation between points.
                    (default: %(default)s)""")


def main():
    args = PARSER.parse_args()
    game, serial = gameio.read_sample_game(json.load(args.input))
    profiles = np.concatenate([serial.from_prof_json(p)[None] for p
                               in json.load(args.profiles)])
    results = CHOICES[args.type](game, profiles, args.num_bootstraps,
                                 args.percentiles, args.processes)
    json.dump(results.tolist(), args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
