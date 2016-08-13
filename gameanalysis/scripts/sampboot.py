"""calculate bootstrap bounds from deviation samples"""
import argparse
import json
import sys
from os import path

import numpy as np

from gameanalysis import bootstrap
from gameanalysis import gameio


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Compute
                                 bootstrap statistics using samples from a
                                 mixture.""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input data to run
                    bootstrap on. Input should be a json list of surpluses
                    drawn from the distribution of interest. (default:
                    stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('--regret', '-r', nargs=2, metavar='<file>',
                    type=argparse.FileType('r'), help="""Compute bootstrap
regret from deviations instead of raw surplus values. This input should be a
list of objects with role, strategy, and payoff indicating the payoff to
deviating. The two arguments to this parameter must be a file with a game
                    specification, and a file with the mixture the deviation
                    samples were drawn from.""")
PARSER.add_argument('--dev-surplus', '-d', nargs=2, metavar='<file>',
                    type=argparse.FileType('r'), help="""Compute bootstrap
                    surplus from deviations instead of raw surplus values. The
                    arguments and inputs must look the same as regret, but this
                    will bootstrap over surplus instead.""")
PARSER.add_argument('--percentiles', '-p', metavar='percentile', type=float,
                    nargs='+', help="""Percentiles to return in [0, 100]. By
                    default all bootstrap values will be returned sorted.""")
PARSER.add_argument('--num-bootstraps', '-n', metavar='num-bootstraps',
                    default=101, type=int, help="""The number of bootstrap
samples to acquire. More samples takes longer, but in general the percentiles
requested should be a multiple of this number minus 1, otherwise there will be
                    some error due to linear interpolation between points.
                    (default: %(default)s)""")


def load_devs(game, serial, data):
    devs = np.empty((len(data) // game.num_role_strats, game.num_role_strats))
    inds = np.zeros(game.num_role_strats, int)
    for datum in data:
        sind = serial.role_strat_index(datum['role'], datum['strategy'])
        devs[inds[sind], sind] = datum['payoff']
        inds[sind] += 1
    return devs


def main():
    args = PARSER.parse_args()
    data = json.load(args.input)
    if args.regret is not None:
        game, serial = gameio.read_base_game(json.load(args.regret[0]))
        mix = serial.from_prof_json(json.load(args.regret[1]))
        devs = load_devs(game, serial, data)
        expect = game.role_reduce(devs * mix)
        result = bootstrap.sample_regret(game, expect, devs,
                                         args.num_bootstraps, args.percentiles)

    elif args.dev_surplus is not None:
        game, serial = gameio.read_base_game(json.load(args.dev_surplus[0]))
        mix = serial.from_prof_json(json.load(args.dev_surplus[1]))
        devs = load_devs(game, serial, data)
        surpluses = np.sum(devs * mix * game.role_repeat(game.num_players), 1)
        result = bootstrap.mean(surpluses, args.num_bootstraps,
                                args.percentiles)

    else:
        result = bootstrap.mean(data, args.num_bootstraps, args.percentiles)

    json.dump(result.tolist(), args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
