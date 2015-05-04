#! /usr/bin/env python2.7

import RandomGames as RG

from GameIO import read, to_JSON_str, io_parser
from Regret import regret
from Nash import mixed_nash, replicator_dynamics, pure_nash
from RoleSymmetricGame import SampleGame

from sys import stdin
from random import sample
from copy import copy, deepcopy
from functools import partial

import numpy as np

def subsample(game, num_samples):
	"""
	Returns a game with a random subset of the input game's payoff samples.

	Note: this is not intended for use on games with different numbers of 
	samples per profile.
	"""
	sg = copy(game)
	sg.sample_values = map(lambda p: p[:,:,sample(range(p.shape[-1]), \
			num_samples)], sg.sample_values)
	sg.reset()
	sg.max_samples = num_samples
	return sg


def holdout(game, num_samples):
	"""
	Returns the same game as subsample(), but also the game consisting of the 
	remaining samples. Won't work if payoffs have different numbers of samples.
	"""
	game.makeArrays()
	sg = copy(game)
	withheld = copy(game)
	sample_values = deepcopy(sg.sample_values)
	sample_values = sample_values.swapaxes(0,3)
	np.random.shuffle(sample_values)
	sample_values = sample_values.swapaxes(0,3)
	sg.sample_values = sample_values[:,:,:,:num_samples]
	sg.max_samples = sg.min_samples = num_samples
	sg.reset()
	withheld.sample_values = sample_values[:,:,:,num_samples:]
	withheld.max_samples = withheld.min_samples = game.max_samples - num_samples
	withheld.reset()
	return sg, withheld


def pre_aggregate(game, count):
	"""
	Aggregates samples to produce a game with |samples| / count samples.

	Note: this is not intended for use on games with different numbers of 
	samples per profile.
	"""
	agg = copy(game)
	sv = agg.sample_values
	sv = np.swapaxes(sv, 0, -1)
	np.random.shuffle(sv)
	sv = np.swapaxes(sv, 0, -1)
	shape = list(sv.shape)
	samples = shape[-1]
	if samples % count != 0:
		sv = sv[:,:,:,:samples - (samples % count)]
	shape.append(count)
	shape[-2] /= count
	sv = np.reshape(sv, shape)
	sv = np.average(sv, axis=-1)
	agg.sample_values = sv
	agg.reset()
	agg.min_samples = agg.max_samples = shape[-2]
	return agg


def bootstrap(game, equilibria, statistic=regret, method="resample", method_args=[], points=1000):
	"""
	Returns a bootstrap distribution for the statistic.

	To run a resample regret boostrap:
		bootstrap(game, eq)
	To run a single-sample regret bootstrap: 
		bootstrap(game, eq, method='single_sample')
	To run a replicator dynamics bootstrap:
		bootstrap(game, eq, replicator_dynamics)
	"""
        boot_lists = [[] for x in xrange(0,len(equilibria))]
        method = getattr(game, method)
        for __ in range(points):
            method(*method_args)
            count = 0
            for eq in equilibria:
                boot_lists[count].append(statistic(game, eq))
                count = count + 1
        game.reset()
        conf_intervals = []
        for boot_list in boot_lists:
            boot_list.sort()
            conf_intervals.append(boot_list[950])
        return conf_intervals


def parse_args():
    parser = io_parser()
    parser.add_argument("profiles", type=str, help="File with profiles from"+\
        " input games for which confidence intervals should be calculated.")
    return parser.parse_args()


def main():
        args = parse_args()
        game = args.input
        profiles = read(args.profiles)
        if not isinstance(profiles, list):
            profiles = [profiles]
        results = bootstrap(game,profiles)
        print to_JSON_str(results, indent=None)


if __name__ == "__main__":
	main()

