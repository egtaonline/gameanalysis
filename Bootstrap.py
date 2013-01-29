#! /usr/bin/env python2.7

from GameIO import read, to_JSON_str
from Subgames import subgame
from Regret import regret
from Nash import mixed_nash, replicator_dynamics
from Reductions import deviation_preserving_reduction as DPR

from os import listdir
from os.path import join, exists, dirname
from argparse import ArgumentParser
from random import sample
from copy import copy
from json import dumps
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


def bootstrap(game, equilibrium, stastic=regret, method_args=[], \
					method="resample", points=1000):
	"""
	Returns a bootstrap distribution for the statistic.

	To run a resample regret boostrap:
		bootstrap(game, [eq])
	To run a single-sample regret bootstrap:
		bootstrap(game, [eq], method='single_sample')
	To run a replicator dynamics bootstrap:
		bootstrap(game, [eq], replicator_dynamics)
	"""
	boot_dstr = []
	method = getattr(game, method)
	for i in range(points):
		method(*method_args)
		boot_dstr.append(stastic(game, equilibrium))
	game.reset()
	boot_dstr.sort()
	return boot_dstr


def synthetic_bootstrap_experiment(base_game_func, noise_func, statistic=\
		regret, num_games=1000,sample_sizes=[5,10,20,50,100,200,500], \
		bootstrap_args=[]):
	results = [{} for i in range(num_games)]
	for i in range(num_games):
		base_game = base_game_func()
		sample_game = add_samples(base_game, noise_func, 0)
		for sample_size in sample_sizes:
			new_samples = sample_size - sample_game.max_samples
			sample_game = add_samples(sample_game, noise_func, new_samples)
			equilibria = mixed_nash(sample_game, iters=1000)
			results[i][sample_size] = [ \
				{ \
					"mixture" : eq
					"regret" : regret(base_game, eq)
					"bootstrap" : bootstrap(game, [eq]
				} for eq in equilibria]
			
	raise NotImplementedError("TODO")


def simulation_bootstrap_experiment(simulated_game_data):
	raise NotImplementedError("TODO")


#TODO: replace this function
def run_leg_bootstrap_experiment(directory, bootstrap_iters, pre_agg, \
								subsample_sizes, out_name, bootstrap_func):
	for filename in sorted(filter(lambda f: f[-5:] == ".json", \
			listdir(directory))):
		try:
			n = filename[-8:-5]
			int(n)
		except ValueError:
			continue
		out_filename = join(directory, out_name + n + ".json")
		if exists(out_filename):
			continue
		d = read(join(directory, filename))
		true_game = d['0']
		game_results = {}

		for variance in set(d.keys()) - {'0'}:
			game_results[variance] = {}
			if pre_agg > 0:
				sample_game = pre_aggregate(d[variance], pre_agg)
			else:
				sample_game = d[variance]
			for size in subsample_sizes:
				subsample_game = subsample(sample_game, size)
				equilibria = mixed_nash(subsample_game, iters=1000)
				game_results[variance][size] = [ \
					{ \
					"mixture" : eq, \
					"regret" : regret(true_game, eq), \
					"bootstrap" : bootstrap_func(subsample_game, [eq], \
							points=bootstrap_iters)
					} for eq in equilibria]

		with open(out_filename, 'w') as f:
			f.write(to_JSON_str(game_results))


#TODO: replace this function
def run_cn_bootstrap_experiment(filename, bootstrap_iters=200, base_size=500, \
		num_base_games=200, subsample_sizes=[5,10,20,50,100]):
	DPR6 = partial(DPR, players={'All':6})
	out_filename = join(dirname(filename), "cn_boot_res_" +str(base_size)+ \
			"a.json")
	true_game = read(filename)
	results = []

	for i in range(num_base_games):
		base_game = subsample(true_game, base_size)
		base_game_results = {}
		for size in subsample_sizes:
			print size
			base_game_results[size] = []
			subsample_game = subsample(base_game, size)
			equilibria = mixed_nash(DPR6(subsample_game), iters=1000)
			base_game_results[size] = [ \
				{ \
					"mixture" : eq, \
					"regret" : regret(base_game, eq), \
					"bootstrap" : bootstrap_regret(subsample_game, eq, \
							200, 1, DPR6)
				} for eq in equilibria]
		results.append(base_game_results)

	with open(out_filename, 'w') as f:
		f.write(to_JSON_str(results))


#TODO: replace this function
def bootstrap_all(sample_game, true_game, equilibria, iters=200, \
		reduction=lambda g:g):
	results = [ \
	{ \
		"mixture" : m, \
		"regret" : regret(true_game, m), \
		"RD" : replicator_dynamics(true_game, m, 1000), \
		"profile resamples": [], \
		"game resamples": [] \
	} for m in equilibria]

	for i in range(iters):
		sample_game.resample(1)
		for j,m in enumerate(equilibria):
			results[j]["profile resamples"].append( \
					resample_results(reduction(sample_game), true_game, m))

	for i in range(iters):
		sample_game.resample(2)
		for j,m in enumerate(equilibria):
			results[j]["game resamples"].append( \
					resample_results(reduction(sample_game), true_game, m))

	sample_game.reset()
	return results


def resample_results(resampled_game, true_game, mix):
	rd = replicator_dynamics(resampled_game, mix, 1000)
	results = \
	{ \
		"resample_regret" : regret(resampled_game, mix), \
		"RD" : rd, \
		"RD_regret" : regret(true_game, rd)
	}
	return results


def main():
	parser = ArgumentParser()
	parser.add_argument("directory", type=str, help="Directory of games.")
	parser.add_argument("-res", type=int, default=1000, help="Number of "+\
						"bootstrap resamples per equilibrium.")
	parser.add_argument("-agg", type=int, default=0, help="Number of samples "+\
						"to pre-aggregate.")
	parser.add_argument("--single", action="store_true", help="Set to use "+\
						"the single_sample_regret function.")
	args = parser.parse_args()

	out_name = "leg_ss_res_" if args.single else "leg_boot_res_"
	subsample_sizes = [5,10,20,50,100,200,500,1000]
	if args.agg:
		out_name += "_agg" + str(args.agg)
		subsample_sizes = filter(lambda s: s < 1000/args.agg, subsample_sizes)
	run_leg_bootstrap_experiment(args.directory, args.res, args.agg, \
			subsample_sizes, out_name, single_sample_regret if args.single \
			else bootstrap)



if __name__ == "__main__":
	main()

