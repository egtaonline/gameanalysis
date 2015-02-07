#! /usr/bin/env python2.7

import numpy as np
from numpy.random import multinomial
from random import sample
from os import listdir as ls, mkdir
from os.path import join, exists, isdir, dirname, basename, abspath
from argparse import ArgumentParser
import cPickle
import json

# import GaussianProcess but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	from sklearn.gaussian_process import GaussianProcess
	from sklearn.grid_search import GridSearchCV
except ImportError:
	warnings.warn("sklearn.gaussian_process is required for game learning.")

from Reductions import HR_profiles, DPR_profiles, full_prof_DPR, DPR, HR
import RoleSymmetricGame as RSG
from Nash import mixed_nash
from BasicFunctions import average, nCr, leading_zeros
from HashableClasses import h_array
from ActionGraphGame import LEG_to_AGG
from GameIO import to_JSON_str, read
from itertools import combinations_with_replacement as CwR
from itertools import permutations, product


class DiffGP:
	def __init__(self, GP_mean, GP_diff):
		self.GP_mean = GP_mean
		self.GP_diff = GP_diff

	def predict(self, vec):
		return self.GP_mean.predict(vec) + self.GP_diff.predict(vec)


def GP_learn(game, cross_validate=False):
	"""
	Create a GP regression for each role and strategy.

	Parameters:
	game:			RoleSymmetricGame.SampleGame object with enough data to
					estimate payoff functions.
	cross_validate:	If set to True, cross-validation will be used to select
					parameters of the GPs.
	"""
	#X[r][s] stores the vectorization of each profile in which some players in
	#role r select strategy s. Profiles are listed in the same order as in the
	#input game. Y[r][s] stores corresponding payoff values in the same order.
	#Yd (d=diff) stores the difference from the average payoff. Ywd (w=weighted)
	#stores difference from expected payoff for a random agent. Ym (m=mean)
	#stores average payoffs. Ywm stores expected payoff for a random agent.
	X = {
		"profiles":[],
		"samples":{r:{s:[] for s in game.strategies[r]} for r in game.roles}
	}
	Y = {
		"Y":{r:{s:[] for s in game.strategies[r]} for r in game.roles},
		"Yd":{r:{s:[] for s in game.strategies[r]} for r in game.roles},
		"Ywd":{r:{s:[] for s in game.strategies[r]} for r in game.roles},
		"Ym":{r:[] for r in game.roles},
		"Ywm":{r:[] for r in game.roles}
	}
	for p in range(len(game)):#fill X and Y
		prof = game.counts[p]
		samples = game.sample_values[p]
		x = np.array(prof2vec(game, prof), dtype=float)[None].T
		x = np.tile(x, (1,1,samples.shape[-1]))
		x += np.random.normal(0,1e-9, x.shape)
		ym = samples.mean(1).mean(1)
		ywm = ((samples * x).sum(1) / x.sum(1)).mean(1)
		X["profiles"].append(x[0,:,0])
		for r,role in enumerate(game.roles):
			Y["Ym"][role].append(ym[r])
			Y["Ywm"][role].append(ywm[r])
			for s,strat in enumerate(game.strategies[role]):
				if prof[r][s] > 0:
					y = samples[r,s]
					Y["Y"][role][strat].extend(y)
					Y["Yd"][role][strat].extend(y - ym)
					Y["Ywd"][role][strat].extend(y - ywm)
					for i in range(y.size):
						X["samples"][role][strat].append(x[0,:,i])

	#GPs stores the learned GP for each role and strategy
	GPs = {y:{} for y in Y}
	for role in game.roles:
		for y in ["Ym", "Ywm"]:
			GPs[y][role] = train_GP(X["profiles"], Y[y][role], cross_validate)
		for y in ["Y", "Yd", "Ywd"]:
			GPs[y][role] = {}
			for strat in game.strategies[role]:
				GPs[y][role][strat] = train_GP(X["samples"][role][strat], \
										Y[y][role][strat], cross_validate)
	return GPs


def train_GP(X, Y, cross_validate=False):
	if cross_validate:
		gp = GaussianProcess(storage_mode='light', thetaL=1e-4, thetaU=1e9, \
							normalize=True)
		params = {
				"corr":["absolute_exponential","squared_exponential",
						"cubic","linear"],
				"nugget":[1e-10,1e-6,1e-4,1e-2,1e0,1e2,1e4]
		}
		cv = GridSearchCV(gp, params)
		cv.fit(X, Y)
		return cv.best_estimator_
	else:
		gp = GaussianProcess(storage_mode='light', corr="cubic", nugget=1)
		gp.fit(X, Y)
		return gp


def GP_DPR(game, GPs, players):
	if len(game.roles) == 1 and isinstance(players, int):
		players = {game.roles[0]:players}
	elif isinstance(players, list):
		players = dict(zip(game.roles, players))

	learned_game = RSG.Game(game.roles, players, game.strategies)
	for prof in learned_game.allProfiles():
		role_payoffs = {}
		for role in game.roles:
			role_payoffs[role] = []
			for strat,count in prof[role].iteritems():
				full_prof = full_prof_DPR(prof, role, strat, game.players)
				prof_x = prof2vec(game, full_prof)
				prof_y = GPs[role][strat].predict(prof_x)
				role_payoffs[role].append(RSG.PayoffData(strat, count, prof_y))
		learned_game.addProfile(role_payoffs)

	return learned_game


def GP_sample(game, GPs, mix, samples=1000):
	"""
	Mimics game.ExpectedValues via sampling from the GPs.
	"""
	EVs = np.zeros(mix.shape)
	partial_profiles = []
	for r,role in enumerate(game.roles):
		partial_profiles.append(multinomial(game.players[role], mix[r],
															samples))
	profiles = [prof2vec(game, p) for p in zip(*partial_profiles)]
	for r,role in enumerate(game.roles):
		for s,strat in enumerate(game.strategies[role]):
			EVs[r,s] = GPs[role][strat].predict(profiles).mean()
	return EVs


def GP_point(game, GPs, mix, *args):
	"""
	Mimics game.ExpectedValues by returning the GPs' value estimates for the
	profile with population proportions equal to the mixture probabilities.

	*args is ignored ... it allows the same signature as GP_sample()
	"""
	prof = [mix[r]*game.players[role] for r,role in enumerate(game.roles)]
	vec = prof2vec(game, prof)
	EVs = np.zeros(mix.shape)
	for r,role in enumerate(game.roles):
		for s,strat in enumerate(game.strategies[role]):
			EVs[r,s] = GPs[role][strat].predict(vec)
	return EVs


def GP_RD(game, GPs, regret_thresh=1e-2, dist_thresh=1e-3, \
			random_restarts=0, at_least_one=False, iters=1000, \
			converge_thresh=1e-6, ev_samples=1000, EV_func=GP_sample):
	"""
	Estimate equilibria with RD using from GP regression models.
	"""
	candidates = []
	regrets = {}
	for mix in game.biasedMixtures() + [game.uniformMixture() ]+ \
			[game.randomMixture() for _ in range(random_restarts)]:
		for _ in range(iters):
			old_mix = mix
			EVs = EV_func(game, GPs, mix, game.players, ev_samples)
			mix = (EVs - game.minPayoffs + RSG.tiny) * mix
			mix = mix / mix.sum(1).reshape(mix.shape[0],1)
			if np.linalg.norm(mix - old_mix) <= converge_thresh:
				break
		mix[mix < 0] = 0
		candidates.append(h_array(mix))
		EVs = EV_func(game, GPs, mix, game.players, ev_samples)
		regrets[h_array(mix)] = (EVs.max(1) - (EVs * mix).sum(1)).max()

	candidates.sort(key=regrets.get)
	equilibria = []
	for c in filter(lambda c: regrets[c] < regret_thresh, candidates):
		if all(np.linalg.norm(e - c, 2) >= dist_thresh for e in equilibria):
			equilibria.append(c)
	if len(equilibria) == 0 and at_least_one:
		return [min(candidates, key=regrets.get)]
	return equilibria


def prof2vec(game, prof):
	"""
	Turns a profile (represented as Profile object or count array) into a
	1-D vector of strategy counts.
	"""
	if isinstance(prof, RSG.Profile):
		prof = game.toArray(prof)
	vec = []
	for r in range(len(game.roles)):
		vec.extend(prof[r][:game.numStrategies[r]])
	return vec


def sample_at_reduction(AGG, samples, reduction_profiles, players):
	"""
	AGG:		ActionGraphGame.Noisy_AGG object
	samples:	total number of samples to collect; will be spread as evenly as
				possibl across generated profiles; profiles that get sampled one
				extra time are chosen uniformly at random
	reduction_profiles:
				function that takes a number of players and generates a set of
				profiles; intended settings: DPR_profiles, HR_profiles
	players:	number of players in the reduced game; passed as an argument to
				reduction_profiles

	RETURNS:	RoleSymmetricGame.SampleGame object with samples drawn from AGG;
				samples are allocated as evenly as possible to profiles
				generated by reduction_profiles
	"""
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})
	profiles = reduction_profiles(g, {"All":players})
	s = samples/len(profiles)
	extras = sample(profiles, samples - s*len(profiles))
	counts = {p:(s+1 if p in extras else s) for p in profiles}
	for prof,count in counts.items():
		values = AGG.sample(prof["All"], count)
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})
	return g


def sample_near_reduction(AGG, samples, reduction_profiles, players):
	"""
	AGG:		ActionGraphGame.Noisy_AGG object
	samples:	total number of samples to collect; will be spread as evenly as
				possibl across generated profiles; profiles that get sampled one
				extra time are chosen uniformly at random
	reduction_profiles:
				function that takes a number of players and generates a set of
				profiles; intended settings: DPR_profiles, HR_profiles
	players:	number of players in the reduced game; passed as an argument to
				reduction_profiles

	RETURNS:	RoleSymmetricGame.SampleGame object with samples drawn from AGG;
				samples are allocated near near the profiles generated by
				reduction_profiles by treating the fraction of players in a
				profile playing each strategy as a distribution and drawing N
				samples from it; the resulting N-player profile gets sampled
				from the noisy AGG.
	"""
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})
	profiles = reduction_profiles(g, {"All":players})
	s = samples/len(profiles)
	extras = sample(profiles, samples - s*len(profiles))
	counts = {p:(s+1 if p in extras else s) for p in profiles}
	random_profiles = {}
	for prof,count in counts.items():
		dist = np.array([prof["All"].get(s,0) for s in AGG.strategies],
														dtype=float)
		dist /= float(AGG.players)
		for _ in range(count):
			rp = np.random.multinomial(AGG.players, dist)
			rp = filter(lambda p:p[1], zip(AGG.strategies,rp))
			rp = RSG.Profile({"All":dict(rp)})
			random_profiles[rp] = random_profiles.get(rp,0) + 1

	for prof,count in random_profiles.iteritems():
		values = AGG.sample(prof["All"], count)
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})

	return g


def sample_games(folder, players=[2], samples=[100], reductions=["HR","DPR"]):
	game_types = []
	if "HR" in reductions:
		game_types += ["at_HR", "near_HR"]
	if "DPR" in reductions:
		game_types += ["at_DPR", "near_DPR"]
	AGG_names = filter(lambda s: s.endswith(".json"), ls(folder))
	for AGG_name in sorted(AGG_names):
		with open(join(folder, AGG_name)) as f:
			AGG = LEG_to_AGG(json.load(f))
		for p in players:
			for n in samples:
				for game_type in game_types:
					sub_folder = game_type + "_"+str(p)+"p"+str(n)+"n"
					if exists(join(folder, sub_folder, AGG_name)):
						continue
					if game_type=="at_DPR":
						game = sample_at_reduction(AGG, n, DPR_profiles, p)
					elif game_type=="at_HR":
						game = sample_at_reduction(AGG, n, HR_profiles, p)
					elif game_type=="near_DPR":
						game = sample_near_reduction(AGG, n, DPR_profiles, p)
					elif game_type=="near_HR":
						game = sample_near_reduction(AGG, n, HR_profiles, p)
					write_game(game, folder, sub_folder, AGG_name)
					del game


def learn_games(folder, cross_validate=False):
	"""
	Goes through sub-folders of folder (which should have all been created by
	sample_games) and runs GP_learn on each sample game; creates a pkl file
	corresponding to each input json file

	If the folder has no sub-folders, learn_games instead learns all the json
	files that sit directly in that folder.
	"""
	sub_folders = filter(lambda s: isdir(join(folder, s)), ls(folder))
	if len(sub_folders) == 0:
		sub_folders = [folder]
	for sub_folder in sub_folders:
		for game_name in filter(lambda s: s.endswith(".json"), ls(folder)):
			GPs_name = join(folder, sub_folder, game_name[:-5] + "_GPs.pkl")
			if exists(GPs_name):
				continue
			game = read(join(folder, sub_folder, game_name))
			if type(game) != RSG.SampleGame:
				continue
			GPs = GP_learn(game, cross_validate)
			with open(GPs_name, "w") as f:
				cPickle.dump(GPs, f)


def write_game(game, base_folder, sub_folder, game_name):
	game_folder = join(base_folder, sub_folder)
	if not exists(game_folder):
		mkdir(game_folder)
	game_file = join(game_folder, game_name)
	with open(game_file, "w") as f:
		f.write(to_JSON_str(game))


def run_experiments(AGG_folder, samples_folder, exp_type, reduction, players,
					results_file, *args, **kwds):
	"""
	Extracting common code for running regrets and EVs experiments

	No extra args required for regrets_experiment.
	EVs_experiment can take points or GP_DPR_players.
	"""
	if exists(results_file):
		results = read(results_file)
	else:
		results = {}
	file_names = [f.split(".")[0] for f in sorted(filter(lambda f:
					f.endswith(".json"), ls(AGG_folder)))]
	for fn in file_names:
		if fn in results:
			continue
		AGG_fn = join(AGG_folder, fn + ".json")
		samples_fn = join(samples_folder, fn + ".json")
		GPs_fn = join(samples_folder, fn + "_GPs.pkl")
		with open(AGG_fn) as f:
			AGG = LEG_to_AGG(json.load(f))
		samples_game = read(samples_fn)
		reduced_game = reduction(samples_game, players)
		with open(GPs_fn) as f:
			GPs = cPickle.load(f)

		results[fn] = exp_type(AGG, samples_game, reduced_game, GPs,
								*args, **kwds)
		with open(results_file, "w") as f:
			json.dump(results, f)


def regrets_experiment(AGG, samples_game, reduced_game, GPs):
	"""
	"""
	results = {"reduction":{}, "GP":{}}

	reduced_eq = mixed_nash(reduced_game, at_least_one=True)
	results["reduction"]["eq"] = map(samples_game.toProfile, reduced_eq)
	results["reduction"]["regrets"] = [AGG.regret(e[0]) for e in reduced_eq]

	GP_eq = GP_RD(samples_game, GPs, at_least_one=True)
	results["GP"]["eq"] = map(samples_game.toProfile, GP_eq)
	results["GP"]["regrets"] = [AGG.regret(e[0]) for e in GP_eq]

	return results


def EVs_experiment(AGG, samples_game, reduced_game, GPs, sample_points=1000,
						GP_DPR_players=[3,5,7]):
	"""
	"""
	predictors = {"Y":GPs["Y"], "Yd":{}, "Ywd":{}}
	for role in samples_game.roles:
		predictors["Yd"][role] = {}
		predictors["Ywd"][role] = {}
		for strat in samples_game.strategies[role]:
			predictors["Yd"][role][strat] = DiffGP(GPs["Ym"][role],
												GPs["Yd"][role][strat])
			predictors["Ywd"][role][strat] = DiffGP(GPs["Ywm"][role],
												GPs["Ywd"][role][strat])
	GP_DPR_games = {p:GP_DPR(samples_game, predictors["Ywd"], p) for
												p in GP_DPR_players}
	results = {"AGG":{}, "reduction":{}, "GP_sample":{},
				"GP_point":{y:{} for y in predictors},
				"GP_DPR":{p:{} for p in GP_DPR_players}}

	for mix in [reduced_game.uniformMixture()] + mixture_grid(reduced_game, 5):
		try:
			prof = str(tuple(mix.flat))
			results["AGG"][prof] = tuple(AGG.expectedValues(mix[0]).flat)
			results["reduction"][prof] = \
							tuple(reduced_game.expectedValues(mix).flat)
			results["GP_sample"][prof] = tuple(GP_sample(samples_game, \
							predictors["Ywd"], mix, sample_points).flat)
			for y in ["Y","Yd","Ywd"]:
				results["GP_point"][y][prof] = tuple(GP_point(samples_game,
													predictors[y], mix).flat)
			for p in GP_DPR_players:
				results["GP_DPR"][p][prof] = \
								tuple(GP_DPR_games[p].expectedValues(mix).flat)
		except ValueError:
			continue
	return results


def mixture_grid(game, points=5):
	"""
	Cross-product of sym_mix_grid outputs for each role.
	"""
	role_mixtures = []
	for r,role in enumerate(game.roles):
		mix = sym_mix_grid(game.numStrategies[r], points)
		mix += [0]*(game.maxStrategies - game.numStrategies[r])
		role_mixtures.append(mix)
	return [h_array(p).squeeze(1) for p in  product(*role_mixtures)]


def sym_mix_grid(num_strats, points):
	"""
	Generate all choose(S, points) grid points in the simplex.

	There must be a better way to do this!
	"""
	a = np.linspace(0, 1, points)
	mixtures = set()
	for p in filter(lambda x: abs(sum(x) - 1) < .5/points, CwR(a,num_strats)):
		for m in permutations(p):
			mixtures.add(h_array([m]))
	return sorted(mixtures)


def main():
	p = ArgumentParser(description="Perform game-learning experiments on " +\
									"a set of action graph games.")
	p.add_argument("mode", type=str, choices=["games","learn","regrets","EVs"],
				help="games mode creates at_DPR, at_HR, near_DPR, and near_HR "+
				"directories. It requires players and samples arguments "+
				"(other modes don't). learn mode creates _GPs.pkl files. "+
				"regrets mode computes equilibria and regrets in all games. "+
				"EVs mode computes expected values of many mixtures in all "+
				"games.")
	p.add_argument("folder", type=str, help="Folder containing pickled AGGs.")
	p.add_argument("-p", type=int, nargs="*", default=[], help=\
				"Player sizes of reduced games to try. Only for 'games' mode.")
	p.add_argument("-n", type=int, nargs="*", default=[])
	p.add_argument("--skip", type=str, choices=["HR", "DPR", ""], default="",
				help="Don't generate games of the specified reduction type.")
	p.add_argument("--CV", action="store_true", help="Perform cross-validation")
	a = p.parse_args()
	if a.mode == "games":
		assert a.p
		assert a.n
		reductions = ["HR", "DPR"]
		if a.skip in reductions:
			reductions.remove(a.skip)
		sample_games(a.folder, a.p, a.n, reductions)
	elif a.mode == "learn":
		learn_games(a.folder, a.CV)
	else:
		assert len(a.p)==1, "please specify one reduction size with -p"
		reduction = HR if "HR" in basename(a.folder) else DPR
		if a.mode == "regrets":
			exp_type = regrets_experiment
			res_file = join(a.folder, "regrets_results.json")
		elif a.mode == "EVs":
			exp_type = EVs_experiment
			res_file = join(a.folder, "EVs_results.json")
		run_experiments(dirname(abspath(a.folder)), a.folder,
						exp_type, reduction, a.p[0], res_file)


if __name__ == "__main__":
	main()
