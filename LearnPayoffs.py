import numpy as np
from numpy.random import multinomial
from os import listdir as ls, mkdir
from os.path import join, exists
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

from Reductions import DPR_profiles, full_prof_DPR, DPR
import RoleSymmetricGame as RSG
from Nash import mixed_nash
from BasicFunctions import average, nCr
from HashableClasses import h_array
from ActionGraphGame import LEG_to_AGG
from GameIO import to_JSON_str, read
from itertools import combinations_with_replacement as CwR, permutations

def GP_learn(game, cross_validate=False, diffs=False):
	"""
	Create a GP regression for each role and strategy.

	Parameters:
	game:			RoleSymmetricGame.SampleGame object with enough data to
					estimate payoff functions.
	cross_validate:	If set to True, cross-validation will be used to select
					parameters of the GPs.
	diffs:			If set to True, GPs will not learn a strategy's payoff, but
					rather the difference between the strategy's payoff and the
					mean payoff for the profile.
	"""
	x = {r:{s:[] for s in game.strategies[r]} for r in game.roles}
	y = {r:{s:[] for s in game.strategies[r]} for r in game.roles}
	GPs = {r:{s:None for s in game.strategies[r]} for r in game.roles}

	for p in range(len(game)):
		c = prof2vec(game, game.counts[p])
		for r,role in enumerate(game.roles):
			for s,strat in enumerate(game.strategies[role]):
				if game.counts[p][r][s] > 0:
					try: #try will work on RSG.SampleGame
						samples = game.sample_values[p][r,s]
						if diffs:
							samples -= samples.mean()
						for i in range(len(samples)):
							x[role][strat].append(c + \
									np.random.normal(0,1e-6,c.shape))
						y[role][strat].extend(samples)
					except AttributeError: #except will work on RSG.Game
						x[role][strat].append(c)
						y[role][strat].append(game.values[p][r,s])

	for role in game.roles:
		for strat in game.strategies[role]:
			if cross_validate:
				gp = GaussianProcess(storage_mode='light', random_start=3, \
									thetaL=1e-4, thetaU=1e9, normalize=True)
				params = {"corr":["absolute_exponential","squared_exponential",\
						"cubic","linear"], "nugget":[1e-14,1e-10,1e-6,.01,1,\
						100,1e4]}
				cv = GridSearchCV(gp, params)
				cv.fit(x[role][strat], y[role][strat])
				GPs[role][strat] = cv.best_estimator_
			else:
				gp = GaussianProcess(storage_mode='light', corr="cubic", \
									nugget=1)
				gp.fit(x[role][strat], y[role][strat])
				GPs[role][strat] = gp
	return GPs


def GP_DPR(game, players, GPs):
	"""
	Estimate equilibria of a DPR game from GP regression models.
	"""
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


def GP_EVs(GPs, mix, players, samples=1000):
	"""
	Mimics game.ExpectedValues via sampling from the GPs.

	WARNING: assumes that the game is symmetric!
	"""
	EVs = np.zeros(mix.shape)
	for r,role in enumerate(sorted(GPs)):
		for s,strat in enumerate(sorted(GPs[role])):
			profs = multinomial(players, mix[0], samples)
			EVs[r,s] = GPs[role][strat].predict(profs).mean()
	return EVs


def GP_point(GPs, mix, players):
	"""
	Mimics game.ExpectedValues by returning the GPs' value estimates for the ML
	profile under mix.

	WARNING: assumes that the game is symmetric!
	"""
	prof = mix * players
	EVs = np.zeros(mix.shape)
	for r,role in enumerate(sorted(GPs)):
		for s,strat in enumerate(sorted(GPs[role])):
			EVs[r,s] = GPs[role][strat].predict(prof)
	return EVs


def GP_RD(game, GPs, regret_thresh=1e-2, dist_thresh=1e-3, \
			random_restarts=0, at_least_one=False, iters=1000, \
			converge_thresh=1e-6, ev_samples=1000, EV_func=GP_EVs):
	"""
	Estimate equilibria with RD using from GP regression models.
	"""
	candidates = []
	regrets = {}
	players = game.players.values()[0] # assumes game is symmetric
	for mix in game.biasedMixtures() + [game.uniformMixture() ]+ \
			[game.randomMixture() for _ in range(random_restarts)]:
		for _ in range(iters):
			old_mix = mix
			EVs = EV_func(GPs, mix, players, ev_samples)
			mix = (EVs - game.minPayoffs + RSG.tiny) * mix
			mix = mix / mix.sum(1).reshape(mix.shape[0],1)
			if np.linalg.norm(mix - old_mix) <= converge_thresh:
				break
		mix[mix < 0] = 0
		candidates.append(h_array(mix))
		EVs = EV_func(GPs, mix, players, ev_samples)
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


def sample_at_DPR(AGG, players, samples=10):
	"""
	"""
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})
	for prof in DPR_profiles(g, {"All":players}):
		values = AGG.sample(prof["All"], samples)
		if all(hasattr(v,"__len__") for v in values.values()):
			g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
									s,c in prof["All"].iteritems()]})
		else: #only necessary because of old Noisy_AGGs returning non-lists
			g.addProfile({"All":[RSG.PayoffData(s,c,[values[s]]) for \
									s,c in prof["All"].iteritems()]})
	return g


def sample_near_DPR(AGG, players, samples_per_profile=10, total_samples=0):
	"""
	If samples_per_profile > 0 then total_samples is ignored, and each DPR
	profile gets the same number of samples drawn near it. Otherwise,
	total_samples are allocated so that each DPR profile gets the same number
	+/- 1. The profiles that get one extra sample are chosen uniformly at
	random.
	"""
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})
	profiles = DPR_profiles(g, {"All":players})
	np.random.shuffle(profiles)
	if samples_per_profile > 0:
		total_samples = len(profiles) * samples_per_profile
	else:
		samples_per_profile = total_samples / len(profiles)
	remainder = total_samples % len(profiles)
	random_profiles = {}
	for i,prof in enumerate(profiles):
		counts = np.array([prof["All"].get(s,0) for s in AGG.strategies])
		for _ in range(samples_per_profile + (1 if i < remainder else 0)):
			rp = np.random.multinomial(AGG.players, counts / float(AGG.players))
			rp = filter(lambda p:p[1], zip(AGG.strategies,rp))
			rp = RSG.Profile({"All":dict(rp)})
			random_profiles[rp] = random_profiles.get(rp,0) + 1

	for prof,count in random_profiles.iteritems():
		values = AGG.sample(prof["All"], count)
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})

	return g


def learn_AGGs(folder, players=2, samples=10, CV=False, diffs=False, extras=0):
	"""
	Takes a folder full of action graph games and create sub-folders full of
	DPR, GP_sample games corresponding to each AGG.
	"""
	if not exists(join(folder, "DPR")):
		mkdir(join(folder, "DPR"))
	if not exists(join(folder, "samples")):
		mkdir(join(folder, "samples"))
	if not exists(join(folder, "GPs")):
		mkdir(join(folder, "GPs"))

	AGG_fn = join(folder, filter(lambda s: s.endswith(".json"), ls(folder))[0])
	with open(AGG_fn) as f:
		strategies = len(json.load(f)["strategies"])
		total_samples = samples * nCr(players + strategies - 1, players)

	for AGG_fn, DPR_fn, samples_fn, GPs_fn in learned_files(folder):
		if exists(DPR_fn) and exists(samples_fn) and exists(GPs_fn):
			continue
		with open(AGG_fn) as f:
			AGG = LEG_to_AGG(json.load(f))
		DPR_game = DPR(sample_at_DPR(AGG, players, samples), players)
		with open(DPR_fn, "w") as f:
			f.write(to_JSON_str(DPR_game))
		del DPR_game
		sample_game = sample_near_DPR(AGG, players + extras, 0, total_samples)
		del AGG
		with open(samples_fn, "w") as f:
			f.write(to_JSON_str(sample_game))
		GPs = GP_learn(sample_game, CV, diffs)
		del sample_game
		with open(GPs_fn, "w") as f:
			cPickle.dump(GPs,f)
		del GPs


def regrets_experiment(folder):
	"""
	Takes a folder filled with AGGs, plus the sub-folders for DPR and GPs
	created by the learn_AGGs() and computes equilibria in each small game,
	then outputs those equilibria and their regrets in the corresponding AGGs.
	"""
	DPR_eq_file = join(folder, "DPR_eq.json")
	DPR_regrets_file = join(folder, "DPR_regrets.json")
	GP_eq_file = join(folder, "GP_eq.json")
	GP_regrets_file = join(folder, "GP_regrets.json")
	if exists(GP_eq_file):
		DPR_eq = read(DPR_eq_file)
		DPR_regrets = read(DPR_regrets_file)
		GP_eq = read(GP_eq_file)
		GP_regrets = read(GP_regrets_file)
	else:
		DPR_eq = []
		GP_eq = []
		DPR_regrets = []
		GP_regrets = []

	for i, (AGG_fn, DPR_fn, samples_fn, GPs_fn) in \
					enumerate(learned_files(folder)):
		if i < len(GP_eq):
			continue
		with open(AGG_fn) as f:
			AGG = LEG_to_AGG(json.load(f))
		DPR_game = read(DPR_fn)
		samples_game = read(samples_fn)
		with open(GPs_fn) as f:
			GPs = cPickle.load(f)

		eq = mixed_nash(DPR_game, at_least_one=True)
		DPR_eq.append(map(samples_game.toProfile, eq))
		DPR_regrets.append([AGG.regret(e[0]) for e in eq])
		eq = GP_RD(samples_game, GPs, at_least_one=True)
		GP_eq.append(map(samples_game.toProfile, eq))
		GP_regrets.append([AGG.regret(e[0]) for e in eq])

		with open(DPR_eq_file, "w") as f:
			f.write(to_JSON_str(DPR_eq))
		with open(join(folder, "DPR_regrets.json"), "w") as f:
			f.write(to_JSON_str(DPR_regrets))
		with open(join(folder, "GP_eq.json"), "w") as f:
			f.write(to_JSON_str(GP_eq))
		with open(join(folder, "GP_regrets.json"), "w") as f:
			f.write(to_JSON_str(GP_regrets))


def EVs_experiment(folder):
	"""
	Takes a folder filled with AGGs, plus the sub-folders for DPR and GPs
	created by learn_AGGs(), and computes expected values for a number of
	mixed strategies in the full game, in the DPR game and using several
	different methods to extract EVs from the GPs.
	"""
	DPR_game = read(join(folder, "DPR", ls(join(folder, "DPR"))[0]))
	mixtures = [DPR_game.uniformMixture()] +\
				mixture_grid(len(DPR_game.strategies["All"]))
	if not exists(join(folder, "mixtures.json")):
		with open(join(folder, "mixtures.json"), "w") as f:
			f.write(to_JSON_str(mixtures))
	out_file = join(folder, "mixture_values.csv")
	if not exists(out_file):
		last_game = 0
		last_mix = -1
		with open(out_file, "w") as f:
			f.write("game,mixture,")
			f.write(",".join("AGG EV "+s for s in \
							DPR_game.strategies["All"]) + ",")
			f.write(",".join("DPR EV "+s for s in \
							DPR_game.strategies["All"]) + ",")
			f.write(",".join("GP_sample EV " + s for s in \
							DPR_game.strategies["All"]) + ",")
			f.write(",".join("GP_point value " + s for s in \
							DPR_game.strategies["All"]) + "\n")
	else:
		with open(out_file) as f:
			last_line = [l for l in f][-1].split(",")
		last_game = int(last_line[0])
		last_mix = int(last_line[1])

	for i, (AGG_fn, DPR_fn, samples_fn, GPs_fn) in \
								enumerate(learned_files(folder)):
		if i < last_game:
			continue
		with open(AGG_fn) as f:
			AGG = LEG_to_AGG(json.load(f))
		DPR_game = read(DPR_fn)
		samples_game = read(samples_fn)
		with open(GPs_fn) as f:
			GPs = cPickle.load(f)
		for j,mix in enumerate(mixtures):
			if i == last_game and j <= last_mix:
				continue
			line = [i,j]
			line.extend(AGG.expectedValues(mix[0]))
			line.extend(DPR_game.expectedValues(mix)[0])
			line.extend(GP_EVs(GPs, mix, AGG.players)[0])
			line.extend(GP_point(GPs, mix, AGG.players)[0])
			with open(out_file, "a") as f:
				f.write(",".join(map(str, line)) + "\n")


def learned_files(folder):
	for fn in sorted(filter(lambda s: s.endswith(".json"), ls(folder))):
		AGG_fn = join(folder, fn)
		DPR_fn = join(folder, "DPR", fn)
		samples_fn = join(folder, "samples", fn)
		GPs_fn = join(folder, "GPs", fn[:-4] + "pkl")
		yield AGG_fn, DPR_fn, samples_fn, GPs_fn


def mixture_grid(S, points=5, digits=2):
	"""
	Generate all choose(S, points) grid points in the simplex.

	There must be a better way to do this!
	"""
	a = np.linspace(0, 1, points)
	mixtures = set()
	for p in filter(lambda x: abs(sum(x) - 1) < .5/points, CwR(a,S)):
		for m in permutations(map(lambda x: round(x, digits), p)):
			mixtures.add(h_array([m]))
	return sorted(mixtures)


def main():
	p = ArgumentParser(description="Perform game-learning experiments on " +\
									"a set of action graph games.")
	p.add_argument("mode", type=str, choices=["games","regrets","EVs"], help=\
				"games mode creates DPR, GPs, and samples "+\
				"directories. It requires players and samples arguments "+\
				"(other modes don't). regrets mode computes equilibria and "+\
				"regrets in all games. EVs mode computes expected values of "+\
				"many mixtures in all games.")
	p.add_argument("folder", type=str, help="Folder containing pickled AGGs.")
	p.add_argument("-p", type=int, default=0, help=\
				"Number of players in DPR game. Only for 'games' mode.")
	p.add_argument("-s", type=int, default=-1, help=\
				"Samples drawn per DPR profile. Only for 'games' mode. Set "+\
				"to 0 for exact values.")
	p.add_argument("-e", type=int, default=0, help="Extra players for GP.")
	p.add_argument("--CV", action="store_true", help="Perform cross-validation")
	p.add_argument("--diff", action="store_true", help="Learn differences "+\
					"from mean payoffs.")
	a = p.parse_args()
	if a.mode == "games":
		assert a.p > 0 and a.s > -1
		learn_AGGs(a.folder, a.p, a.s, a.CV, a.diff, a.e)
	elif a.mode == "regrets":
		regrets_experiment(a.folder)
	elif a.mode =="EVs":
		EVs_experiment(a.folder)


if __name__ == "__main__":
	main()
