import numpy as np
from itertools import repeat
from os import listdir as ls, mkdir
from os.path import join, exists
from cPickle import load, dump
from argparse import ArgumentParser

# import GaussianProcess but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	from sklearn.gaussian_process import GaussianProcess
except ImportError:
	warnings.warn("sklearn.gaussian_process is required for game learning.")

from Reductions import DPR_profiles, full_prof_DPR, DPR
import RoleSymmetricGame as RSG
from Nash import mixed_nash
from BasicFunctions import average
from HashableClasses import h_array
from ActionGraphGame import local_effect_AGG, Noisy_AGG
from GameIO import to_JSON_str, read
from itertools import combinations_with_replacement as CwR, permutations

def GP_learn(game):
	"""
	Create a GP regression for each role and strategy.

	Parameters:
	game:		RoleSymmetricGame.SampleGame object with enough data to
				estimate payoff functions.
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
						for i in range(len(samples)):
							x[role][strat].append(c + \
									np.random.normal(0,1e-6,c.shape))
						y[role][strat].extend(samples)
					except AttributeError: #except will work on RSG.Game
						x[role][strat].append(c)
						y[role][strat].append(game.values[p][r,s])

	for role in game.roles:
		for strat in game.strategies[role]:
			gp = GaussianProcess(storage_mode='light', theta0=10)
			gp.fit(x[role][strat], y[role][strat])
			GPs[role][strat] = gp
	return GPs


def GP_DPR(game, players, GPs=None):
	"""
	Estimate equilibria of a DPR game from GP regression models.
	"""
	if len(game.roles) == 1 and isinstance(players, int):
		players = {game.roles[0]:players}
	elif isinstance(players, list):
		players = dict(zip(game.roles, players))
	if GPs == None:
		GPs = GP_learn(game)

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


def GP_sampling_RD(game, GPs=None, regret_thresh=1e-2, dist_thresh=1e-3, \
					random_restarts=0, at_least_one=False, iters=1000, \
					converge_thresh=1e-6, ev_samples=100):
	"""
	Estimate equilibria with RD using random samples from GP regression models.
	"""
	if GPs == None:
		GPs = GP_learn(game)

	candidates = []
	regrets = {}
	for mix in game.biasedMixtures() + [game.uniformMixture() ]+ \
			[game.randomMixture() for _ in range(random_restarts)]:
		for _ in range(iters):
			old_mix = mix
			EVs = GP_EVs(game, mix, GPs, ev_samples)
			mix = (EVs - game.minPayoffs + RSG.tiny) * mix
			mix = mix / mix.sum(1).reshape(mix.shape[0],1)
			if np.linalg.norm(mix - old_mix) <= converge_thresh:
				break
		mix[mix < 0] = 0
		candidates.append(h_array(mix))
		EVs = GP_EVs(game, mix, GPs, ev_samples)
		regrets[h_array(mix)] = (EVs.max(1) - (EVs * mix).sum(1)).max()

	candidates.sort(key=regrets.get)
	equilibria = []
	for c in filter(lambda c: regrets[c] < regret_thresh, candidates):
		if all(np.linalg.norm(e - c, 2) >= dist_thresh for e in equilibria):
			equilibria.append(c)
	if len(equilibria) == 0 and at_least_one:
		return [min(candidates, key=regrets.get)]
	return equilibria


def GP_EVs(game, mix, GPs, samples=100):
	"""Mimics game.ExpectedValues via sampling from the GPs."""
	EVs = game.zeros()
	profs = [prof2vec(game, p) for p in sample_profiles(game, mix, samples)]
	for r,role in enumerate(game.roles):
		for s,strat in enumerate(game.strategies[role]):
			EVs[r,s] = GPs[role][strat].predict(profs).mean()
	return EVs


def GP_point(GPs, mix, players):
	prof = mix * players
	EVs = np.zeros(mix.shape)
	for r,role in enumerate(sorted(GPs)):
		for s,strat in enumerate(sorted(GPs[role])):
			EVs[r,s] = GPs[role][strat].predict(prof)
	return EVs


def sample_profiles(game, mix, count=1):
	"""
	Gives a list of pure-strategy profiles sampled from mix.

	Profiles returned are not necessarily unique.
	"""
	profiles = []
	for _ in range(count):
		prof = {}
		for r,role in enumerate(game.roles):
			prof[role] = {}
			rp = np.random.multinomial(game.players[role], mix[r][:\
												game.numStrategies[r]])
			for strat,count in zip(game.strategies[role], rp):
				if count > 0:
					prof[role][strat] = count
		profiles.append(RSG.Profile(prof))
	return profiles


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


def sample_near_DPR(AGG, players, samples=10):
	"""
	"""
	random_profiles = {}
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})

	for prof in DPR_profiles(g, {"All":players}):
		counts = np.array([prof["All"].get(s,0) for s in AGG.strategies])
		for i in range(max(1,samples)):
			rp = np.random.multinomial(AGG.players, counts / float(AGG.players))
			rp = filter(lambda p:p[1], zip(AGG.strategies,rp))
			rp = RSG.Profile({"All":dict(rp)})
			random_profiles[rp] = random_profiles.get(rp,0) + 1

	for prof,count in random_profiles.iteritems():
		values = AGG.sample(prof["All"], count)
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})

	return g


def learn_AGGs(folder, players=2, samples=10):
	"""
	Takes a folder full of action graph games and create sub-folders full of
	DPR, GP_DPR, and GP_sample games corresponding to each AGG.
	"""
	if not exists(join(folder, "DPR")):
		mkdir(join(folder, "DPR"))
	if not exists(join(folder, "samples")):
		mkdir(join(folder, "samples"))
	if not exists(join(folder, "GPs")):
		mkdir(join(folder, "GPs"))
	if not exists(join(folder, "GP_DPR")):
		mkdir(join(folder, "GP_DPR"))
	for AGG_fn, DPR_fn, samples_fn, GPs_fn, GP_DPR_fn in learned_files(folder):
		if exists(DPR_fn) and exists(samples_fn) and \
				exists(GPs_fn) and exists(GP_DPR_fn):
			continue
		with open(AGG_fn) as f:
			AGG = load(f)
		DPR_game = DPR(sample_at_DPR(AGG, players, samples), players)
		sample_game = sample_near_DPR(AGG, players, samples)
		GPs = GP_learn(sample_game)
		GP_DPR_game = GP_DPR(sample_game, players, GPs)
		with open(DPR_fn, "w") as f:
			f.write(to_JSON_str(DPR_game))
		with open(samples_fn, "w") as f:
			f.write(to_JSON_str(sample_game))
		with open(GPs_fn, "w") as f:
			dump(GPs,f)
		with open(GP_DPR_fn, "w") as f:
			f.write(to_JSON_str(GP_DPR_game))


def regrets_experiment(folder):
	"""
	Takes a folder filled with AGGs, plus the sub-folders for DPR, GP_DPR, and
	GP_sample created by the learn_AGGs() and computes equilibria in each small
	game, then outputs those equilibria and their regrets in the corresponding
	AGGs.
	"""
	DPR_eq = []
	GP_DPR_eq = []
	GP_sample_eq = []
	DPR_regrets = []
	GP_DPR_regrets = []
	GP_sample_regrets = []

	for AGG_fn, DPR_fn, samples_fn, GPs_fn, GP_DPR_fn in learned_files(folder):
		with open(AGG_fn) as f:
			AGG = load(f)
		DPR_game = read(DPR_fn)
		samples_game = read(samples_fn)
		with open(GPs_fn) as f:
			GPs = load(f)
		GP_DPR_game = read(GP_DPR_fn)

		eq = mixed_nash(DPR_game)
		DPR_eq.append(map(samples_game.toProfile, eq))
		DPR_regrets.append([AGG.regret(e[0]) for e in eq])
		eq = mixed_nash(GP_DPR_game)
		GP_DPR_eq.append(map(samples_game.toProfile, eq))
		GP_DPR_regrets.append([AGG.regret(e[0]) for e in eq])
		eq = GP_sampling_RD(samples_game, GPs)
		GP_sample_eq.append(map(samples_game.toProfile, eq))
		GP_sample_regrets.append([AGG.regret(e[0]) for e in eq])

	with open(join(folder, "DPR_eq.json"), "w") as f:
		f.write(to_JSON_str(DPR_eq))
	with open(join(folder, "DPR_regrets.json"), "w") as f:
		f.write(to_JSON_str(DPR_regrets))
	with open(join(folder, "GP_DPR_eq.json"), "w") as f:
		f.write(to_JSON_str(GP_DPR_eq))
	with open(join(folder, "GP_DPR_regrets.json"), "w") as f:
		f.write(to_JSON_str(GP_DPR_regrets))
	with open(join(folder, "GP_sample_eq.json"), "w") as f:
		f.write(to_JSON_str(GP_sample_eq))
	with open(join(folder, "GP_sample_regrets.json"), "w") as f:
		f.write(to_JSON_str(GP_sample_regrets))


def EVs_experiment(folder):
	"""
	Takes a folder filled with AGGs, plus the sub-folders for DPR, GP_DPR, and
	GPs created by learn_AGGs(), and computes expected values for a number of
	mixed strategies in the full game, in the DPR game and using several
	different methods to extract EVs from the GPs.
	"""
	print join(folder, "DPR", ls(join(folder, "DPR"))[0])
	DPR_game = read(join(folder, "DPR", ls(join(folder, "DPR"))[0]))
	mixtures = [DPR_game.uniformMixture()] +\
				mixture_grid(len(DPR_game.strategies["All"]))
	with open(join(folder, "mixtures.json"), "w") as f:
		f.write(to_JSON_str(mixtures))
	out_file = join(folder, "mixture_values.csv")
	with open(out_file, "w") as f:
		f.write("game,mixture,")
		f.write(",".join("AGG EV "+s for s in DPR_game.strategies["All"])+",")
		f.write(",".join("DPR EV "+s for s in DPR_game.strategies["All"])+",")
		f.write(",".join("GP_DPR EV " + s for s in \
						DPR_game.strategies["All"]) + ",")
		f.write(",".join("GP_sample EV " + s for s in \
						DPR_game.strategies["All"]) + ",")
		f.write(",".join("GP_point value " + s for s in \
						DPR_game.strategies["All"]) + "\n")

	for i, (AGG_fn, DPR_fn, samples_fn, GPs_fn, GP_DPR_fn) in \
								enumerate(learned_files(folder)):
		with open(AGG_fn) as f:
			AGG = load(f)
		DPR_game = read(DPR_fn)
		samples_game = read(samples_fn)
		with open(GPs_fn) as f:
			GPs = load(f)
		GP_DPR_game = read(GP_DPR_fn)
		for j,mix in enumerate(mixtures):
			line = [i,j]
			line.extend(AGG.expectedValues(mix[0]))
			line.extend(DPR_game.expectedValues(mix)[0])
			line.extend(GP_DPR_game.expectedValues(mix)[0])
			line.extend(GP_EVs(samples_game, mix, GPs)[0])
			line.extend(GP_point(GPs, mix, AGG.players)[0])
			with open(out_file, "a") as f:
				f.write(",".join(map(str, line)) + "\n")


def learned_files(folder):
	for fn in sorted(filter(lambda s: s.endswith(".pkl"), ls(folder))):
		AGG_fn = join(folder, fn)
		DPR_fn = join(folder, "DPR", fn[:-4]+".json")
		samples_fn = join(folder, "samples", fn[:-4]+".json")
		GPs_fn = join(folder, "GPs", fn)
		GP_DPR_fn = join(folder, "GP_DPR", fn[:-4]+".json")
		yield AGG_fn, DPR_fn, samples_fn, GPs_fn, GP_DPR_fn


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
				"games mode creates DPR, GPs, GP_DPR, and samples "+\
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
	a = p.parse_args()
	if a.mode == "games":
		assert a.p > 0 and a.s > -1
		learn_AGGs(a.folder, a.p, a.s)
	elif a.mode == "regrets":
		regrets_experiment(a.folder)
	elif a.mode =="EVs":
		EVs_experiment(a.folder)


if __name__ == "__main__":
	main()
