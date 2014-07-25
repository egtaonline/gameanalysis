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

def GP_learn(game, var_thresh=10):
	"""
	Create a GP regression for each role and strategy.

	Parameters:
	game:		RoleSymmetricGame.SampleGame object with enough data to
				estimate payoff functions.
	var_thresh:	minimum observations before a profile's variance is estimated.
	"""
	x = {r:{s:[] for s in game.strategies[r]} for r in game.roles}
	y = {r:{s:[] for s in game.strategies[r]} for r in game.roles}
	GPs = {r:{s:None for s in game.strategies[r]} for r in game.roles}

	var = []
	for p in range(len(game)):
		c = prof2vec(game, game.counts[p])
		for r,role in enumerate(game.roles):
			for s,strat in enumerate(game.strategies[role]):
				if game.counts[p][r][s] > 0:
					try: #try will work on RSG.SampleGame
						samples = game.sample_values[p][r,s]
						if len(samples) >= var_thresh:
							var.append(np.var(samples))
						for i in range(len(samples)):
							x[role][strat].append(c + \
									np.random.normal(0,1e-8,c.shape))
						y[role][strat].extend(samples)
					except AttributeError: #except will work on RSG.Game
						x[role][strat].append(c)
						y[role][strat].append(game.values[p][r,s])
	var = average(var)

	for role in game.roles:
		for strat in game.strategies[role]:
			gp = GaussianProcess(storage_mode='light', normalize=False, \
								nugget=var, random_start=10)
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
	for prof in sample_profiles(game, mix, samples):
		for r,role in enumerate(game.roles):
			for s,strat in enumerate(game.strategies[role]):
				EVs[r,s] += GPs[role][strat].predict(prof2vec(game,prof))
	EVs /= samples
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
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})
	return g


def sample_near_DPR(AGG, players, samples=10):
	"""
	"""
	random_profiles = {}
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})

	for prof in DPR_profiles(g, {"All":players}):
		counts = np.array([prof["All"].get(s,0) for s in AGG.strategies])
		for i in range(samples):
			rp = np.random.multinomial(AGG.players, counts / float(AGG.players))
			rp = filter(lambda p:p[1], zip(AGG.strategies,rp))
			rp = RSG.Profile({"All":dict(rp)})
			random_profiles[rp] = random_profiles.get(rp,0) + 1

	for prof,count in random_profiles.iteritems():
		values = AGG.sample(prof["All"], count)
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})

	return g


"""
The following three functions - learn_AGGs(), parse_args(), and main() - take
a folder full of action graph games and create sub-folders full of DPR, GP_DPR,
and GP_sample games corresponding to each AGG.
"""
def learn_AGGs(directory, players=2, samples=10):
	"""
	"""
	for fn in filter(lambda s: s.endswith(".pkl"), ls(directory)):
		with open(join(directory, fn)) as f:
			AGG = load(f)
		DPR_game = sample_at_DPR(AGG, players, samples)
		sample_game = sample_near_DPR(AGG, players, samples)
		GPs = GP_learn(sample_game, samples/2)
		if not exists(join(directory, "DPR")):
			mkdir(join(directory, "DPR"))
		if not exists(join(directory, "samples")):
			mkdir(join(directory, "samples"))
		if not exists(join(directory, "GPs")):
			mkdir(join(directory, "GPs"))
		with open(join(directory, "DPR", fn[:-4]+".json"), "w") as f:
			f.write(to_JSON_str(DPR_game))
		with open(join(directory, "samples", fn[:-4]+".json"), "w") as f:
			f.write(to_JSON_str(sample_game))
		with open(join(directory, "GPs", fn), "w") as f:
			dump(GPs,f)

def parse_args():
	p = ArgumentParser(description="Perform game-learning experiments on " +\
									"a set of action graph games.")
	p.add_argument("folder", type=str, help="Folder containing pickled AGGs.")
	p.add_argument("players", type=int, help="Number of players in DPR game.")
	p.add_argument("samples", type=int, help="Samples drawn per DPR profile.")
	return p.parse_args()

def main():
	a = parse_args()
	learn_AGGs(a.folder, a.players, a.samples)


"""
The following main() function takes a folder filled with AGGs, plus the
sub-folders for DPR, GP_DPR, and GP_sample created by the previous section
and computes equilibria in each small game, then outputs those equilibria and
their regrets in the corresponding AGGs.
"""
#def main():
#	p = ArgumentParser()
#	p.add_argument("folder", type=str, help="Base dir for DPR, GPS, samples.")
#	a = p.parse_args()
#
#	DPR_eq = []
#	GP_DPR_eq = []
#	GP_sample_eq = []
#
#	DPR_files = sorted(ls(join(a.folder, "DPR")))
#	samples_files = sorted(ls(join(a.folder, "samples")))
#	GP_files = sorted(ls(join(a.folder, "GPs")))
#
#	for DPR_fn, sam_fn, GP_fn in zip(DPR_files, samples_files, GP_files):
#		DPR_game = read(join(a.folder, "DPR", DPR_fn))
#		samples_game = read(join(a.folder, "samples", sam_fn))
#		with open(join(a.folder, "GPs", GP_fn)) as f:
#			GPs = load(f)
#		eq = mixed_nash(DPR_game)
#		DPR_eq.append(map(DPR_game.toProfile, eq))
#		eq = mixed_nash(GP_DPR(samples_game, DPR_game.players, GPs))
#		GP_DPR_eq.append(map(samples_game.toProfile, eq))
#		eq = GP_sampling_RD(samples_game, GPs)
#		GP_sample_eq.append(map(samples_game.toProfile, eq))
#
#	with open(join(a.folder, "DPR_eq.json"), "w") as f:
#		f.write(to_JSON_str(DPR_eq))
#	with open(join(a.folder, "GP_DPR_eq.json"), "w") as f:
#		f.write(to_JSON_str(GP_DPR_eq))
#	with open(join(a.folder, "GP_sample_eq.json"), "w") as f:
#		f.write(to_JSON_str(GP_sample_eq))


if __name__ == "__main__":
	main()
