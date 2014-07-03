import numpy as np
from itertools import repeat
from sys import argv

# import GaussianProcess but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	from sklearn.gaussian_process import GaussianProcess as GP
except ImportError:
	warnings.warn("sklearn.gaussian_process is required for game learning.")

from Reductions import DPR_profiles, full_prof_DPR, DPR
import RoleSymmetricGame as RSG
from Nash import mixed_nash
from BasicFunctions import average
from HashableClasses import h_array

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
						x[role][strat].extend(repeat(c, len(samples)))
						y[role][strat].extend(samples)
					except AttributeError: #except will work on RSG.Game
						x[role][strat].append(c)
						y[role][strat].append(game.values[p][r,s])
	var = average(var)

	for role in game.roles:
		for strat in game.strategies[role]:
			gp = GP(storage_mode='light', normalize=False, nugget=var, \
					random_start=10)
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

	return mixed_nash(learned_game)


def GP_sampling_RD(game, GPs=None, regret_thresh=1e-2, dist_thresh=1e-3, \
					random_restarts=0, at_least_one=False, iters=10000, \
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



from ActionGraphGame import local_effect_AGG

def main(experiments):
	# run an AGG experiment
	players = {"All":4}
	samples = 20
	print "trial, method, regret"
	for j in range(experiments):
		leg = local_effect_AGG(41,5,2,3,100)

		fg_reduce = RSG.SampleGame(["All"], {"All":leg.players}, \
									{"All":leg.strategies})
		fg_learn = RSG.SampleGame(["All"], {"All":leg.players}, \
									{"All":leg.strategies})

		random_profiles = {}
		for prof in DPR_profiles(fg_reduce, players):
			values = leg.sample(prof["All"], samples)
			fg_reduce.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
										s,c in prof["All"].iteritems()]})
			counts = np.array([prof["All"].get(s,0) for s in leg.strategies])
			for i in range(samples):
				rp = np.random.multinomial(leg.players, counts / \
											float(leg.players))
				rp = filter(lambda p:p[1], zip(leg.strategies,rp))
				rp = RSG.Profile({"All":dict(rp)})
				random_profiles[rp] = random_profiles.get(rp,0) + 1
		rg_reduce = DPR(fg_reduce, players)
		NE_reduce = mixed_nash(rg_reduce, at_least_one = True)
		
		var = []
		for prof,count in random_profiles.iteritems():
			values = leg.sample(prof["All"], count)
			fg_learn.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
										s,c in prof["All"].iteritems()]})
		GPs = GP_learn(fg_learn)
		NE_DPR_learn = GP_DPR(fg_learn, players, GPs)
		NE_sample_learn = GP_sampling_RD(fg_learn, GPs)
		
		for eq in NE_reduce:
			print str(j) +", DPR, "+ str(leg.regret(eq[0]))
		for eq in NE_DPR_learn:
			print str(j) +", DPR_learn, "+ str(leg.regret(eq[0]))
		for eq in NE_sample_learn:
			print str(j) +", sample_learn, "+ str(leg.regret(eq[0]))



if __name__ == "__main__":
	main(int(argv[1]))
