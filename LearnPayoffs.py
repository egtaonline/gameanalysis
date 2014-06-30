import numpy as np
from sklearn.gaussian_process import GaussianProcess as GP
from itertools import repeat
from sys import argv

from Reductions import DPR_profiles, full_prof_DPR, DPR
import RoleSymmetricGame as RSG
from Nash import mixed_nash
from BasicFunctions import average

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
	gps = {r:{s:None for s in game.strategies[r]} for r in game.roles}

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
			gps[role][strat] = gp
	return gps


def GP_DPR(game, players, gps=None):
	"""
	Estimate a DPR game from GP regression models.
	"""
	if len(game.roles) == 1 and isinstance(players, int):
		players = {game.roles[0]:players}
	elif isinstance(players, list):
		players = dict(zip(game.roles, players))

	if gps == None:
		gps = GP_learn(game)

	learned_game = RSG.Game(game.roles, players, game.strategies)
	for prof in learned_game.allProfiles():
		role_payoffs = {}		
		for role in game.roles:
			role_payoffs[role] = []			
			for strat,count in prof[role].iteritems():
				full_prof = full_prof_DPR(prof, role, strat, game.players)
				prof_x = prof2vec(game, full_prof)
				prof_y = gps[role][strat].predict(prof_x)
				role_payoffs[role].append(RSG.PayoffData(strat, count, prof_y))
		learned_game.addProfile(role_payoffs)

	return learned_game





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
	print "trial, reduction regret, learning regret"
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
		
		var = []
		for prof,count in random_profiles.iteritems():
			values = leg.sample(prof["All"], count)
			fg_learn.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
										s,c in prof["All"].iteritems()]})
		rg_learn = GP_DPR(fg_learn, players)

		NE_reduce = mixed_nash(rg_reduce, at_least_one = True)[0][0]
		NE_learn = mixed_nash(rg_learn, at_least_one = True)[0][0]

		print str(j) +", "+ str(leg.regret(NE_reduce)) +", "+ \
							str(leg.regret(NE_learn))



if __name__ == "__main__":
	main(int(argv[1]))