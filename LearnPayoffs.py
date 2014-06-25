import numpy as np
from sklearn.gaussian_process import GaussianProcess as GP
from itertools import repeat

from Reductions import DPR_profiles
import RoleSymmetricGame as RSG

def GP_learn(game, players):
	"""
	Create a GP regression game with the specified number of players.

	The returned game is constructed using deviation-preserving reduction
	after estimating the payoffs for the relevant profiles using Gaussian
	process regression.

	Parameters:
	game:		RoleSymmetricGame.SampleGame object with enough data to
				estimate payoff functions.
	players:	number of players per role in the output game; if symmetric
				players can be an int, otherwise it must map roles to ints.
	"""
	if len(game.roles) == 1 and isinstance(players, int):
		players = {game.roles[0]:players}
	elif isinstance(players, list):
		players = dict(zip(game.roles, players))

	x = {r:{s:[] for s in game.strategies[r]} for r in game.roles}
	y = {r:{s:[] for s in game.strategies[r]} for r in game.roles}
	gps = {r:{s:None for s in game.strategies[r]} for r in game.roles}

	for p in range(len(game)):
		c = prof2vec(game, game.counts[p])
		for r,role in enumerate(game.roles):
			for s,strat in enumerate(game.strategies[role]):
				if game.counts[p][r][s] > 0:
					try: #try will work on RSG.SampleGame
						samples = game.sample_values[p][r,s]
						x[role][strat].extend(repeat(c, len(samples)))
						y[role][strat].extend(samples)
					except AttributeError: #except will work on RSG.Game
						x[role][strat].append(c)
						y[role][strat].append(game.values[p][r,s])

	var = 1 #TODO: actually estimate the noise variance

	for role in game.roles:
		for strat in game.strategies[role]:
			gp = GP(storage_mode='light', normalize=False, nugget=var)
			gp.fit(x[role][strat], y[role][strat])
			gps[role][strat] = gp

	learned_game = Game(game.roles, players, game.strategies)
	for prof in learned_game.allProfiles():
		role_payoffs = {}		
		for role in game.roles:
			role_payoffs[r] = []			
			for strat,count in prof[role].iteritems():
				full_prof = full_prof_DPR(prof, role, strat, game.players)
				prof_x = prof2vec(game, full_prof)
				prof_y = gps[role][strat].predict(prof_x)
				role_payoffs[r].append(PayoffData(strat, count, prof_y))
		learned_game.addProfile(role_payoffs)

	return learned_game
	

def prof2vec(game, prof)
	"""
	Turns a profile (represented as Profile object or count array) into a
	1-D vector of strategy counts.
	"""
	if isinstance(prof, Profile):
		prof = game.toArray(prof)
	vec = []
	for r in range(len(game.roles))
		vec.extend(prof[r][:game.numStrategies[r]])
	return vec
