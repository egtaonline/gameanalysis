from itertools import product
from math import isnan

from RoleSymmetricGame import *

def subgame(game, strategies={}):
	"""
	Creates a game with a subset each role's strategies.

	default settings result in a subgame with no strategies
	"""
	if not strategies:
		strategies = {r:[] for r in game.roles}
	sg = Game(game.roles, game.counts, strategies)
	for p in sg.allProfiles():
		if p in game:
			sg.addProfile({r:[payoff_data(s, p[r][s], game.getPayoff(p,r,s)) \
					for s in p[r]] for r in p})
	return sg


def isSubgame(small_game, big_game):
	if any((r not in big_game.roles for r in small_game.roles)):
		return False
	if any((small_game.counts[r] != big_game.counts[r] for r \
			in small_game.roles)):
		return False
	for r in small_game.roles:
		if any((s not in big_game.strategies[r] for s in \
				small_game.strategies[r])):
			return False
	return True


def cliques(full_game, subgames=set()):
	"""
	Finds maximal subgames for which all profiles are known.

	input:
	subgames = known complete subgames to be expanded (any payoff data in
	the known subgames is ignored, so for faster loading, give only the
	header information).
	"""
	subgames = {full_game.subgame(g.strategies) for g in subgames}.union(\
			{subgame(full_game)})
	maximal_subgames = set()
	while(subgames):
		sg = subgames.pop()
		maximal = True
		for role in full_game.roles:
			for s in set(full_game.strategies[role]) - \
					set(sg.strategies[role]):
				new_sg = subgame(full_game, {r:list(sg.strategies[r])\
						+ ([s] if r == role else []) for r in full_game.roles})
				if not new_sg.isComplete():
					continue
				maximal=False
				if new_sg in subgames or new_sg in maximal_subgames:
					continue
				if any([isSubgame(new_sg, g) for g in \
							subgames.union(maximal_subgames)]):
					continue
				subgames.add(new_sg)
		if maximal:
			maximal_subgames.add(sg)
	return maximal_subgames


def IteratedElimination(game, criterion, *args, **kwargs):
	"""
	iterated elimination of dominated strategies

	input:
	criterion = function to find dominated strategies
	"""
	sg = criterion(game)
	if game == sg:
		return game
	return IteratedElimination(sg, criterion, *args, **kwargs)


def NeverBestResponse(game, conditional=True):
	"""
	conditional never-a-weak-best-response criterion for IEDS

	This criterion is very strong: it can eliminate strict Nash equilibria.
	"""
	best_responses = {r:set() for r in game.roles}
	for profile in game:
		for r in game.roles:
			for s in profile[r]:
				br, unknown = game.bestResponses(profile, r, s)
				best_responses[r].update(br)
				if conditional:
					best_responses[r].update(unknown)
	return subgame(game, best_responses)


def PureStrategyDominance(game, conditional=True):
	"""
	conditional strict pure-strategy dominance criterion for IEDS
	"""
	undominated = {r:set(game.strategies[r]) for r in game.roles}
	for r in game.roles:
		for dominant, dominated in product(game.strategies[r], repeat=2):
			if dominant == dominated or dominated not in undominated[r]:
				continue
			regret = float("nan")
			for profile in game:
				if dominated not in profile[r]:
					continue
				try:
					regret = game.regret(profile, r, dominated, dominant)
				except KeyError:
					if conditional:
						regret = float("nan")
						break
				if regret <= 0:
					break
			if regret > 0:
				undominated[r].remove(dominated)
	return subgame(game, undominated)


def MixedStrategyDominance(game):
	"""
	conditional strict mixed-strategy dominance criterion for IEDS
	"""
	raise NotImplementedError("TODO")


def pureNash(game, epsilon=0):
	"""
	Finds all pure-strategy epsilon-Nash equilibria.

	input:
	epsilon = largest allowable regret for approximate equilibria

	output:
	NE = exact Nash equilibria
	eNE = e-Nash equilibria for 0 < e <= epsilon
	mrp = minimum regret profile (of interest if there is no exact NE)
	mr = regret of mrp
	"""
	equilibria = []
	for profile in game:
		try:
			if game.regret(profile) <= epsilon:
				equilibria.append(profile)
		except KeyError:
			continue
	return equilibria


def minRegretProfile(game):
	raise NotImplementedError("TODO")


def mixedNash(game, regret_thresh=0, dist_thresh=1e-2, verbose=False):
	raise NotImplementedError("TODO")


def RD(game, mixedProfile=None, array_data=None, iters=10000, thresh=1e-8, \
		verbose=False):
	"""
	Replicator dynamics.
	"""
	raise NotImplementedError("TODO")

