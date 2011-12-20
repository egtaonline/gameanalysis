#!/usr/local/bin/python2.7

import numpy as np

import GameIO
import RoleSymmetricGame as RSG

def cliques(full_game, subgames=set()):
	"""
	Finds maximal subgames for which all profiles are known.

	input:
	subgames = known complete subgames to be expanded (any payoff data in
	the known subgames is ignored, so for faster loading, give only the
	header information).
	"""
	subgames = {full_game.subgame(strategies=g.strategies) for g in \
			subgames}.union({full_game.subgame()})
	maximal_subgames = set()
	while(subgames):
		subgame = subgames.pop()
		maximal = True
		for role in full_game.roles:
			for s in set(full_game.strategies[role]) - \
					set(subgame.strategies[role]):
				new_subgame = full_game.subgame(strategies={r:list(\
						subgame.strategies[r]) + ([s] if r == role \
						else []) for r in full_game.roles})
				if new_subgame.size != len(new_subgame):
					continue
				maximal=False
				if new_subgame in subgames or new_subgame in maximal_subgames:
					continue
				if any([new_subgame.isSubgame(g) for g in \
							subgames.union(maximal_subgames)]):
					continue
				subgames.add(new_subgame)
		if maximal:
			maximal_subgames.add(subgame)
	return maximal_subgames


def IEDS(game, criterion):
	"""
	iterated elimination of dominated strategies

	input:
	criterion = function to find dominated strategies
	"""
	subgame = criterion(game)
	if game == subgame:
		return game
	return IEDS(subgame, criterion)


def CNWBR(game):
	"""
	conditional never-a-weak-best-response criterion for IEDS

	This criterion is very strong: it can eliminate strict Nash equilibria.

	The criterion is 'conditional' in that it will eliminate strategies that
	are never weak best-responses given all available data in a partial game.
	"""
	best_responses = {r:set() for r in game.roles}
	for profile in game:
		for role in profile:
			for strategy in profile[role].getStrategies():
				best_responses[role].update(game.BR(role, \
						profile.remove(role, strategy)))
	return game.subgame(strategies=best_responses)


def CPSD(game):
	"""
	conditional pure-strategy dominance criterion for IEDS

	The criterion is 'conditional' in that it will eliminate strategies that
	are dominated given all available data in a partial game.
	"""
	dominated = {r:set() for r in game.roles}
	for r in game.roles:
		for s1, s2 in RSG.product(game.strategies[r], repeat=2):
			for profile in game:
				v1 = v2 = float('-inf')
				try:
					v1 = game.getPayoff(profile, r, s1)
					v2 = game.getPayoff(profile.remove(r,s1).add(r,s2), r, s2)
					if v1 >= v2:
						break
				except KeyError:
					continue
			if v1 < v2:
				dominated[r].add(s1)
	return game.subgame(strategies={r:set(game.strategies[r]) - dominated[r] \
			for r in game.roles})


def PSD(game):
	"""
	confirmed pure-strategy dominance criterion for IEDS

	This criterion differs from CPSD in that it will only declare a strategy
	dominated if all profiles in which both it and the dominating strategy
	appear have payoff data available.
	"""
	dominated = {r:set() for r in game.roles}
	for r in game.roles:
		for s1, s2 in RSG.product(game.strategies[r], repeat=2):
			data_missing = False
			for profile in game:
				v1 = v2 = float('-inf')
				try:
					v1 = game.getPayoff(profile, r, s1)
					v2 = game.getPayoff(profile.remove(r,s1).add(r,s2), r, s2)
					if v1 >= v2:
						break
				except KeyError:
					data_missing = True
					break
			if v1 < v2 and not data_missing:
				dominated[r].add(s1)
	return game.subgame(strategies={r:set(game.strategies[r]) - dominated[r] \
			for r in game.roles})


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
	NE = set()
	eNE = set()
	mrp = None
	mr = float("inf")
	for profile in game:
		try:
			r = game.exactRegret(profile)
		except KeyError:
			continue
		if r < mr:
			mr = r
			mrp = profile
		if r <= 0:
			NE.add(profile)
		elif r <= epsilon:
			eNE.add(profile)
	return NE, eNE, mrp, mr


def mixedNash(game, regret_thresh=0, dist_thresh=1e-2, verbose=False):
	arrays = [(prof.countArray(game), prof.valueArray(game), \
			prof.repetitionsArray(game)) for prof in game]
	MNE_candidates = [RD(game, array_data=arrays, verbose=verbose)]
	for r in game.roles:
		for s in game.strategies[r]:
			eq = RD(game, game.biasedMixedProfile(r,s), verbose=verbose)
			if all(map(lambda e: e.dist(eq) > dist_thresh, MNE_candidates)):
				MNE_candidates.append(eq)
	regrets = {eq:game.exactRegret(eq) for eq in MNE_candidates}
	mrp = min(regrets, key=regrets.get)
	eMNE = filter(lambda eq: regrets[eq] <= regret_thresh, MNE_candidates)
	return eMNE, mrp


def RD(game, mixedProfile=None, array_data=None, iters=10000, thresh=1e-8, \
		verbose=False):
	"""
	Replicator dynamics.
	"""
	if not mixedProfile:
		mixedProfile = game.uniformMixedProfile()
	if not array_data:
		array_data = [(prof.countArray(game), prof.valueArray(game), \
				prof.repetitionsArray(game)) for prof in game]
	mix = mixedProfile.probArray(game)
	minPayoffs = np.zeros(mix.shape, dtype=float)
	for i,r in enumerate(game.roles):
		minPayoffs[:,i].fill(min(game.payoffList(r)))

	e = np.finfo(np.float64).tiny
	for i in range(iters):
		old_mix = mix
		EVs = sum([payoff_arr * (mix**count_arr).prod() * reps_arr / (mix+e) \
				for count_arr, payoff_arr, reps_arr in array_data])
		mix = (EVs - minPayoffs) * mix
		mix /= sum(mix)
		if max(abs(mix - old_mix).flat) <= thresh:
			break
	eq = RSG.Profile({r:RSG.SymmetricProfile([RSG.mixture(game.strategies[r],\
			mix[:len(game.strategies[r]), i])]*game.counts[r]) for i,r in \
			enumerate(game.roles)})
	if verbose:
		print "iterations =", i, "...", eq, "... regret =", game.exactRegret(eq)
	return eq


from os.path import abspath
from argparse import ArgumentParser

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("file", type=str, help="Game file to be analyzed. " +\
			"Suported file types: EGAT symmetric XML, EGAT strategic XML, " +\
			"testbed role-symmetric JSON.")
	parser.add_argument("-r", metavar="REGRET", type=float, default=0, \
			help="Max allowed regret for approximate Nash equilibria")
	parser.add_argument("-d", metavar="DISTANCE", type=float, default=1e-2, \
			help="L2-distance threshold to consider equilibria distinct")
#	parser.add_argument("--subgames", type=str, help="optinal files " +\
#			"containing known full subgames; useful for speeding up " +\
#			"clique-finding", default = "", nargs="*")
	args = parser.parse_args()
	input_game = GameIO.readGame(args.file)
	print "input game =", abspath(args.file), "\n", input_game, "\n\n"

	#iterated elimination of pure-dominated strategies
	rational_game = IEDS(input_game, CPSD)
	eliminated = {r:sorted(set(input_game.strategies[r]).difference( \
			rational_game.strategies[r])) for r in filter(lambda role: \
			input_game.strategies[role] != rational_game.strategies[role], \
			input_game.roles)}
	print "strategies removed by IEDS:"
	print (eliminated if eliminated else "none"), "\n\n"

	#pure strategy Nash equilibrium search
	PNE, ePNE, mrp, mr = pureNash(rational_game, args.d)
	if PNE:
		print len(PNE), "exact pure strategy Nash equilibria:\n", \
				RSG.list_repr(PNE, sep="\n"), "\n"
	if ePNE:
		print len(ePNE), "approximate pure strategy Nash equilibria", \
				"(0 < epsilon <= " + str(args.d) + "):\n", \
				RSG.list_repr(map(lambda eq: str(eq) + ", regret=" + \
				str(rational_game.exactRegret(eq)), ePNE), sep="\n"), "\n"
	print "minimum regret pure profile:", mrp, "\nregret =", mr, "\n\n"

	#clique finding
	print "cliques:"
	if len(rational_game) == rational_game.size:
		maximal_subgames = {rational_game}
		print "input game is maximal\n\n"
	else:
		maximal_subgames = cliques(rational_game)
		print "found", len(maximal_subgames), "maximal subgames\n\n"

	#mixed strategy Nash equilibrium search over maximal complete subgames
	for i, subgame in enumerate(sorted(maximal_subgames)):
		print "\nclique", i+1, ":", subgame.strategies, "\n"
		eMNE, mrmp = mixedNash(subgame, args.r, args.d)
		if eMNE:
			print "RD found", len(eMNE), "approximate symmetric mixed strategy"\
					+ " Nash equilibria:"
			print RSG.list_repr(map(lambda eq: str(eq) + \
					"\n\tclique regret:\t\t" + str(subgame.exactRegret(eq)) + \
					"\n\tfull game regret:\t" + str(rational_game.\
					confirmedRegret(eq)[0]) + "\n\tbest deviation:\t\t" + str( \
					rational_game.confirmedRegret(eq)[1]), eMNE), sep="\n"),"\n"
		else:
			print "no approximate equilibria with regret at most", args.r
			print "lowest regret symmetric mixed profile found by RD:"
			print mrmp
			print "\tclique regret:\t\t" + str(subgame.exactRegret(mrmp))
			fg_regret, best_dev = rational_game.confirmedRegret(mrmp)
			print "\tfull game regret:\t" + str(fg_regret)
			print "\tbest deviation:\t" + str(best_dev) + "\n"
