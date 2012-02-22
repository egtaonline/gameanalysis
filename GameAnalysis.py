from numpy.linalg import norm
from itertools import product
from math import isinf

from RoleSymmetricGame import *

def HierarchicalReduction(game, players={}):
	if not players:
		players = {r : game.players[r] / 2 for r in game.roles}
	HR_game = Game(game.roles, players, game.strategies)
	for reduced_profile in HR_game.allProfiles():
		full_profile = Profile({r:FullGameProfile(reduced_profile[r], \
				game.players[r]) for r in game.roles})
		HR_game.addProfile({r:[payoff_data(s, reduced_profile[r][s], \
				game.getPayoff(full_profile, r, s)) for s in full_profile[r]] \
				for r in full_profile})
	return HR_game


def FullGameProfile(HR_profile, N):
	"""
	Returns the symmetric full game profile corresponding to the given
	symmetric reduced game profile.
	"""
	n = sum(HR_profile.values())
	full_profile = {s : c * N / n  for s,c in HR_profile.items()}
	while sum(full_profile.values()) < N:
		full_profile[max([(float(N) / n * HR_profile[s] \
				- full_profile[s], s) for s in full_profile])[1]] += 1
	return full_profile


def DeviationPreservingReduction(game, players={}):
	if not players:
		players = {r:2 for r in game.roles}
	DPR_game = Game(game.roles, players, game.strategies)
	for reduced_profile in DPR_game.allProfiles():
		role_payoffs = {}
		for role in game.roles:
			role_payoffs[role] = []
			for s in reduced_profile[role]:
				full_profile = {}
				for r in game.roles:
					if r == role:
						opp_prof = reduced_profile.asDict()[r]
						opp_prof[s] -= 1
						full_profile[r] = FullGameProfile(opp_prof, \
								game.players[r] - 1)
						full_profile[r][s] += 1
					else:
						full_profile[r] = FullGameProfile(reduced_profile[r], \
								game.players[r])
				role_payoffs[r].append(payoff_data(s, reduced_profile[r][s], \
						game.getPayoff(Profile(full_profile), r, s)))
		DPR_game.addProfile(role_payoffs)
	return DPR_game


def Subgame(game, strategies={}):
	"""
	Creates a game with a subset each role's strategies.

	default settings result in a subgame with no strategies
	"""
	if not strategies:
		strategies = {r:[] for r in game.roles}
	sg = Game(game.roles, game.players, strategies)
	for p in sg.allProfiles():
		if p in game:
			sg.addProfile({r:[payoff_data(s, p[r][s], game.getPayoff(p,r,s)) \
					for s in p[r]] for r in p})
	return sg


def IsSubgame(small_game, big_game):
	if any((r not in big_game.roles for r in small_game.roles)):
		return False
	if any((small_game.players[r] != big_game.players[r] for r \
			in small_game.roles)):
		return False
	for r in small_game.roles:
		if any((s not in big_game.strategies[r] for s in \
				small_game.strategies[r])):
			return False
	return True


def Cliques(full_game, subgames=set()):
	"""
	Finds maximal subgames for which all profiles are known.

	input:
	subgames = known complete subgames to be expanded (any payoff data in
	the known subgames is ignored, so for faster loading, give only the
	header information).
	"""
	subgames = {full_game.Subgame(g.strategies) for g in subgames}.union(\
			{Subgame(full_game)})
	maximal_subgames = set()
	while(subgames):
		sg = subgames.pop()
		maximal = True
		for role in full_game.roles:
			for s in set(full_game.strategies[role]) - \
					set(sg.strategies[role]):
				new_sg = Subgame(full_game, {r:list(sg.strategies[r])\
						+ ([s] if r == role else []) for r in full_game.roles})
				if not new_sg.isComplete():
					continue
				maximal=False
				if new_sg in subgames or new_sg in maximal_subgames:
					continue
				if any([IsSubgame(new_sg, g) for g in \
							subgames.union(maximal_subgames)]):
					continue
				subgames.add(new_sg)
		if maximal and len(sg) > 0:
			maximal_subgames.add(sg)
	return maximal_subgames


def IteratedElimination(game, criterion, *args, **kwargs):
	"""
	iterated elimination of dominated strategies

	input:
	criterion = function to find dominated strategies
	"""
	g = criterion(game, *args, **kwargs)
	if game == g:
		return game
	return IteratedElimination(g, criterion, *args, **kwargs)


def NeverBestResponse(game, conditional=True):
	"""
	never-a-weak-best-response criterion for IEDS

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
	return Subgame(game, best_responses)


def PureStrategyDominance(game, conditional=True, weak=False):
	"""
	pure-strategy dominance criterion for IEDS
	"""
	undominated = {r:set(game.strategies[r]) for r in game.roles}
	for r in game.roles:
		for dominant, dominated in product(game.strategies[r], repeat=2):
			if dominant == dominated or dominated not in undominated[r]:
				continue
			dominance_proved = False
			for profile in game:
				if dominated in profile[r]:
					regret = game.regret(profile, r, dominated, dominant)
					if regret > 0 and not isinf(regret):
						dominance_proved = True
					elif (regret < 0) or (regret == 0 and not weak) or \
							(isinf(regret) and conditional):
						dominance_proved = False
						break
				elif dominant in profile[r] and conditional:
					if profile.deviate(r, dominant, dominated) not in game:
						dominance_proved = False
						break
			if dominance_proved:
				undominated[r].remove(dominated)
	return Subgame(game, undominated)


def MixedStrategyDominance(game, conditional=True, weak=False):
	"""
	mixed-strategy dominance criterion for IEDS
	"""
	raise NotImplementedError("TODO")


def PureNash(game, epsilon=0):
	"""
	Finds all pure-strategy epsilon-Nash equilibria.
	"""
	return filter(lambda profile: game.regret(profile) <= epsilon, game)


def MinRegretProfile(game):
	"""
	Finds the profile with the confirmed lowest regret.
	"""
	return min([(game.regret(profile), profile) for profile in game])[1]


def MixedNash(game, regret_thresh=1e-4, dist_thresh=1e-2, *RD_args, **RD_kwds):
	"""
	Runs replicator dynamics from multiple starting mixtures.
	"""
	equilibria = []
	for m in game.biasedMixtures() + [game.uniformMixture()]:
		eq = ReplicatorDynamics(game, m, *RD_args, **RD_kwds)
		distances = map(lambda e: norm(e-eq,2), equilibria)
		if game.regret(eq) <= regret_thresh and all([d >= dist_thresh \
				for d in distances]):
			equilibria.append(eq)
	return equilibria


def ReplicatorDynamics(game, mix, iters=10000, converge_thresh=1e-8, \
		verbose=False):
	"""
	Replicator dynamics.
	"""
	for i in range(iters):
		old_mix = mix
		mix = (game.expectedValues(mix) - game.minPayoffs) * mix
		mix = mix / mix.sum(1).reshape(mix.shape[0],1)
#		if np.allclose(mix, old_mix, converge_thresh):
		if np.linalg.norm(mix - old_mix) <= converge_thresh:
			break
	if verbose:
		print i+1, "iterations ; mix =", mix, "; regret =", game.regret(mix)
	return mix

