from numpy.linalg import norm
from itertools import product
from math import isinf

from RoleSymmetricGame import *

def HierarchicalReduction(game, players={} ):
	if not players:
		players = {r : game.players[r] / 2 for r in game.roles}
	HR_game = Game(game.roles, players, game.strategies)
	for reduced_profile in HR_game.allProfiles():
		try:
			full_profile = Profile({r:FullGameProfile(reduced_profile[r], \
					game.players[r]) for r in game.roles})
			HR_game.addProfile({r:[payoff_data(s, reduced_profile[r][s], \
					game.getPayoff(full_profile, r, s)) for s in \
					full_profile[r]] for r in full_profile})
		except KeyError:
			continue
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
		try:
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
							full_profile[r] = FullGameProfile(reduced_profile[\
									r], game.players[r])
					role_payoffs[r].append(payoff_data(s, reduced_profile[r\
							][s], game.getPayoff(Profile(full_profile), r, s)))
			DPR_game.addProfile(role_payoffs)
		except KeyError:
			continue
	return DPR_game


def translate(arr, source_game, target_game):
	"""
	Translates a mixture, profile, count, or payoff array between related
	games based on role/strategy indices.

	Useful for testing full-game regret of subgame equilibria.
	"""
	a = target_game.zeros()
	for role in target_game.roles:
		for strategy in source_game.strategies[role]:
			a[target_game.index(role), target_game.index(role, strategy)] = \
					arr[source_game.index(role), source_game.index(role, \
					strategy)]
	return a


def Subgame(game, strategies={}):
	"""
	Creates a game with a subset each role's strategies.

	default settings result in a subgame with no strategies
	"""
	if not strategies:
		strategies = {r:[] for r in game.roles}
	sg = Game(game.roles, game.players, strategies)
	if sg.size <= len(game):
		for p in sg.allProfiles():
			if p in game:
				sg.addProfile({r:[payoff_data(s, p[r][s], \
						game.getPayoff(p,r,s)) for s in p[r]] for r in p})
	else:
		for p in game:
			if all([all([s in sg.strategies[r] for s in p[r]]) for r in p]):
				sg.addProfile({r:[payoff_data(s, p[r][s], \
						game.getPayoff(p,r,s)) for s in p[r]] for r in p})
	return sg


def SubgameAvailable(game, strategies = {}):
	sg = Game(game.roles, game.players, strategies)
	for p in sg.allProfiles():
		if p not in game:
			return False
	return True


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
	subgames = {Game(full_game.roles, full_game.players, g.strategies) for g \
			in subgames}.union({Subgame(full_game)})
	maximal_subgames = set()
	while(subgames):
		sg = subgames.pop()
		maximal = True
		for role in full_game.roles:
			for s in set(full_game.strategies[role]) - \
					set(sg.strategies[role]):
				strategies = {r:list(sg.strategies[r]) + ([s] if r == role \
						else []) for r in full_game.roles}
				if not SubgameAvailable(full_game, strategies):
					continue
				new_sg = Game(sg.roles, sg.players, strategies)
				maximal=False
				if new_sg in subgames or new_sg in maximal_subgames:
					continue
				if any([IsSubgame(new_sg, g) for g in \
							subgames.union(maximal_subgames)]):
					continue
				subgames.add(new_sg)
		if maximal:
			sg = Subgame(full_game, sg.strategies)
			if len(sg) > 0:
				maximal_subgames.add(sg)
	return sorted(maximal_subgames, key=len)


def regret(game, p, role=None, strategy=None, deviation=None, bound=False):
	if role == None:
		return max([game.regret(p, r, strategy, deviation, bound) for r \
				in game.roles])
	if strategy == None and isinstance(p, Profile):
		return max([game.regret(p, role, s, deviation, bound) for s \
				in p[role]])
	if deviation == None:
		return max([game.regret(p, role, strategy, d, bound) for d \
				in game.strategies[role]])
	if isinstance(p, Profile):
		dp = p.deviate(role, strategy, deviation)
		if dp in game:
			return game.getPayoff(dp, role, deviation) - \
					game.getPayoff(p, role, strategy)
		else:
			return -float("inf") if bound else float("inf")
	elif isinstance(p, np.ndarray):
		if any(map(lambda prof: prof not in game, game.mixtureNeighbors( \
				p, role, deviation))):
			return -float("inf") if bound else float("inf")
		return game.expectedValues(p)[game.index(role), game.index( \
				role, deviation)] - game.getExpectedPayoff(p, role)
	raise TypeError("unrecognized argument type: " + type(p).__name__)


def neighbors(game, p, *args, **kwargs):
	if isinstance(p, Profile):
		return game.profileNeighbors(p, *args, **kwargs)
	elif isinstance(p, np.ndarray):
		return game.mixtureNeighbors(p, *args, **kwargs)
	raise TypeError("unrecognized argument type: " + type(p).__name__)


def profileNeighbors(game, profile, role=None, strategy=None, \
		deviation=None):
	if role == None:
		return list(chain(*[game.profileNeighbors(profile, r, strategy, \
				deviation) for r in game.roles]))
	if strategy == None:
		return list(chain(*[game.profileNeighbors(profile, role, s, \
				deviation) for s in profile[role]]))
	if deviation == None:
		return list(chain(*[game.profileNeighbors(profile, role, strategy, \
				d) for d in set(game.strategies[role]) - {strategy}]))
	return [profile.deviate(role, strategy, deviation)]


def mixtureNeighbors(game, mix, role=None, deviation=None):
	n = set()
	for profile in game.feasibleProfiles(mix):
		n.update(game.profileNeighbors(profile, role, deviation=deviation))
	return n


def feasibleProfiles(game, mix, thresh=1e-3):
	return [Profile({r:{s:p[game.index(r)].count(s) for s in set(p[ \
			game.index(r)])} for r in game.roles}) for p in product(*[ \
			CwR(filter(lambda s: mix[game.index(r), game.index(r,s)] >= \
			thresh, game.strategies[r]), game.players[r]) for r \
			in game.roles])]


def bestResponses(game, p, role=None, strategy=None):
	"""
	If role is unspecified, bestResponses returns a dict mapping each role
	all of its strategy-level results. If strategy is unspecified,
	bestResponses returns a dict mapping strategies to the set of best
	responses to the opponent-profile without that strategy.

	If conditional=True, bestResponses returns two sets: the known best
	responses, and the deviations whose value is unkown; otherwise it
	returns only the known best response set.
	"""
	if role == None:
		return {r: game.bestResponses(p, r, strategy) for r \
				in game.roles}
	if strategy == None and isinstance(p, Profile):
		return {s: game.bestResponses(p, role, s) for s in \
				p[role]}
	best_deviations = set()
	biggest_gain = float('-inf')
	unknown = set()
	for dev in game.strategies[role]:
		r = game.regret(p, role, strategy, dev)
		if isinf(r):
			unknown.add(dev)
		elif r > biggest_gain:
			best_deviations = {dev}
			biggest_gain = r
		elif r == biggest_gain:
			best_deviations.add(dev)
	return list(best_deviations), list(unknown)


def IteratedElimination(game, criterion, *args, **kwargs):
	"""
	iterated elimination of dominated strategies

	input:
	criterion = function to find dominated strategies
	"""
	reduced_game = criterion(game, *args, **kwargs)
	while game != reduced_game:
		game = reduced_game
		reduced_game = criterion(game, *args, **kwargs)
	return game


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

	conditional==0==False --> unconditional dominance
	conditional==1==True ---> conditional dominance
	conditional==2 ---------> extra-conservative conditional dominance
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
				elif dominant in profile[r] and conditional > 1:
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
		mix = (game.expectedValues(mix) - game.minPayoffs + tiny) * mix
		mix = mix / mix.sum(1).reshape(mix.shape[0],1)
		if np.linalg.norm(mix - old_mix) <= converge_thresh:
			break
	if verbose:
		print i+1, "iterations ; mix =", mix, "; regret =", game.regret(mix)
	return mix


def SymmetricProfileRegrets(game):
	assert len(game.roles) == 1, "game must be symmetric"
	role = game.roles[0]
	return {s: game.regret(Profile({role:{s:game.players[role]}})) for s \
			in game.strategies[role]}


def EquilibriumRegrets(game, eq):
	regrets = {}
	for role in game.roles:
		regrets[role] = {}
		for strategy in game.strategies[role]:
			regrets[role][strategy] = -game.regret(eq, deviation=strategy)
	return regrets

