#!/usr/local/bin/python2.7

import numpy as np

from math import factorial, isnan
from operator import mul, add
from random import choice
from itertools import combinations_with_replacement as CwR, permutations, \
		product

def nCr(n, k):
	"""
	Number of combinations: n choose k.
	"""
	return reduce(mul, range(n-k+1,n+1)) / factorial(k)

def game_size(n,s):
	"""
	Number of profiles in a symmetric game with n players and s strategies.
	"""
	return nCr(n+s-1,n)

def list_repr(l, sep=", "):
	try:
		return reduce(lambda x,y: str(x) + sep + str(y), l)
	except TypeError:
		return ""

class SymmetricProfile(tuple):
	"""
	Profile object for symmetric games: a sorted tuple of strategies.
	"""
	def __new__(self, strategies):
		return tuple.__new__(self, sorted(strategies))

	def getStrategies(self):
		"""without repetitions"""
		return sorted(set(self))

	def remove(self, strategy):
		"""n-1 player profile excluding strategy"""
		i = self.index(strategy)
		return SymmetricProfile(self[:i] + self[i+1:])

	def add(self, strategy):
		"""n+1 player profile including strategy"""
		return SymmetricProfile(list(self) + [strategy])

	def probability(self, profile):
		"""only makes sense for mixed strategy profiles"""
		try:
			return reduce(add, [reduce(mul, [self[i][s] for i,s in \
					enumerate(p)]) for p in set(permutations(profile))])
		except TypeError:
			return 1

	def isMixed(self):
		return any([isinstance(s, mixture) for s in self])

	def isFullyMixed(self):
		return all([isinstance(s, mixture) for s in self])

	def __repr__(self):
		return "(" + list_repr((repr(self.count(s)) + "x" + \
				repr(s) for s in sorted(set(self)))) + ")"

	def repetitions(self):
		"""number of orderings of the profile that could occur"""
		try:
			return self._reps
		except:
			self._reps = factorial(len(self)) / reduce(mul, (factorial( \
					self.count(s)) for s in sorted(set(self))))
			return self._reps


class Profile(dict):
	"""
	Profile object for RoleSymmetricGames: maps roles to SymmetricProfiles.
	"""
	def _blocked_attribute(obj):
		raise TypeError("profiles are immutable.")
	_blocked_attribute = property(_blocked_attribute)

	__delitem__ = clear = pop = popitem = setdefault = _blocked_attribute

	def __new__(self, *args, **kwargs):
		new = dict.__new__(self, *args, **kwargs)
		self.__setitem__ = self.update = self._blocked_attribute
		return new

	def remove(self, role, strategy):
		return Profile({r:self[r] if r != role else self[r].remove(strategy) \
				for r in self})

	def add(self, role, strategy):
		return Profile({r:self[r] if r != role else self[r].add(strategy) \
				for r in self})

	def probability(self, profile):
		"""only makes sense for mixed strategy profiles"""
		return reduce(mul, [sp.probability(profile[r]) for r, sp \
				in self.items()])

	def isMixed(self):
		return any([p.isMixed() for p in self.values()])

	def isFullyMixed(self):
		return all([p.isFullyMixed() for p in self.values()])

	def __hash__(self):
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(tuple(sorted(self.items())))
			return self._hash

	def repetitions(self):
		"""Combinations of orderings of role profiles that could occur."""
		return reduce(mul, [p.repetitions() for p in self.values()])


class mixture(np.ndarray):
	"""
	Symmetric mixed strategy.

	mixture(data) sets negative values to 0, then normalizes the sum to 1. ex:
	mixture( range(-2,5) ) --> mixture([ 0. , 0. , 0. , 0.1 , 0.2 , 0.3 , 0.4 ])

	Probabilities are rounded to the nearest 10^-10 so they can be compared.
	"""
	def __new__(cls, strategies, probabilities):
		a = np.ma.masked_less(np.array(probabilities, dtype=float), 0)
		a = np.ma.fix_invalid(a)
		a = np.ma.filled(a, 0)
		a = np.ndarray.__new__(cls, shape=a.shape, buffer=np.round(a/sum(a),10))
		a.strategies = dict(zip(strategies, a))
		return a

	def __getitem__(self, item):
		try:
			return np.ndarray.__getitem__(self, item)
		except ValueError:
			return self.strategies[item]

	def getStrategies(self):
		return list(self.strategies.keys())

	def __repr__(self):
		return repr(self.strategies)

	def __str__(self):
		return repr(self)

	def __eq__(self, other):
		return self.strategies == other.strategies and \
				list(self) == list(other)

	def __lt__(self, other):
		return self.strategies < other.strategies or list(self) < list(other)


class Game(dict):
	def __init__(self, roles=[], counts={}, strategies={}, payoffs={}):
		"""
		Role-symmetric game representation.

		__init__ parameters:
		roles: list of role-name strings
		counts: mapping from roles to number of players per role
		strategies: mapping from roles to per-role strategy sets
		payoffs: mapping from Profile objects to payoff dicts, which map
				roles to strategies to numbers
		"""
		self.roles = tuple(sorted(set(map(str, roles))))
		self.counts = {r : int(counts[r]) for r in self.roles}
		self.strategies = {r : tuple(sorted(set(map(str, strategies[r])))) \
				for r in self.roles}
		self.size = reduce(mul, map(lambda r: game_size(self.counts[r], \
				len(self.strategies[r])), self.roles))
		dict.update(self, payoffs)

	def __setitem__(self, profile, payoffs):
		assert self.isValidProfile(profile)
		dict.__setitem__(self, profile, payoffs)

	def update(self, *args, **kwargs):
		d = {}
		d.update(*args, **kwargs)
		for profile in d.keys():
			assert self.isVlaidProfile(profile)
		dict.update(self, d)

	def isValidProfile(self, profile):
		if profile in self:
			return True
		if not isinstance(profile, Profile):
			return False
		for r, sp in profile.items():
			if r not in self.roles:
				return False
			if not isinstance(sp, SymmetricProfile):
				return False
			if len(sp) != self.counts[r]:
				return False
			if not all((s in self.strategies[r] for s in sp.getStrategies())):
				return False
		return True

	def __eq__(self, other):
		return self.roles==other.roles and self.counts==other.counts and \
				self.strategeis==other.strategies and dict.__eq__(self,other)

	def __repr__(self):
		return "RoleSymmetricGame:\nroles: " + list_repr(self.roles) + \
				"\ncounts: " + str(self.counts) + "\nstrategies: " + \
				str(self.strategies) + "\npayoff data for " + \
				str(len(self)) + " out of " + str(self.size) + " profiles"

	def getPayoff(self, profile, role, strategy):
		#try to look up payoff for pure strategy & profile
		try:
			return self[profile][role][strategy]
		except KeyError as ke:
			if not (isinstance(strategy, mixture) or profile.isMixed()):
				raise ke
		#try to compute expected payoff for mixed strategy & profile
		if not isinstance(strategy, mixture):
			strategy = mixture(self.strategies[role], [1 if s==strategy \
					else 0 for s in self.strategies[role]])
		if not profile.isFullyMixed():
			rsp = {}
			for r in self.roles:
				if profile[r].isFullyMixed():
					rsp[r] = profile[r]
					continue
				sp = []
				for s in profile[r]:
					if isinstance(s, mixture):
						sp.append(s)
					else:
						sp.append(mixture(self.strategies[r], [1 if strat==s \
								else 0 for strat in self.strategies[r]]))
				rsp[r] = SymmetricProfile(sp)
			profile = Profile(rsp)
		#EVs = self.expectedValues(profile, role, strategy)
		raise NotImplementedError("getPayoff does not yet accomodate mixed" + \
				" strategies")


	def pureNash(self, epsilon):
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
		for profile in self:
			try:
				r = self.regret(profile)
			except KeyError:
				continue
			if r < mr:
				mr = r
				mrp = profile
			if r == 0:
				NE.add(profile)
			elif r <= epsilon:
				eNE.add(profile)
		return NE, eNE, mrp, mr

	def regret(self, profile):
		regret = 0
		for role, symProf in profile.items():
			for strategy in symProf.getStrategies():
				payoff = self.getPayoff(profile, role, strategy)
				deviations = self.deviations(profile, role, strategy)
				for dev_strat, dev_prof in deviations:
					r = self.getPayoff(dev_prof, role, dev_strat) - payoff
					if r > regret:
						regret = r
		return regret

	def deviations(self, profile, role, strategy):
		"""
		Returns a list of pairs (s,p), where s is a strategy for %role and p is
		the profile that results from deviating from %strategy to s in %profile.
		"""
		neighbors = []
		for s in set(self.strategies[role]) - {strategy}:
			strategy_list = list(profile[role])
			strategy_list.remove(strategy)
			strategy_list.append(s)
			p = Profile(profile)
			dict.__setitem__(p, role, SymmetricProfile(strategy_list))
			neighbors.append((s,p))
		return neighbors

	def IE_NWBR(self):
		"""
		Iterated elimination of never-a-weak-best-response strategies.
		"""
		best_responses = {r:set() for r in self.roles}
		for profile in self:
			for role in profile:
				for strategy in profile[role].getStrategies():
					best_responses[role].update(self.BR(role, \
							profile.remove(role, strategy)))
		if all([len(best_responses[r]) == len(self.strategies[r]) for \
				r in self.roles]):
			return self
		game = Game(self.roles, self.counts, best_responses, {p:self[p] for \
				p in filter(lambda p: all([all([s in best_responses[r] for \
				s in p[r]]) for r in self.roles]), self)})
		return game.IE_NWBR()

	def BR(self, role, opponent_profile):
		best_responses = []
		best_payoff = float("-inf")
		for strategy in self.strategies[role]:
			try:
				payoff = self.getPayoff(opponent_profile.add(role, strategy), \
						role, strategy)
			except KeyError:
				continue
			if payoff > best_payoff:
				best_responses = [strategy]
				best_payoff = payoff
			elif payoff == best_payoff:
				best_responses.append(strategy)
		return best_responses

	def RD(self, mixedProfile=None, iterations=1000):
		"""
		Replicator dynamics.
		"""
		if not mixedProfile:
			mixedProfile = self.uniformMixedProfile()
		minPayoffs = {r:min(self.payoffList(r)) for r in self.roles}
		for i in range(iterations):
			EVs = self.expectedValues(mixedProfile)
			old_mix = mixedProfile
			mixedProfile = Profile({r:SymmetricProfile([mixture( \
					self.strategies[r], [(EVs[r][s] - minPayoffs[r]) * \
					mixedProfile[r][0][s] for s in self.strategies[r]])] * \
					self.counts[r]) for r in self.roles})
			if old_mix == mixedProfile:
				break
		print i
		return mixedProfile

	def uniformMixedProfile(self):
		return Profile({r : SymmetricProfile([mixture(self.strategies[r], \
				[1]*len(self.strategies[r]))]*self.counts[r]) for \
				r in self.roles})

	def biasedMixedProfile(self, role, strategy):
		return Profile({r : SymmetricProfile([mixture(self.strategies[r], \
				[100 if r==role and s==strategy else 1 for s in \
				self.strategies[r]])] * self.counts[r]) for r in self.roles})

	def payoffList(self, role):
		"""
		Returns all payoff floats associated with a role.

		Useful for determining the minimum achievable payoff.
		"""
		return reduce(add, map(lambda d: list(d[role].values()), self.values()))

	def expectedValues(self, rsmsp):
		"""
		Gives the EV of each pure strategy when played against %rsmsp.

		expectedValues() is preferable to getPayoff() for replicator dynamics,
		because it requires only one loop over the game's payoffs.

		input:
		rsmsp: a role-symmetric (one-per-role) mixed strategy profile

		output:
		EVs: mapping from roles to pure strategies to payoffs
		"""
		values = {r:{s:0.0 for s in self.strategies[r]} for r in self.roles}
		total_prob = {r:{s:0.0 for s in self.strategies[r]} for r in self.roles}
		deviations = {r:rsmsp.remove(r, rsmsp[r][0]) for r in self.roles}
		for profile in self:
			for role in self.roles:
				for strategy in profile[role].getStrategies():
					without = profile.remove(role, strategy)
					prob = deviations[role].probability(without)
					values[role][strategy] += prob * self.getPayoff(profile, \
							role, strategy)
					total_prob[role][strategy] += prob*rsmsp[role][0][strategy]
		return values

	def symmetricProfiles(self, role, smsp):
		"""
		Return pure strategy symmetric profiles for role that have positive
		probability under symmetric mixed strategy profile smsp.
		"""
		return map(SymmetricProfile, set(map(lambda p: tuple(sorted(p)), \
				product(*[filter(lambda s: mixture[s], self.strategies[role]) \
				for mixture in smsp]))))

	def roleSymmetricProfiles(self, rsmsp):
		"""
		Return pure strategy role-symmetric profiles that have positive
		probability under role-symmetric mixed strategy profile rsmsp
		"""
		return [Profile(zip(self.roles, p)) for p in product(*[ \
				self.symmetricProfiles(r, rsmsp[r]) for r \
				in self.roles])]

#	def mixtureRegret(self, mixture):
#		EVs = self.expectedValues(mixture)
#		return max(EVs) - np.dot(EVs, mixture).sum()



from json import load
from xml.dom.minidom import parse
from os.path import exists, splitext
from argparse import ArgumentParser


def readJSON(filename):
	f = open(filename)
	data = load(f)
	f.close()
	counts = {r["name"] : int(r["count"]) for r in data["roles"]}
	strategies = {r["name"] : r["strategy_array"] for r in data["roles"]}
	roles = list(counts.keys())

	payoffs = {}
	for profileDict in data["profiles"]:
		profile = {}
		for role_str in profileDict["proto_string"].split("; "):
			role, strategy_str = role_str.split(": ")
			profile[role] = SymmetricProfile(strategy_str.split(", "))
		payoffs[Profile(profile)] = {r["name"]: {s["name"]:s["payoff"] \
				for s in r["strategies"]} for r in profileDict["roles"]}
	return Game(roles, counts, strategies, payoffs)


def readXML(filename):
	gameNode = parse(filename).getElementsByTagName("nfg")[0]
	if len(gameNode.getElementsByTagName("player")[0]. \
			getElementsByTagName("action")) > 0:
		return parseStrategicXML(gameNode)
	return parseSymmetricXML(gameNode)


def parseStrategicXML(gameNode):
	strategies = {p.getAttribute('id') : map(lambda s: s.getAttribute('id'), \
			p.getElementsByTagName('action')) for p in \
			gameNode.getElementsByTagName('player')}
	roles = list(strategies.keys())
	counts = {r:1 for r in roles}
	payoffs = {}
	for payoffNode in gameNode.getElementsByTagName('payoff'):
		str_prof = {}
		payoff = {r:{} for r in roles}
		for outcomeNode in payoffNode.getElementsByTagName('outcome'):
			role = outcomeNode.getAttribute('player')
			strategy = outcomeNode.getAttribute('action')
			value = float(outcomeNode.getAttribute('value'))
			payoff[role][strategy] = value
			str_prof[role] = SymmetricProfile([strategy])
		payoffs[Profile(str_prof)] = payoff
	return Game(roles, counts, strategies, payoffs)


def parseSymmetricXML(gameNode):
	roles = ["All"]
	counts= {"All" : len(gameNode.getElementsByTagName("player"))}
	strategies = {"All" : [e.getAttribute("id") for e in \
			gameNode.getElementsByTagName("action")]}
	payoffs = {}
	for payoffNode in gameNode.getElementsByTagName("payoff"):
		sym_prof = []
		payoff = {"All":{}}
		for outcomeNode in payoffNode.getElementsByTagName("outcome"):
			action = outcomeNode.getAttribute("action")
			count = int(outcomeNode.getAttribute("count"))
			value = float(outcomeNode.getAttribute("value"))
			sym_prof.extend([action] * count)
			payoff["All"][action] = value
		payoffs[Profile({"All":SymmetricProfile(sym_prof)})] = payoff
	return Game(roles, counts, strategies, payoffs)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("file", type=str, help="Game file to be analyzed. " +\
			"Suported file types: EGAT symmetric XML, EGAT strategic XML, " +\
			"testbed role-symmetric JSON.")
	parser.add_argument("-e", metavar="EPSILON", type=float, default=0, \
			help="Max allowed epsilon for approximate Nash equilibria")
	args = parser.parse_args()
	assert exists(args.file)
	ext = splitext(args.file)[-1]
	if ext == '.xml':
		input_game = readXML(args.file)
	elif ext == '.json':
		input_game = readJSON(args.file)
	else:
		raise IOError("unsupported file type: " + ext)
	print "input game =", input_game, "\n"

	rational_game = input_game.IE_NWBR()
	eliminated = {r:sorted(set(input_game.strategies[r]).difference( \
			rational_game.strategies[r])) for r in filter(lambda role: \
			input_game.strategies[role] != rational_game.strategies[role], \
			input_game.roles)}
	print "strategies removed by IE_NWBR:"
	print (eliminated if eliminated else "none"), "\n"

	NE, eNE, mrp, mr = input_game.pureNash(args.e)
	if NE:
		print len(NE), "exact pure strategy Nash equilibria:\n", \
				list_repr(NE, sep="\n"), "\n"
	if eNE:
		print len(eNE), "approximate pure strategy Nash equilibria", \
				"(0 < epsilon <= " + str(args.e) + "):\n", \
				list_repr(eNE, sep="\n"), "\n"
	if mr != 0:
		print "minimum regret profile:", mrp, "\nregret =", mr, "\n"

#	mixed_prof = input_game.uniformMixedProfile()
#	print mixed_prof
#	s = 0
#	for pure_prof in input_game.roleSymmetricProfiles(mixed_prof):
#		prob = mixed_prof.probability(pure_prof)
#		print pure_prof, prob
#		s += prob
#	print s
	"""
	print "equilibria ... pr" + str(game.strategies) + ":"
	eq = game.RD(game.uniformMixture())
	print eq
	equilibria = set([eq])
	for i in range(game.numStrategies):
		probabilities = [0.01] * game.numStrategies
		probabilities[i] += 1.0-sum(probabilities)
		mixed_strategy = SG.mixture(probabilities)
		equilibrium = game.RD(mixed_strategy)
		if eq not in equilibria:
			equilibria.add(eq)
			print eq
	"""

