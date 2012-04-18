import numpy as np

from itertools import product, chain, combinations_with_replacement as CwR
from collections import namedtuple
from math import isinf
from string import join

from HashableClasses import *
from BasicFunctions import *

payoff_data = namedtuple("payoff", "strategy count value")

tiny = float(np.finfo(np.float64).tiny)

class Profile(h_dict):
	def __init__(self, role_payoffs):
		try:
			d = {}
			for role, payoffs in role_payoffs.items():
				d[role] = h_dict({p.strategy:p.count for p in payoffs})
			h_dict.__init__(self, d)
		except AttributeError:
			h_dict.__init__(self, {r:h_dict(p) for r,p in role_payoffs.items()})

	def remove(self, role, strategy):
		p = self.asDict()
		p[role][strategy] -= 1
		if p[role][strategy] == 0:
			del p[role][strategy]
		return Profile(p)

	def add(self, role, strategy):
		p = self.asDict()
		p[role][strategy] = p[role].get(strategy, 0) + 1
		return Profile(p)

	def deviate(self, role, strategy, deviation):
		p = self.asDict()
		p[role][strategy] -= 1
		if p[role][strategy] == 0:
			del p[role][strategy]
		p[role][deviation] = p[role].get(deviation, 0) + 1
		return Profile(p)

	def asDict(self):
		return {r:{s:self[r][s] for s in self[r]} for r in self}

	def __repr__(self):
		return join([role +": "+ join([str(count) +" "+ strategy for strategy, \
			count in self[role].items()], ", ") for role in self], "; ")


class Game(dict):
	def __init__(self, roles=[], players={}, strategies={}, payoff_data=[]):
		"""
		Role-symmetric game representation.

		__init__ parameters:
		roles: collection of role-name strings
		players: mapping from roles to number of players per role
		strategies: mapping from roles to per-role strategy sets
		payoff_data: collection of data objects mapping roles to collections
				of (strategy, count, value) tuples
		"""
		self.roles = sorted(set(map(str, roles)))
		self.players = h_dict({r : int(players[r]) for r in self.roles})
		self.strategies = h_dict({r : tuple(sorted(set(map(str, \
				strategies[r])))) for r in self.roles})

		self.numStrategies = [len(self.strategies[r]) for r in self.roles]
		self.maxStrategies = max(self.numStrategies)
		self.minPayoffs = self.zeros(dtype=float, masked=False)
		self.minPayoffs.fill(float('inf'))

		self.mask = np.array([[False]*s + [True]*(self.maxStrategies - s) for \
				s in self.numStrategies])
		self.size = prod([game_size(self.players[r], self.numStrategies[ \
				i]) for i,r in enumerate(self.roles)])
		self.role_index = {r:i for i,r in enumerate(self.roles)}
		self.strategy_index = {r : {s:i for i,s in enumerate( \
				self.strategies[r]) } for r in self.roles}
		self.values = []
		self.counts = []
		self.dev_reps = []

		for profile_data_set in payoff_data:
			self.addProfile(profile_data_set)

	def addProfile(self, role_payoffs):
		if isinstance(self.values, np.ndarray):
			self.values = list(self.values)
			self.counts = list(self.counts)
			self.dev_reps = list(self.dev_reps)
		counts = self.zeros(dtype=int)
		values = self.zeros(dtype=float)
		for role_index, role in enumerate(self.roles):
			for strategy, count, value in role_payoffs[role]:
				if value < self.minPayoffs[role_index][0]:
					self.minPayoffs[role_index] = value
				strategy_index = self.index(role, strategy)
				counts[role_index, strategy_index] = count
				values[role_index, strategy_index] = value
		devs = self.zeros(dtype=int)
		for i, r in enumerate(self.roles):
			for j, s in enumerate(self.strategies[r]):
				if counts[i,j] > 0:
					opp_prof = counts - self.array_index(r,s)
					try:
						devs[i,j] = profile_repetitions(opp_prof)
					except OverflowError:
						devs = np.array(devs, dtype=object)
						devs[i,j] = profile_repetitions(opp_prof)
				else:
					devs[i,j] = 0
		self[Profile(role_payoffs)] = len(self.values)
		self.values.append(values)
		self.counts.append(counts)
		self.dev_reps.append(devs)

	def __hash__(self):
		return hash((self.players, self.strategies))

	def index(self, role, strategy=None):
		"""
		index(r) returns the role-index of r
		index(r,s) returns the strategy index of s (for role r)
		"""
		if strategy != None:
			return self.strategy_index[role][strategy]
		return self.role_index[role]

	def zeros(self, dtype=float, masked=False):
		z = np.zeros([len(self.roles), self.maxStrategies], dtype=dtype)
		return np.ma.array(z, mask=self.mask) if masked else z

	def array_index(self, role, strategy=None, dtype=bool):
		"""
		array_index(r,s) returns a boolean ndarray version of index(r,s)
		"""
		a = self.zeros(dtype=dtype)
		if strategy == None:
			a[self.index(role)] += 1
		else:
			a[self.index(role), self.index(role, strategy)] += 1
		return a

	def getPayoff(self, profile, role, strategy):
		v = self.values[self[profile]]
		return v[self.index(role), self.index(role,strategy)]

	def getExpectedPayoff(self, mix, role=None):
		if role == None:
			return (mix * self.expectedValues(mix)).sum(1)
		return (mix * self.expectedValues(mix)).sum(1)[self.index(role)]

	def expectedValues(self, mix):
		"""
		Computes the expected value of each pure strategy played against
		all opponents playing mix.

		The result is normalized by the sum of all profile weights to cope
		with missing profiles.
		"""
		if isinstance(self.values, list):
			self.values = np.array(self.values)
			self.counts = np.array(self.counts)
			self.dev_reps = np.array(self.dev_reps)
		try:
			weights = ((mix+tiny)**self.counts).prod(1).prod(1).reshape( \
					self.values.shape[0], 1, 1) * self.dev_reps / (mix+tiny)
		except ValueError: #this happens if there's only one strategy
			weights = ((mix+tiny)**self.counts).prod(1).reshape( \
					self.values.shape[0], 1) * self.dev_reps / (mix+tiny)
		return (self.values * weights).sum(0) #/ (weights.sum(0) + tiny)

	def allProfiles(self):
		return [Profile({r:{s:p[self.index(r)].count(s) for s in set(p[ \
				self.index(r)])} for r in self.roles}) for p in product(*[ \
				CwR(self.strategies[r], self.players[r]) for r in self.roles])]

	def knownProfiles(self):
		return self.keys()

	def uniformMixture(self):
		return np.array(1-self.mask, dtype=float) / \
				(1-self.mask).sum(1).reshape(len(self.roles),1)

	def randomMixture(self):
		m = np.random.uniform(0, 1, size=self.mask.shape)
		return m / m.sum(1).reshape(m.shape[0], 1)

	def biasedMixtures(self, role=None, strategy=None, bias=.9):
		"""
		Gives mixtures where the input strategy has %bias weight for its role.

		Probability for that role's remaining strategies is distributed
		uniformly, as is probability for all strategies of other roles.

		Returns a list even when a single role & strategy are specified, since
		the main use case is starting replicator dynamics from several mixtures.
		"""
		assert 0 <= bias <= 1, "probabilities must be between zero and one"
		if self.maxStrategies == 1:
			return [self.uniformMixture()]
		if role == None:
			return list(chain(*[self.biasedMixtures(r, strategy, bias) for r \
					in filter(lambda r: self.numStrategies[self.index(r)] \
					> 1, self.roles)]))
		if strategy == None:
			return list(chain(*[self.biasedMixtures(role, s, bias) for s in \
					self.strategies[role]]))
		i = self.array_index(role, strategy, dtype=float)
		m = 1. - self.mask - i
		m /= m.sum(1).reshape(m.shape[0], 1)
		m[self.index(role)] *= (1. - bias)
		m += i*bias
		return [m]

	def __cmp__(self, other):
		"""does not compare payoffs"""
		return cmp(self.roles, other.roles) or \
				cmp(self.players, other.players) or \
				cmp(self.strategies, other.strategies) or \
				cmp(sorted(self.keys()), sorted(other.keys()))

	def __eq__(self, other):
		return not self.__cmp__(other)

	def __ne__(self, other):
		return self.__cmp__(other)

	def __lt__(self, other):
		return min(self.__cmp__(other), 0)

	def __gt__(self, other):
		return max(self.__cmp__(other), 0)

	def __le__(self, other):
		return self == other or self < other

	def __ge__(self, other):
		return self == other or self > other

	def __repr__(self):
		return ("RoleSymmetricGame:\n\troles: " + join(self.roles, ",") + \
				"\n\tplayers:\n\t\t" + join(map(lambda x: str(x[1]) +"x "+\
				str(x[0]), sorted(self.players.items())), "\n\t\t") + \
				"\n\tstrategies:\n\t\t" + join(map(lambda x: x[0] + \
				":\n\t\t\t" + join(x[1], "\n\t\t\t"), \
				sorted(self.strategies.items())), "\n\t\t") + \
				"\npayoff data for " + str(len(self)) + " out of " + \
				str(self.size) + " profiles").expandtabs(4)

	def isComplete(self):
		return len(self) == self.size

	def regret(self, p, role=None, strategy=None, deviation=None, bound=False):
		if role == None:
			return max([self.regret(p, r, strategy, deviation, bound) for r \
					in self.roles])
		if strategy == None and isinstance(p, Profile):
			return max([self.regret(p, role, s, deviation, bound) for s \
					in p[role]])
		if deviation == None:
			return max([self.regret(p, role, strategy, d, bound) for d \
					in self.strategies[role]])
		if isinstance(p, Profile):
			dp = p.deviate(role, strategy, deviation)
			if dp in self:
				return self.getPayoff(dp, role, deviation) - \
						self.getPayoff(p, role, strategy)
			else:
				return -float("inf") if bound else float("inf")
		elif isinstance(p, np.ndarray):
			if any(map(lambda prof: prof not in self, self.mixtureNeighbors( \
					p, role, deviation))):
				return -float("inf") if bound else float("inf")
			return self.expectedValues(p)[self.index(role), self.index( \
					role, deviation)] - self.getExpectedPayoff(p, role)
		raise TypeError("unrecognized argument type: " + type(p).__name__)

	def neighbors(self, p, *args, **kwargs):
		if isinstance(p, Profile):
			return self.profileNeighbors(p, *args, **kwargs)
		elif isinstance(p, np.ndarray):
			return self.mixtureNeighbors(p, *args, **kwargs)
		raise TypeError("unrecognized argument type: " + type(p).__name__)

	def profileNeighbors(self, profile, role=None, strategy=None, \
			deviation=None):
		if role == None:
			return list(chain(*[self.profileNeighbors(profile, r, strategy, \
					deviation) for r in self.roles]))
		if strategy == None:
			return list(chain(*[self.profileNeighbors(profile, role, s, \
					deviation) for s in profile[role]]))
		if deviation == None:
			return list(chain(*[self.profileNeighbors(profile, role, strategy, \
					d) for d in set(self.strategies[role]) - {strategy}]))
		return [profile.deviate(role, strategy, deviation)]

	def mixtureNeighbors(self, mix, role=None, deviation=None):
		n = set()
		for profile in self.feasibleProfiles(mix):
			n.update(self.profileNeighbors(profile, role, deviation=deviation))
		return n

	def feasibleProfiles(self, mix, thresh=1e-3):
		return [Profile({r:{s:p[self.index(r)].count(s) for s in set(p[ \
				self.index(r)])} for r in self.roles}) for p in product(*[ \
				CwR(filter(lambda s: mix[self.index(r), self.index(r,s)] >= \
				thresh, self.strategies[r]), self.players[r]) for r \
				in self.roles])]

	def bestResponses(self, p, role=None, strategy=None):
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
			return {r: self.bestResponses(p, r, strategy) for r \
					in self.roles}
		if strategy == None and isinstance(p, Profile):
			return {s: self.bestResponses(p, role, s) for s in \
					p[role]}
		best_deviations = set()
		biggest_gain = float('-inf')
		unknown = set()
		for dev in self.strategies[role]:
			r = self.regret(p, role, strategy, dev)
			if isinf(r):
				unknown.add(dev)
			elif r > biggest_gain:
				best_deviations = {dev}
				biggest_gain = r
			elif r == biggest_gain:
				best_deviations.add(dev)
		return list(best_deviations), list(unknown)

	def translate(self, other, array):
		"""
		Translates a mixture, profile, count, or payoff array from a related
		game based on role/strategy indices.

		Useful for testing full-game regret of subgame equilibria.
		"""
		a = self.zeros()
		for role in self.roles:
			for strategy in other.strategies[role]:
				a[self.index(role), self.index(role, strategy)] = array[ \
						other.index(role), other.index(role, strategy)]
		return a
