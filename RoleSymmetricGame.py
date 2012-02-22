import numpy as np

from itertools import product, combinations_with_replacement as CwR
from collections import namedtuple
from math import isinf

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
		return self.remove(role, strategy).add(role, deviation)

	def asDict(self):
		return {r:{s:self[r][s] for s in self[r]} for r in self}

	def __repr__(self):
		return '{' + OrderedDict.__repr__(self)[9:-2] + '}'


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
		"""
		if isinstance(self.values, list):
			self.values = np.array(self.values)
			self.counts = np.array(self.counts)
			self.dev_reps = np.array(self.dev_reps)
		weights = (mix**self.counts).prod(1).prod(1).reshape( \
				self.values.shape[0], 1, 1) * self.dev_reps / (mix+tiny)
		return (self.values * weights).sum(0)# / (weights.sum(0) + tiny)

	def allProfiles(self):
		return [Profile({r:{s:p[self.index(r)].count(s) for s in set(p[ \
				self.index(r)])} for r in self.roles}) for p in product(*[ \
				CwR(self.strategies[r], self.players[r]) for r in self.roles])]

	def knownProfiles(self):
		return self.keys()

	def uniformMixture(self):
		return np.array(1-self.mask, dtype=float) / \
				(1-self.mask).sum(1).reshape(len(self.roles),1)

	def biasedMixtures(self, role=None, strategy=None):
		"""
		Gives mixtures where the input strategy has 90% weight for its role.

		Probability for that role's remaining strategies is distributed
		uniformly, as is probability for all strategies of other roles.

		Returns a list even when a single role & strategy are specified, since
		the main use case is starting replicator dynamics from several mixtures.
		"""
		if role == None:
			return sum([self.biasedMixtures(r) for r in filter(lambda r: \
					self.numStrategies[self.index(r)] > 1, self.roles)], [])
		if strategy == None:
			return sum([self.biasedMixtures(role, s) for s in \
					self.strategies[role]], [])
		m = (1 - self.mask) + self.array_index(role, strategy, dtype=float) \
				* (np.array(self.numStrategies) * 9 - 10).reshape( \
				len(self.roles),1)
		return [(m.T / m.sum(1)).T]

	def __cmp__(self, other):
		return cmp(self.roles, other.roles) or \
				cmp(self.players, other.players) or \
				cmp(self.strategies, other.strategies) or \
				dict.__cmp__(self, other)

	def __eq__(self, other):
		return not cmp(self, other)

	def __lt__(self, other):
		return min(self.__cmp__(other), 0)

	def __gt__(self, other):
		return max(self.__cmp__(other), 0)

	def __le__(self, other):
		return self == other or self < other

	def __ge__(self, other):
		return self == other or self > other

	def __repr__(self):
		return ("RoleSymmetricGame:\n\troles: " + list_repr(self.roles) + \
				"\n\tplayers:\n\t\t" + list_repr(map(lambda x: str(x[1]) +"x "+\
				str(x[0]), sorted(self.players.items())), sep="\n\t\t") + \
				"\n\tstrategies:\n\t\t" + list_repr(map(lambda x: x[0] + \
				":\n\t\t\t" + list_repr(x[1], sep="\n\t\t\t"), \
				sorted(self.strategies.items())), sep="\n\t\t") + \
				"\npayoff data for " + str(len(self)) + " out of " + \
				str(self.size) + " profiles").expandtabs(4)

	def isComplete(self):
		return len(self) == self.size

	def regret(self, p, *args, **kwargs):
		if isinstance(p, Profile):
			return self.profileRegret(p, *args, **kwargs)
		elif isinstance(p, np.ndarray):
			return self.mixtureRegret(p, *args, **kwargs)
		raise TypeError("unrecognized argument type: " + type(p).__name__)

	def profileRegret(self, profile, role=None, strategy=None, deviation=None):
		if role == None:
			return max([self.profileRegret(profile, r) for r in self.roles])
		if strategy == None:
			return max([self.profileRegret(profile, role, s) for s in \
					profile[role]])
		if deviation == None:
			return max([self.profileRegret(profile, role, strategy, d) for d \
					in set(self.strategies[role]) - {strategy}])
		try:
			return self.getPayoff(profile.deviate(role, strategy, deviation), \
					role, deviation) - self.getPayoff(profile, role, strategy)
		except KeyError:
			return float("inf")

	def mixtureRegret(self, mix, role=None, deviation=None):
		if role == None:
			return float((self.expectedValues(mix).T - \
					self.getExpectedPayoff(mix)).max())
		if deviation == None:
			return float((self.expectedValues(mix)[self.index(role)] - \
					self.getExpectedPayoff(mix, role)).max())
		return self.expectedValues(mix)[self.index(role), self.index(role, \
				deviation)] - self.getExpectedPayoff(mix, role)

	def neighbors(self, profile, role=None, strategy=None):
		if role == None:
			return sum([self.neighbors(profile, r) for r in self.roles], [])
		if strategy == None:
			return sum([self.neighbors(profile, role, s) for s in \
					profile[role]], [])
		return [profile.deviate(role, strategy, dev) for dev in \
				set(self.strategies[role]) - {strategy}]

	def bestResponses(self, profile, role=None, strategy=None):
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
			return {r: self.bestResponses(profile, r) for r in self.roles}
		if strategy == None:
			return {s: self.bestResponses(profile, role, s) for s in \
					profile[role]}
		best_deviations = set()
		biggest_gain = float('-inf')
		unknown = set()
		for dev in self.strategies[role]:
			r = self.profileRegret(profile, role, strategy, dev)
			if isinf(r):
				unknown.add(dev)
			elif r > biggest_gain:
				best_deviations = {dev}
				biggest_gain = r
			elif r == biggest_gain:
				best_deviations.add(dev)
		return best_deviations, unknown

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
