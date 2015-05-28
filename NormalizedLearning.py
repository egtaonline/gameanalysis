#! /usr/bin/env python2.7

import numpy as np
from numpy.random import multinomial
from string import join
import json

# import GaussianProcess but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	from sklearn.gaussian_process import GaussianProcess
	from sklearn.grid_search import GridSearchCV
except ImportError:
	warnings.warn("sklearn.gaussian_process is required for game learning.")

from RoleSymmetricGame import Game
from HashableClasses import h_array
from BasicFunctions import prod, game_size


def _blocked_attribute(*args, **kwds):
	raise TypeError("unsupported operation")


class GP_Game(Game):
	"""
	"""
	addProfile = addProfileArrays = isComplete = _blocked_attribute
	makeLists = makeArrays = allProfiles = knownProfiles = _blocked_attribute
	toJSON = to_TB_JSON = _blocked_attribute

	def __init__(self, json_data, EVs="point"):
		"""
		game:		SampleGame json object.
		CV:			If set to True, cross-validation will be used to select
					parameters of the GPs.
		EVs:		'point': estimate EVs via point_EVs
					'sample': estimate EVs via sample_EVs
		"""
		#set up RSG.Game member variables
		self.roles = sorted(json_data["players"].keys())
		self.players = json_data["players"]
		self.strategies = {r:sorted(json_data["strategies"][r]) \
							for r in self.roles}
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

		#set up GP_Game variables
		assert EVs in {"point", "sample"}
		self.EVs = EVs
		self.payoffMeans = self.zeros(dtype=float)
		self.payoffVars = self.zeros(dtype=float)
		X = {r:{s:[] for s in self.strategies[r]} for r in self.roles}
		Y = {r:{s:[] for s in self.strategies[r]} for r in self.roles}
		S = {r:{s:[] for s in self.strategies[r]} for r in self.roles}

		#extract data in an appropriate form for learning
		for prof in json_data["profiles"]:
			n = len(prof.values()[0][0][2])
			x = self.zeros(dtype=int)
			y = self.zeros(dtype=int)
			s = np.zeros(list(x.shape)+[n], dtype=float)
			for r,role in enumerate(self.roles):
				for strat,count,values in prof[role]:
					s = self.index(role, strat)
					v = np.array(values, dtype=float)
					x[r,s] = count
					Y[role][strat].append(v.mean())
					S[role][strat].append(v)
			for r,role in enumerate(self.roles):
				for s,strat in enumerate(self.strategies[role]):
					if x[r,s] > 0:
						dev = self.array_index(role, strat)
						X[role][strat].append(self.flatten(x - dev))

		#normalize Y so that mean=0 and var=1
		#convert S to normalized variances
		for r,role in enumerate(self.roles):
			for s,strat in enumerate(self.strategies[role]):
				y = np.array(Y[role][strat])
				self.payoffMeans[r,s] = m = y.mean()
				self.payoffVars[r,s] = v = y.var()
				Y[role][strat] = (y - m) / v
				var = np.array([np.var((s - m) / v) for s in S[role][strat]])
				S[role][strat] = var / Y[role][strat]**2

		#learn the GPs
		self.GPs = {r:{} for r in self.roles}
		for role in self.roles:
			for strat in self.strategies[role]:
				#normalize variance to get nugget, but don't increase it
				self.GPs[role][strat] = train_GP(X[role][strat],
								Y[role][strat], S[role][strat])


	def flat_index(self, role, strat):
		return sum(self.numStrategies[:self.index(role)]) + \
					self.strategies[role].index(strat)


	def flatten(self, prof):
		"""
		Turns a profile (represented as Profile object or count array) into a
		1-D vector of strategy counts.
		"""
		vec = []
		for r in range(len(self.roles)):
			vec.extend(prof[r][:self.numStrategies[r]])
		return np.array(vec)


	def getPayoff(self, profile, role, strategy):
		profile = self.flatten(self.toArray(profile))
		return self.predict(role, None, profile) - \
				self.predict(role, strategy, profile)


	def getSocialWelfare(self, profile):
		return self.expectedValues(self.toArray(profile)).sum()


	def GP_estimates(self, mix):
		if self.EVs == "point":
			return self.point_EVs(mix)
		elif EVs == "sample":
			return self.sample_EVs(mix, samples)


	def expectedValues(self, mix):
		return self.GP_estimates(mix)*self.payoffVars + self.payoffMeans


	def __repr__(self):
		return (str(self.__class__.__name__) + ":\n\troles: " + \
				join(self.roles, ",") + "\n\tplayers:\n\t\t" + \
				join(map(lambda x: str(x[1]) + "x " + str(x[0]), \
				sorted(self.players.items())), "\n\t\t") + \
				"\n\tstrategies:\n\t\t" + join(map(lambda x: x[0] + \
				":\n\t\t\t" + join(x[1], "\n\t\t\t"), \
				sorted(self.strategies.items())), "\n\t\t")).expandtabs(4)

	def __cmp__(self, other):
		return cmp(type(self), type(other)) or\
				cmp(self.roles, other.roles) or \
				cmp(self.players, other.players) or \
				cmp(self.strategies, other.strategies) or \
				cmp(self.GPs, other.GPs)


	def sample_EVs(self, mix, samples=1000):
		EVs = self.zeros()
		profiles = np.array(zip(*[multinomial(self.players[role] - 1, mix[r], \
							samples) for r,role in enumerate(self.roles)]))
		deviators = np.array(zip(*[multinomial(1, mix[r], samples) for \
							r,role in enumerate(self.roles)]))
		for r,role in enumerate(self.roles):
			x = profiles + deviators
			x[:,r,:] -= deviators[:,r,:]
			x = map(self.flatten, x)
			for s,strat in enumerate(self.strategies[role]):
				EVs[r,s] = self.predict(role, strat, x, samples).mean()
		return EVs


	def point_EVs(self, mix, *args):
		prof = np.array([mix[r]*self.players[role] for r,role in \
										enumerate(self.roles)])
		dev = np.array([mix[r]*(self.players[role]-1) for r,role in \
										enumerate(self.roles)])
		EVs = self.zeros()
		for r,role in enumerate(self.roles):
			x = np.array(prof)
			x[r,:] = dev[r,:]
			x = self.flatten(x)
			for s,strat in enumerate(self.strategies[role]):
				EVs[r,s] = self.predict(role, strat, x)
		return EVs

	def predict(self, role, strat, x, samples=1):
		"""
		Exists because games learned with sklearn 0.14 are missing y_ndim_
		"""
		try:
			return self.GPs[role][strat].predict(x, samples)
		except AttributeError:
			self.GPs[role][strat].y_ndim_ = 1
			return self.GPs[role][strat].predict(x, samples)


def train_GP(X, Y, nugget):
	params = {
		"storage_mode":"light",
		"thetaL":1e-2,
		"theta0":1,
		"thetaU":1e3,
		"normalize":False,
		"corr":"squared_exponential",
		"nugget":nugget
	}
	gp = GaussianProcess(**params)
	gp.fit(X, Y)
	return gp
