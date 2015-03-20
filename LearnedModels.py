#! /usr/bin/env python2.7

from numpy import array
from numpy.random import multinomial, normal
from string import join

# import GaussianProcess but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	from sklearn.gaussian_process import GaussianProcess
	from sklearn.grid_search import GridSearchCV
except ImportError:
	warnings.warn("sklearn.gaussian_process is required for game learning.")

from dpr import full_prof_DPR
from RoleSymmetricGame import Game, PayoffData
from HashableClasses import h_dict


class ZeroPredictor:
	def predict(self, *args, **kwds):
		return 0


def _blocked_attribute(*args, **kwds):
	raise TypeError("unsupported operation")


class GP_Game(Game):
	"""
	"""
	addProfile = addProfileArrays = isComplete = _blocked_attribute
	makeLists = makeArrays = allProfiles = knownProfiles = _blocked_attribute
	toJSON = to_TB_JSON = _blocked_attribute

	def __init__(self, sample_game, CV=False, diffs=None, EVs="point", \
					DPR_size={}):
		"""
		game:		RoleSymmetricGame.SampleGame object with enough data to
					estimate payoff functions.
		CV:			If set to True, cross-validation will be used to select
					parameters of the GPs.
		diffs:		None: learn payoffs directly
					'strat': learn differences from average strategy payoff
					'player': learn differences from average player payoff
		EVs:		'point': estimate EVs via point_EVs
					'sample': estimate EVs via sample_EVs
					'DPR': estimate EVs via DPR_EVs
		DPR_size:	default number of players to use for DPR_EVs
		"""
		assert diffs in {None, "strat", "player"}
		assert EVs in {"point", "sample", "DPR"}
		self.CV = CV
		self.diffs = diffs
		self.EVs = EVs
		self.DPR = {}
		for attr in sample_game.__dict__.keys():
			if attr not in {"dev_reps", "values", "counts", "sample_values"}:
				self.__dict__[attr] = sample_game.__dict__[attr]
		if DPR_size != {}:
			self.DPR_size = DPR_size
		else:
			self.DPR_size = {r:min(3, self.players) for r in self.roles}
		self.learn(sample_game.counts, sample_game.sample_values)


	def learn(self, counts, sample_values):
		X_profiles = []
		X_samples = {r:{s:[] for s in self.strategies[r]} for r in self.roles}
		Y_mean = {r:[] for r in self.roles}
		Y_diff = {r:{s:[] for s in self.strategies[r]} for r in self.roles}

		#extract data in an appropriate form for learning
		for p in range(len(counts)):
			prof = counts[p]
			samples = sample_values[p]
			x = self.flatten(prof)
			X_profiles.append(x)
			if self.diffs == None:
				ym = [0]*len(self.roles)
			elif self.diffs == "strat":
				ym = samples.mean(2).sum(1) / (prof > 0).sum(1)
			elif self.diffs == "player":
				ym = (samples.mean(2) * prof).sum(1) / prof.sum(1)
			for r,role in enumerate(self.roles):
				Y_mean[role].append(ym[r])
				for s,strat in enumerate(self.strategies[role]):
					if prof[r][s] > 0:
						y = ym[r] - samples[r,s]
						Y_diff[role][strat].extend(y)
						fi = self.flat_index(role, strat)
						for i in range(len(y)):
							x_s = x + normal(0, 1e-9, len(x))
							x_s[fi] -= 1
							X_samples[role][strat].append(x_s)
		#learn the GPs
		self.GPs = {r:{} for r in self.roles}
		for role in self.roles:
			if self.diffs:
				self.GPs[role][None] = train_GP(X_profiles, Y_mean)
			else:
				self.GPs[role][None] = ZeroPredictor()
			for strat in self.strategies[role]:
				self.GPs[role][strat] = train_GP(X_samples[role][strat], \
												Y_diff[role][strat], self.CV)


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
		return array(vec)


	def getPayoff(self, profile, role, strategy):
		profile = self.flatten(self.toArray(profile))
		return self.GPs[role][None].predict(profile) - \
				self.GPs[role][strategy].predict(profile)


	def getSocialWelfare(self, profile):
		return self.expectedValues(self.toArray(profile)).sum()


	def expectedValues(self, mix, EVs=None, DPR_size=0, samples=1000):
		if EVs == None:
			EVs = self.EVs
		if EVs == "point":
			return self.point_EVs(mix)
		elif EVs == "sample":
			return self.sample_EVs(mix, samples)
		elif EVs == "DPR":
			return self.DPR_EVs(mix, DPR_size)


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


	def DPR_EVs(self, mix, players=0):
		if not players:
			players = self.DPR_size
		if len(self.roles) == 1 and isinstance(players, int):
			players = {self.roles[0]:players}
		elif isinstance(players, list):
			players = dict(zip(self.roles, players))
		if not isinstance(players, h_dict):
			players = h_dict(players)
		if players not in self.DPR:
			self.fill_DPR(players)
		return self.DPR[players].expectedValues(mix)


	def fill_DPR(self, players):
		learned_game = Game(self.roles, players, self.strategies)
		for prof in learned_game.allProfiles():
			role_payoffs = {}
			for role in self.roles:
				role_payoffs[role] = []
				for strat,count in prof[role].iteritems():
					full_prof = full_prof_DPR(prof, role, strat, self.players)
					x = self.flatten(self.toArray(full_prof))
					x[self.flat_index(role, strat)] -= 1
					y_mean = self.GPs[role][None].predict(x)
					y_diff = self.GPs[role][strat].predict(x)
					y = y_mean - y_diff
					role_payoffs[role].append(PayoffData(strat, count, y))
			learned_game.addProfile(role_payoffs)
		self.DPR[players] = learned_game


	def sample_EVs(self, mix, samples=1000):
		EVs = self.zeros()
		profiles = array(zip(*[multinomial(self.players[role] - 1, mix[r], \
							samples) for r,role in enumerate(self.roles)]))
		deviators = array(zip(*[multinomial(1, mix[r], samples) for \
							r,role in enumerate(self.roles)]))
		for r,role in enumerate(self.roles):
			x = profiles + deviators
			x[:,r,:] -= deviators[:,r,:]
			x = map(self.flatten, x)
			for s,strat in enumerate(self.strategies[role]):
				EVs[r,s] = (self.GPs[role][None].predict(x) - \
							self.GPs[role][strat].predict(x)).mean()
		return EVs


	def point_EVs(self, mix, *args):
		prof = array([mix[r]*self.players[role] for r,role in \
										enumerate(self.roles)])
		dev = array([mix[r]*(self.players[role]-1) for r,role in \
										enumerate(self.roles)])
		EVs = self.zeros()
		for r,role in enumerate(self.roles):
			x = array(prof)
			x[r,:] = dev[r,:]
			x = self.flatten(x)
			y_mean = self.GPs[role][None].predict(x)
			for s,strat in enumerate(self.strategies[role]):
				EVs[r,s] = y_mean - self.GPs[role][strat].predict(x)
		return EVs


constant_params = {
	"storage_mode":"light",
	"thetaL":1e-4,
	"thetaU":1e9,
	"normalize":True
}
CV_params = {
	"corr":["absolute_exponential","squared_exponential","cubic","linear"],
	"nugget":[1e-10,1e-6,1e-4,1e-2,1e0,1e2,1e4]
}
default_params = {
	"corr":"cubic",
	"nugget":1
}


def train_GP(X, Y, cross_validate=False):
	if cross_validate:
		gp = GaussianProcess(**constant_params)
		cv = GridSearchCV(gp, CV_params)
		cv.fit(X, Y)
		params = cv.best_estimator_.get_params()
	else:
		params = dict(constant_params, **default_params)
	gp = GaussianProcess(**params)
	gp.fit(X, Y)
	return gp


