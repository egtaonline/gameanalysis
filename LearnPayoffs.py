#! /usr/bin/env python2.7

from numpy import array
from numpy.random import multinomial, normal

# import GaussianProcess but don't crash if it wasn't loaded
import warnings
warnings.formatwarning = lambda msg, *args: "warning: " + str(msg) + "\n"
try:
	from sklearn.gaussian_process import GaussianProcess
	from sklearn.grid_search import GridSearchCV
except ImportError:
	warnings.warn("sklearn.gaussian_process is required for game learning.")

import RoleSymmetricGame as RSG
from Reductions import full_prof_DPR


class ZeroPredictor:
	def predict(self, *args, **kwds):
		return 0


def _blocked_attribute(*args, **kwds):
	raise TypeError("unsupported operation")


class GP_Game(RSG.game):
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
		EVs:		'point': estimate EVs via GP_point
					'sample': estimate EVs via GP_sample
					'DPR': estimate EVs via GP_DPR
		DPR_size:	default number of players to use for GP_DPR EV estimation
		"""
		assert diffs in {None, "strat", "player"}
		assert EVs in {"point", "sample", "DPR"}
		self.CV = CV
		self.diffs = diffs
		self.EVs = EVs
		if DPR_size != {}:
			self.DPR_size = DPR_size
		else:
			self.DPR_size = {r:3 for r in sample_game.roles}
		for attr in sample_game.__dict__.keys():
			if attr not in {"dev_reps", "values", "counts", "sample_values"}:
				self.__dict__[attr] = sample_game.__dict__[attr]
		self.learn(sample_game.counts, sample_game.sample_values)


	def learn(self, counts, sample_values):
		X_profiles = []
		X_samples = {r:{s:[] for s in self.strategies[r]} for r in self.roles}
		Y_mean = {r:"" for r in self.roles}
		Y_diff = {r:{s:[] for s in self.strategies[r]} for r in self.roles}

		#extract data in an appropriate form for learning
		for p in range(len(counts)):
			prof = counts[p]
			samples = sample_values[p]
			x = self.flatten(prof)
			X_profiles.append(x)
			if diff == 0:
				ym = [0]*len(self.roles)
			elif diff == 1:
				ym = samples.mean(2).sum(1) / (x > 0).sum(1)
			elif diff == 2:
				ym = (samples.mean(2) * x).sum(1) / x.sum(1)
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
		self.GPs = {}
		for role in self.roles:
			self.GPs[r] = {}
			if self.diffs:
				self.GPs[r][None] = train_GP(X_profiles, Y_mean)
			else:
				self.GPs[r][None] = ZeroPredictor()
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


	def getEV(self, mix, role, strategy, EVs=None, DPR_players=0, samples=1000):
		if EVs == None:
			EVs = self.EVs
		if EVs == "point":
			return self.GP_point(mix, role, strategy)
		elif EVs == "sample":
			return self.GP_sample(mix, role, strategy, samples)
		elif EVs == "DPR":
			return self.GP_DPR(mix, role, strategy, DPR_players)


	def getSocialWelfare(self, profile):
		return self.expectedValues(self.toArray(profile)).sum()


	def expectedValues(self, mix, EVs=None, DPR_players=0, samples=1000):
		values = self.zeros()
		for r,role in enumerate(self.roles):
			for s,strat in enumerate(self.strategies[role]):
				values[r][s] = self.getEV(mix,role,strat,DPR_players,samples)
		return values


	def __repr__(self):
		return (str(self.__class__.__name__) + ":\n\troles: " + \
				join(self.roles, ",") + "\n\tplayers:\n\t\t" + \
				join(map(lambda x: str(x[1]) + "x " + str(x[0]), \
				sorted(self.players.items())), "\n\t\t") + \
				"\n\tstrategies:\n\t\t" + join(map(lambda x: x[0] + \
				":\n\t\t\t" + join(x[1], "\n\t\t\t"), \
				sorted(self.strategies.items())), "\n\t\t")).expandtabs(4)

	def __cmp__(self, other):
		return cmp(type(self), type(other) or\
				cmp(self.roles, other.roles) or \
				cmp(self.players, other.players) or \
				cmp(self.strategies, other.strategies) or \
				cmp(self.GPs, other.GPs)


	def GP_DPR(self, mix, players=0):
		if not players:
			players = self.DPR_size
		if len(self.roles) == 1 and isinstance(players, int):
			players = {self.roles[0]:players}
		elif isinstance(players, list):
			players = dict(zip(self.roles, players))
		if not isinstance(players, RSG.Profile):
			players = RSG.Profile(players)
		if players not in self.DPR:
			self.fill_DPR(players)
		return self.DPR[players].expectedValues(mix)


	def fill_DPR(self, players)
		learned_game = RSG.Game(self.roles, players, self.strategies)
		for prof in learned_game.allProfiles():
			role_payoffs = {}
			for role in self.roles:
				role_payoffs[role] = []
				for strat,count in prof[role].iteritems():
					full_prof = full_prof_DPR(prof, role, strat, self.players)
					prof_x = self.flatten(full_prof)
					prof_x[self.flat_index(role, strat)] -= 1
					prof_y = self.GPs[role][strat].predict(prof_x)
					role_payoffs[role].append(RSG.PayoffData(strat, count, \
																prof_y))
			learned_game.addProfile(role_payoffs)
		self.DPR[players] = learned_game


	def GP_sample(self, mix, samples=1000):
		EVs = self.zeros()
		partial_profs = []
		for r,role in enumerate(self.roles):
			partial_profs.append(multinomial(self.players[role],mix[r],samples))
		profiles = [self.flatten(p) for p in zip(*partial_profs)]
		for r,role in enumerate(self.roles):
			for s,strat in enumerate(self.strategies[role]):
				opp_profs = copy(profiles)
				fi = self.flat_index(role, strat)
				for prof in opp_profs:
					prof[fi] -= 1
				EVs[r,s] = (self.GPs[role][None].predict(opp_profs) - \
							self.GPs[role][strat].predict(opp_profs)).mean()
		return EVs


	def GP_point(self, mix, *args):
		prof = [mix[r]*self.players[role] for r,role in enumerate(self.roles)]
		vec = self.flatten(prof)
		EVs = self.zeros()
		for r,role in enumerate(self.roles):
			for s,strat in enumerate(self.strategies[role]):
				EVs[r,s] = self.GPs[role][None].predict(vec) - \
							self.GPs[role][strat].predict(vec)
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
		cv = GridSearchCV(gp, **CV_params)
		cv.fit(X, Y)
		params = cv.best_estimator_.get_params()
	else:
		params = dict(constant_params, **default_params)
	gp = GaussianProcess(**params)
	gp.fit(X, Y)
	return gp


import cPickle
from GameIO import io_parser

def parse_args():
	parser = io_parser()
	parser.add_argument("--EVs", choices=["point","sample","DPR"], default=
						"point", help="default method for game.expectedValues")
	parser.add_argument("--diffs", choices=["None", "strat", "player"],
						default="None", help="set to strat or player to learn"+
						"differences from profile-average payoffs")
	parser.add_argument("--CV", action="store_true", help="cross-validation")
	parser.add_argument("--DPR_size", type=int, default=3, help="")
	args = parser.parse_args()
	if args.diffs == "None":
		args.diffs = None
	return args


def main():
	a = parse_args()
	g = args.input
	g = GP_game(a.input, a.CV, a.diffs, a.EVs, a.DPR_size)
	if a.output != "":
		sys.stdout.close()
		with open(a.output, "w") as f:
			cPickle.dump(g, f)
	else:
		print cPickle.dumps(g)


if __name__ == "__main__":
	main()

