import numpy as np
from random import sample
from itertools import combinations_with_replacement as CwR
from os.path import join, exists
from os import mkdir
from cPickle import dump
from argparse import ArgumentParser

from BasicFunctions import profile_repetitions, leading_zeros
from HashableClasses import h_dict
import RoleSymmetricGame as RSG

tiny = float(np.finfo(np.float64).tiny)

class Sym_AGG:
	def __init__(self, players, action_graph={}, utilities={}):
		"""
		Symmetric action graph game.

		Supports drawing noisy payoff samples and computing regret
		of symmetric mixed strategy profiles.

		players:		Integer specifying the number of players
		action_graph:	Mapping of strategies to lists of strategies. The lists
						specify origins of in-edges in the action graph, so
						that each strategy is paired with the list of strategies
						on which its payoff depends. If a strategy's payoff
						depends on its own count (usually the case) it should
						have a self-edge in the action graph.
		utilities:		Mapping of strategies to payoff functions. Each payoff
						function is specified as mapping from neighborhood
						configurations (denoted by h_dict symmetric profiles)
						to floating point payoff values.
		"""
		self.players = players
		self.strategies = tuple(sorted(action_graph.keys()))
		self.neighbors = {s:{n:i for i,n in enumerate(sorted(action_graph[s]))}\
							for s in self.strategies}
		self.neighbor_index = {s:np.array([n in self.neighbors[s] for n in \
								self.strategies]) for s in self.strategies}
		self.values = {s:[] for s in self.strategies}
		self.counts = {s:[] for s in self.strategies}
		self.dev_reps = {s:[] for s in self.strategies}
		self.indices = {s:{} for s in self.strategies}

		for s,strat in enumerate(self.strategies):
			if strat not in utilities:
				continue
			for prof,val in utilities[strat].iteritems():
				self.indices[strat][prof] = len(self.values[strat])
				self.values[strat].append(val)
				counts = np.zeros(len(self.neighbors[strat])+1, dtype=int)
				for n,i in self.neighbors[strat].iteritems():
					if n in prof:
						counts[i] = prof[n]
				#last column gives number of players outside the neighborhood
				counts[-1] = self.players - counts.sum()
				self.counts[strat].append(counts)
				devs = np.array(counts, dtype=int)
				if strat in self.neighbors[strat]:
					devs[self.neighbors[strat][strat]] -= 1
				else:
					devs[-1] -= 1
				self.dev_reps[strat].append(profile_repetitions([devs]))

		self.values = {s:np.array(v) for s,v in self.values.iteritems()}
		self.counts = {s:np.array(c) for s,c in self.counts.iteritems()}
		self.dev_reps = {s:np.array(d) for s,d in self.dev_reps.iteritems()}

	def expectedValues(self, mix):
		EVs = np.zeros(mix.shape, dtype=float)
		for s,strat in enumerate(self.strategies):
			local_mix = list(mix[self.neighbor_index[strat]])
			local_mix.append(mix[True - self.neighbor_index[strat]].sum())
			local_mix = np.array(local_mix)
			try:
				local_index = self.neighbors[strat][strat]
			except ValueError:
				local_index = len(self.neighbors[strat])
			EVs[s] = (self.values[strat] * ((local_mix + tiny) ** \
						self.counts[strat]).prod(1) * self.dev_reps[strat] / \
						(local_mix[local_index] + tiny)).sum()
		return EVs

	def regret(self, mix):
		EVs = self.expectedValues(mix)
		return max(EVs - sum(mix * EVs))

	def getExpectedPayoff(self, mix):
		EVs = self.expectedValues(mix)
		return sum(mix * EVs)


class Noisy_AGG(Sym_AGG):
	def __init__(self, players, action_graph={}, utilities={}, sigma=1):
		"""
		noise should be a random function from strategy s and profile p
		to an array of floats with length c.
		"""
		self.sigma = sigma
		Sym_AGG.__init__(self, players, action_graph, utilities)

	def sample(self, profile, count):
		"""
		set count=0 to get exact values
		"""
		noisy_vals = {}
		for strat in profile:
			p = {s:profile[s] for s in self.neighbors[strat] if s in profile}
			i = self.indices[strat][h_dict(p)]
			val = self.values[strat][i]
			if count == 0:
				noisy_vals[strat] = val
			else:
				noisy_vals[strat] = val + np.random.normal(0,self.sigma,count)
		return noisy_vals

	def sampleGame(self, count=1):
		g = RSG.SampleGame(["All"],{"All":self.players},{"All":self.strategies})
		for p in g.allProfiles():
			d = []
			for s,v in self.sample(p["All"],count).items():
				d.append(RSG.PayoffData(s,p["All"][s],v))
			g.addProfile({"All":d})
		return g

	def exactGame(self):
		g = RSG.Game(["All"],{"All":self.players},{"All":self.strategies})
		for p in g.allProfiles():
			d = []
			for s,v in self.sample(p["All"],0).items():
				d.append(RSG.PayoffData(s,p["All"][s],v))
			g.addProfile({"All":d})
		return g


def local_effect_AGG(N, S, D_min=0, D_max=-1, noise=100, self_mean=[0,-4,-2], \
				self_var=[4,2,1], other_mean=[0,0,0], other_var=[tiny,2,1]):
	"""
	Creates an AGG representing a noisy LEG with quadratic payoff functions

	parameters:
	N			number of players
	S			number of strategies
	D_min		minimum action-graph degree (default=0)
	D_max		maximum action-graph degree (default=S)
	noise		noise variance
	self_mean	means for constant, linear, and quadratic
				self-interaction coefficients
	self_var	variances for constant, linear, and quadratic
				self-interaction coefficients
	other_mean	means for constant, linear, and quadratic
				neighbor-interaction coefficients
	other_var	variances for constant, linear, and quadratic
				neighbor-interaction coefficients
	"""
	if D_min < 0 or D_min >= S:
		D_min = 0
	if D_max < 0 or D_max >= S:
		D_max = S-1
	strategies = ["s"+str(i) for i in range(S)]
	action_graph = {}
	local_effects = {}
	utilities = {}
	for s,strat in enumerate(strategies):
		num_neighbors = np.random.randint(D_min, D_max+1)
		neighbors = sorted(sample(strategies[:s] + strategies[s+1:],\
										num_neighbors) + [strat])
		action_graph[strat] = neighbors
		local_effects = {n:np.array([np.random.normal(m,v) for m,v in \
					zip(other_mean,other_var)]) for n in action_graph[strat]}
		local_effects[strat] = np.array([np.random.normal(m,v) for m,v in \
					zip(self_mean, self_var)])
		u = {}
		for i in range(N):
			for strats in CwR(neighbors, i):
				counts = {n:strats.count(n) for n in neighbors if n in strats}
				counts[strat] = counts.get(strat,0) + 1
				prof = h_dict(counts)
				u[prof] = 0
				for n,c in prof.iteritems():
					u[prof] += sum(c**np.arange(3) * local_effects[n])
		utilities[strat] = u
	return Noisy_AGG(N, action_graph, utilities, noise)


def uniform_AGG(N, S, D_min=0, D_max=-1, noise=100, min_payoff=-100, \
				max_payoff=100):
	"""
	Creates an AGG with payoff values drawn from a uniform distribution.

	parameters:
	N			number of players
	S			number of strategies
	D_min		minimum action-graph degree (default=0)
	D_max		maximum action-graph degree (default=S)
	min_payoff	min value for uniform payoff draws
	max_payoff	max value for uniform payoff draws
	noise		noise variance
	"""
	if D_min < 0 or D_min >= S:
		D_min = 0
	if D_max < 0 or D_max >= S:
		D_max = S-1
	strategies = ["s"+str(i) for i in range(S)]
	action_graph = {}
	utilities = {}
	for s,strat in enumerate(strategies):
		num_neighbors = np.random.randint(D_min, D_max+1)
		neighbors = sorted(sample(strategies[:s] + strategies[s+1:],\
										num_neighbors) + [strat])
		action_graph[strat] = neighbors
		u = {}
		for i in range(N):
			for strats in CwR(neighbors, i):
				counts = {n:strats.count(n) for n in neighbors if n in strats}
				counts[strat] = counts.get(strat,0) + 1
				prof = h_dict(counts)
				u[prof] = np.random.uniform(min_payoff, max_payoff)
		utilities[strat] = u
	return Noisy_AGG(N, action_graph, utilities, noise)


def parse_args():
	p = ArgumentParser(description="Generate random action graph games.")
	p.add_argument("folder", type=str, help="Folder to put games in.")
	p.add_argument("count", type=int, help="Number of games to create.")
	p.add_argument("players", type=int, help="Number of players.")
	p.add_argument("strategies", type=int, help="Number of strategies.")
	p.add_argument("min_neighbors", type=int, help="Mimimum number of "+\
				"other strategies that payoffs can depend on.")
	p.add_argument("max_neighbors", type=int, help="Maximum number of "+\
				"other strategies that payoffs can depend on.")
	p.add_argument("noise", type=float, help="Variance of payoff "+\
				"observation noise.")
	return p.parse_args()

def main():
	a = parse_args()
	for i in range(a.count):
		g = local_effect_AGG(a.players, a.strategies, a.min_neighbors, \
							a.max_neighbors, a.noise)
		folder = join(a.folder, "LEG_" + reduce(lambda x,y:str(x)+"-"+str(y), \
				[a.players, a.strategies, a.min_neighbors, a.max_neighbors, \
				int(a.noise)]))
		if not exists(folder):
			mkdir(folder)
		with open(join(folder, leading_zeros(i,a.count-1) + ".pkl"), "w") as f:
			dump(g,f)

if __name__ == "__main__":
	main()

