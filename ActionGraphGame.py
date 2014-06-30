import numpy as np
from random import sample
from itertools import combinations_with_replacement as CwR

from BasicFunctions import profile_repetitions
from HashableClasses import h_dict
import RoleSymmetricGame as RSG

tiny = float(np.finfo(np.float64).tiny)

class Sym_AGG:
	def __init__(self, players, action_graph={}, utilities={}):
		"""
		Symmetric action graph game.
		
		Supports drawing noisy payoff samples and computing regret
		of symmetric mixed strategy profiles.

		players: 		Integer specifying the number of players
		action_graph: 	Mapping of strategies to lists of strategies. The lists
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

	def regret(self, mix):
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
		return max(EVs - sum(mix * EVs))


class Noisy_AGG(Sym_AGG):
	def __init__(self, players, action_graph={}, utilities={}, \
				noise=lambda s,p,c:np.random.normal(0,1,c)):
		"""
		noise should be a random function from strategy s and profile p 
		to an array of floats with length c.
		"""
		self.noise = noise
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
				noisy_vals[strat] = val + self.noise(strat, profile, count)
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
		


def local_effect_AGG(N, S, D_min=0, D_max=-1, sigma=1):
	"""
	Creates an AGG representing a noisy LEG with quadratic payoff functions

	parameters:
	N		number of players
	S		number of strategies
	D_min	minimum action-graph degree (default=0)
	D_max	maximum action-graph degree (default=S)
	sigma	noise magnitude
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
		local_effects = {n:np.random.normal(0,1,3) * np.arange(3,0,-1) for \
						n in action_graph[strat]}
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
	noise = lambda s,p,c: np.random.normal(0,sigma,c)
	return Noisy_AGG(N, action_graph, utilities, noise)
