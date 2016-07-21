import numpy as np
import scipy.special as sps
from scipy.misc import comb
from random import sample
from itertools import combinations_with_replacement as CwR

from gameanalysis import rsgame
import sys

tiny = float(np.finfo(np.float64).tiny)

class Sym_AGG_FNA(rsgame.EmptyGame):
    def __init__(self, players, strategies, function_nodes, action_graph, \
                 utilities, functions):
        """
        Symmetric Action Graph Games with Function Nodes + Additive Structure
        Inputs:
            players: count
            strategies: [strategies]
            function_nodes: [function nodes]
            action_graph: {strategy: [neighbor]}
            utilities: {strategy: np.array[weight]}
                    For each strategy, store the weight of all its neighbors
            functions: {function node: list(type of function,(param))}
        Class Variables:
            self.neighbors: {node: {neighbor: neighbor's local index}}
            self.neighbor_index: {node: [if a node is in this neighborhood]}
            self._dev_reps: np.array[(n-1)C(0),(n-1)C(1),...,(n-1)C(n-1)]
            self.func_table: store the tabular format of functions
        """
        super().__init__({"All":players}, {"All":set(sorted(strategies))})
        self.function_nodes = function_nodes
        self.utilities = utilities
        self.functions = functions
        self.nodes = sorted(strategies) + function_nodes
        self.neighbors = {s:{n:i \
                             for i,n in enumerate(sorted(action_graph[s]))} \
                                        for s in self.nodes}
        self.neighbor_index = {s:np.array([n in self.neighbors[s] \
                                            for n in sorted(self.strategies['All'])]) \
                                            for s in self.nodes}
        self.dev_reps = np.array([np.log(comb(players-1,i)) \
                                        for i in range(players)])

        self.func_table = {p: self.build_func_table(self.functions[p],players) \
                                for p in self.function_nodes}

    def build_func_table(self,func,N):
        """
        Input:
          function: a [func_type, parameters] list
          N: number of players
        Output:
          function values represented as a list
        """
        func_type = func[0]
        param = func[1]
        if func_type == 'linear':
            k, c = param
            return [k*n + c for n in range(N+1)]
        elif func_type == 'quadratic':
            a,b,c = param
            return [a*n**2 + b*n + c for n in range(N+1)]
        else:
            raise KeyError("No such function type")

    @staticmethod
    def from_json(json_):
        """
        Build a game from the information stored in a dictionary in the json
        format
        """
        return Sym_AGG_FNA(json_['players'],
                           json_['strategies'],
                           json_['function_nodes'],
                           json_['action_graph'],
                           json_['utilities'],
                           json_['functions'])


    def min_payoffs(self, as_array=False):
        """
        Find a lower bound of the payoffs
        """
        # d is a dictionary that maps a node to its (min,max) values
        d = {}
        for node in self.nodes:
            if node in self.function_nodes:
                d[node] = (np.min(self.func_table[node]),
                           np.max(self.func_table[node]))
            elif node in self.strategies['All']:
                d[node] = (0, self.players['All'])

        min_payoff = float('inf')
        for s in self.strategies['All']:
            min_config = np.zeros(len(self.neighbors[s]))
            for n,i in self.neighbors[s].items():
                if self.utilities[s][i] >= 0:
                    min_config[i] = d[n][0]
                else:
                    min_config[i] = d[n][1]
            min_s = np.dot(np.array(min_config), self.utilities[s])
            if min_s < min_payoff:
                min_payoff = min_s

        if as_array or as_array is None:
            return np.array([min_payoff])
        else:
            return {'All': min_payoff}


    def deviation_payoffs(self, mix, verify=True, assume_complete=False,
                          jacobian=False, as_array=False):
        """
        Computes the deviation_payoffs
        EVs: list of payoffs of playing each strategy against the mixture
        local_mix: the mixture projected to this neighborhood
        """
        mix = self.as_mixture(mix, verify=False, as_array=True)
        EVs = []

        p_probs = [sum(mix[self.neighbor_index[p]]) for p in self.function_nodes]
        probs = np.append(mix, np.array(p_probs))

        for strat in sorted(self.strategies['All']):

            # EV:
            EV = np.zeros(len(self.neighbors[strat]),float)
            for s,i in self.neighbors[strat].items():
                prob = probs[self.nodes.index(s)]
                if prob + tiny >= 1:
                    prob = 1 - 2 * tiny

                sigma = np.array([(np.log(prob+tiny) * i + \
                                  np.log(1-prob+tiny) * (self.players['All']-i-1)) \
                                  for i in range(self.players['All'])])

                if s in self.strategies['All']:
                    f = np.array(range(self.players['All']))
                    if s == strat:
                        f += 1
                elif s in self.function_nodes:
                    if strat in self.neighbors[s]:
                        f = self.func_table[s][1:]
                    else:
                        f = self.func_table[s][:-1]

                EV[i] = np.dot(f, np.exp(self.dev_reps + sigma))
            EVs.append(np.dot(EV,self.utilities[strat]))

        return np.array(EVs)

    def to_json(self):
        """
        Creates a json format of the game for storage
        """
        json = {}
        json['players'] = self.players['All']
        json['strategies'] = list(self.strategies['All'])
        json['function_nodes'] = self.function_nodes
        action_graph = {s:list(self.neighbors[s].keys()) \
                for s in self.neighbors}
        json['action_graph'] = action_graph
        json['utilities'] = self.utilities
        json['functions'] = self.functions
        return json

    @staticmethod
    def randomAGG(num_players, num_strats, num_FNs, D_min=0, D_max=-1,
                  w_mean=0, w_var=3):
        """
        D_min: minimum in-degree for an action node
        D_max: maximum in-degree for an action node
        w_mean: mean of weights
        w_var: variance of weights
        """
        if D_min < 0 or D_min >= num_strats:
            D_min = 0
        if D_max < 0 or D_max >= num_strats:
            D_max = num_strats / 2

        # This maps a function to the (mean, var) tuple for params
        func = {
                'quadratic': ([0,0,0],[tiny,2,1]),
                'linear': ([0,0],[2,2])
        }

        strategies = ["s"+str(i) for i in range(num_strats)]
        FNs = ["p"+str(i) for i in range(num_FNs)]
        nodes = strategies + FNs
        action_graph = {}
        utilities = {}
        functions = {}

        # Connect the function nodes first
        for fn in FNs:
            num_neighbors = np.random.randint(num_strats/2, num_strats)
            neighbors = sorted(sample(strategies, num_neighbors))
            action_graph[fn] = neighbors
            func_type = sample(func.keys(),1)[0]
            param = [np.random.normal(m,v) for m,v in zip(*func[func_type])]
            functions[fn] = [func_type, tuple(param)]

        for s, strat in enumerate(strategies):
            num_neighbors = np.random.randint(D_min,D_max+1)
            neighbors = sorted(sample(strategies[:s] + strategies[s+1:] + FNs,\
                                      num_neighbors) + [strat])
            action_graph[strat] = neighbors
            u = [np.random.normal(w_mean,w_var) for neighbor in neighbors]
            utilities[strat] = np.array(u)

        return Sym_AGG_FNA(num_players, strategies, FNs, action_graph,
                           utilities, functions)

    def pure_payoff(self, strat, profile):
        """
        Returns the payoff to the given pure strategy profile
        Input:
            strat: strategy
            prof: profile
        Output:
            The payoff
        """
        payoff = 0
        for s,i in self.neighbors[strat].items():
            if s in self.strategies['All']:
                count = profile[sorted(self.strategies['All']).index(s)]
                payoff += self.utilities[strat][i] * count
            if s in self.function_nodes:
                count = 0
                for n in self.neighbors[s]:
                    count += profile[ sorted(self.strategies['All']).index(n) ]
                payoff += self.utilities[strat][i] * self.func_table[s][count]

        return payoff

    def to_rsgame(self):
        """
        This method builds an rsgame object that represent the same
        game. There will only be one role "All" in the constructed
        rsgame
        """
        aprofiles = []
        apayoffs = []
        for profile in CwR(self.strategies['All'],self.players['All']):
            count = np.array(
                    [profile.count(n) for n in sorted(self.strategies['All'])])
            u = np.array(
                [self.pure_payoff(n, count) for n in sorted(self.strategies['All'])])
            nz = np.nonzero(count)
            mask = np.zeros(count.shape)
            mask[nz] = 1
            u = u * mask
            aprofiles.append(count)
            apayoffs.append(u)
        aprofiles = np.array(aprofiles)
        apayoffs = np.array(apayoffs)
        return rsgame.Game(self.players, self.strategies,
                           aprofiles, apayoffs)
