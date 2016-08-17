import numpy as np
import scipy.special as sps
from scipy.misc import comb
from scipy.special import gammaln
from random import sample
from itertools import combinations_with_replacement as CwR

from gameanalysis import rsgame
import sys

_TINY = float(np.finfo(np.float64).tiny)

class Sym_AGG_FNA(rsgame.BaseGame):
    """Action Graph Game with Function Nodes.

    Represented games are symmetric. Action node utilities have additive structure.
    Function nodes are contribution-independent. Function nodes have in-edges only
    from action nodes.
    """
    def __init__(self, num_players, num_strategies, action_weights, function_inputs=[],
                 node_functions=[]):
        """
        Parameters
        ----------
            num_players
            num_strategies
            action_weights: floating point matrix with num_strategies rows and
                    (num_strategies + |node_functions|) columns. Each entry specifies the
                    incoming weight in the action graph for the action node (row).
            function_inputs: boolean matrix with |node_functions| rows and num_strategies
                    columns. Each entry specifies whether the action node (column) is an
                    input to the function node (row).
            node_functions: Activation functions for each function node.
                    This can either be a list of python functions that work
                    correctly when applied to a vector of inputs, or an array
                    already storing the correct tabular representation.
        Class Variables
        ---------------
            self.action_weights
            self.function_inputs
            self.configs
            self.log_dev_reps
            self.func_table
        """
        super().__init__(num_players, num_strategies)
        self.action_weights = np.array(action_weights, dtype=float)
        self.function_inputs = np.array(function_inputs, dtype=bool)
        self.configs = np.arange(num_players+1)[:,None]
        self.log_dev_reps = gammaln(num_players) - gammaln(self.configs[1:]) - \
                            gammaln(num_players - self.configs[:-1])
        if isinstance(node_functions, np.ndarray): # already a table
            self.func_table = node_functions
        else: # list of python functions, need to be applied to configurations
            self.func_table = np.array([f(self.configs[:,0]) for f in
                                        node_functions], dtype=float).T
        self.num_funcs = self.func_table.shape[1]
        self.num_nodes = self.num_funcs + self.num_strategies[0]
        self._min_payoffs = None
        self._max_payoffs = None


    @staticmethod
    def from_json(json_):
        """
        Build a game from the information stored in a dictionary in the json
        format
        """
        num_players = json_["num_players"]
        strategy_names = sorted(json_["strategy_names"])
        num_strats = len(strategy_names)
        function_names = sorted(json_["function_names"])
        num_funcs = len(function_names)
        functions = np.array([json_["function_tables"][f] for
                              f in function_names]).T

        action_weights = np.empty([num_strats, num_strats + num_funcs],
                                  dtype=float)
        function_inputs = np.zeros([num_strats, num_funcs], dtype=bool)

        for s,strat in enumerate(strategy_names):
            for n,node in enumerate(strategy_names + function_names):
                action_weights[s][n] = json_["action_weights"][strat].get(node,0)
            for f,func in enumerate(function_names):
                if strat in json_["function_inputs"][func]:
                    function_inputs[s][f] = True
        return Sym_AGG_FNA(num_players, num_strats, action_weights,
                           function_inputs, functions)


    def min_payoffs(self):
        """Returns a lower bound on the payoffs."""
        if self._min_payoffs is None:
            minima = np.zeros([self.num_strategies[0], self.num_nodes])
            minima[:,-self.num_funcs:] = self.func_table.min(0)
            minima[self.action_weights <= 0] = 0

            maxima = np.zeros(minima.shape)
            maxima[:,-self.num_funcs:] = self.func_table.max(0)
            maxima[self.action_weights >= 0] = 0

            self._min_payoffs = ((minima + maxima) *
                                 self.action_weights).sum(1).min(keepdims=True)
            self._min_payoffs.setflags(write=False)
        return self._min_payoffs.view()


    def max_payoffs(self):
        """Returns an upper bound on the payoffs."""
        if self._max_payoffs is None:
            minima = np.zeros([self.num_strategies[0], self.num_nodes])
            minima[:,-self.num_funcs:] = self.func_table.min(0)
            minima[self.action_weights >= 0] = 0

            maxima = np.zeros(minima.shape)
            maxima[:,-self.num_funcs:] = self.func_table.max(0)
            maxima[self.action_weights <= 0] = 0

            self._max_payoffs = ((minima + maxima) *
                                 self.action_weights).sum(1).max(keepdims=True)
            self._max_payoffs.setflags(write=False)
        return self._max_payoffs.view()

    def is_complete(self):
        # Action graph games are always complete
        return True

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        # TODO To add jacobian support.
        assert not jacobian, "Sym_AGG_FNA doesn't support jacobian"
        func_node_probs = mix[:,None].repeat(self.num_funcs, 1)
        func_node_probs[np.logical_not(self.function_inputs)] = 0
        func_node_probs = func_node_probs.sum(0)

        act_conf_probs = np.exp(np.log(mix + _TINY) * self.configs[:-1] +
                  np.log(1 - mix + _TINY) * (self.num_players -
                  self.configs[1:]) + self.log_dev_reps)
        func_conf_probs = np.exp(np.log(func_node_probs + _TINY) *
                  self.configs[:-1] + np.log(1 - func_node_probs + _TINY) *
                  (self.num_players - self.configs[1:]))

        EVs = np.empty(self.num_strategies)
        for s in range(self.num_strategies[0]):
            action_outputs = self.configs[:-1].repeat(self.num_strategies, 1)
            action_outputs[:,s] += 1
            function_outputs = np.array(self.func_table[:-1])
            function_outputs[:,self.function_inputs[s]] = \
                            self.func_table[1:,self.function_inputs[s]]
            node_EVs = np.append((act_conf_probs * action_outputs).sum(0),
                                 (func_conf_probs * function_outputs).sum(0))
            EVs[s] = np.dot(node_EVs, self.action_weights[s])
        return EVs


    def to_json(self):
        """
        Creates a json format of the game for storage
        """
        raise NotImplementedError("not yet compatible with array implementation.")
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
        raise NotImplementedError("not yet compatible with array implementation.")
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
        raise NotImplementedError("not yet compatible with array implementation.")
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
        raise NotImplementedError("not yet compatible with array implementation.")
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
