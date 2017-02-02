import numpy as np
import scipy.special as sps
from scipy.misc import comb
from scipy.special import gammaln
from numpy.random import uniform, normal, choice
from itertools import combinations_with_replacement as CwR
from functools import partial

from gameanalysis import rsgame
import sys

_TINY = float(np.finfo(np.float64).tiny)

class Sym_AGG_FNA(rsgame.BaseGame):
    """Action Graph Game with Function Nodes.

    Represented games are symmetric. Action node utilities have additive
    structure. Function nodes are contribution-independent. Graph is bipartite
    so that function nodes have in-edges only from action nodes and vise versa.
    """
    def __init__(self, num_players, num_strategies, action_weights,
                 function_inputs, node_functions):
        """
        Parameters
        ----------
            num_players
            num_strategies
            action_weights: floating point matrix with |node_functions| rows
                    num_strategies columns. Each entry specifies the incoming
                    weight in the action graph for the action node (column).
            function_inputs: boolean matrix with num_strategies rows and
                    |node_functions| columns. Each entry specifies whether the
                    action node (row) is an input to the function node (col).
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
        self.configs = np.arange(num_players+1)
        self.log_dev_reps = gammaln(num_players) - gammaln(self.configs[1:]) -\
                            gammaln(num_players - self.configs[:-1])
        if isinstance(node_functions, np.ndarray): # already a table
            self.func_table = node_functions
        else: # list of python functions, need to be applied to configurations
            self.func_table = np.array([f(self.configs) for f in
                                    node_functions], dtype=float)
        self.num_functions = self.func_table.shape[0]
        self._min_payoffs = None
        self._max_payoffs = None

    def is_complete(self):
        return True

    @staticmethod
    def from_json(j_):
        """
        Build a game from the info stored in a dictionary in the json format.
        """
        num_players = j_["num_players"]
        strategy_names = sorted(j_["strategy_names"])
        num_strats = len(strategy_names)
        function_names = sorted(j_["function_names"])
        num_functions = len(function_names)
        functions = np.array([j_["function_tables"][f] for
                              f in function_names])

        action_weights = np.empty([num_functions, num_strats], dtype=float)
        function_inputs = np.zeros([num_strats, num_functions], dtype=bool)

        for s,strat in enumerate(strategy_names):
            for f,func in enumerate(function_names):
                action_weights[f,s] = j_["action_weights"][strat].get(func,0)
            for f,func in enumerate(function_names):
                if strat in j_["function_inputs"][func]:
                    function_inputs[s,f] = True
        return Sym_AGG_FNA(num_players, num_strats, action_weights,
                           function_inputs, functions)


    def to_json(self, strategy_names=None, function_names=None):
        """
        Creates a json format of the game for storage
        """
        j_ = {}
        j_["num_players"] = int(self.num_players[0])
        if strategy_names is None:
            strategy_names = ["s" + str(i) for i in
                              range(self.num_strategies[0])]
        if function_names is None:
            function_names = ["f" + str(i) for i in range(self.num_functions)]
        j_["strategy_names"] = strategy_names
        j_["function_names"] = function_names
        j_["function_inputs"] = {}
        for f, func in enumerate(function_names):
            inputs = []
            for s, strat in enumerate(strategy_names):
                if self.function_inputs[s,f]:
                    inputs.append(strat)
            j_["function_inputs"][func] = inputs
        j_["action_weights"] = {}
        for s, strat in enumerate(strategy_names):
            weights = {}
            for f, func in enumerate(function_names):
                if self.action_weights[f,s]:
                    weights[func] = self.action_weights[f,s]
            j_["action_weights"][strat] = weights
        j_["function_tables"] = dict(zip(function_names,
                                       map(list, self.func_table)))
        return j_


    def min_payoffs(self):
        """Returns a lower bound on the payoffs."""
        if self._min_payoffs is None:
            minima = self.func_table.min(1, keepdims=True).repeat(
                                    self.num_strategies[0], axis=1)
            minima[self.action_weights <= 0] = 0

            maxima = self.func_table.max(1, keepdims=True).repeat(
                                    self.num_strategies[0], axis=1)
            maxima[self.action_weights >= 0] = 0

            self._min_payoffs = ((minima + maxima) *
                                 self.action_weights).sum(0).min(keepdims=True)
            self._min_payoffs.setflags(write=False)
        return self._min_payoffs.view()


    def max_payoffs(self):
        """Returns an upper bound on the payoffs."""
        if self._max_payoffs is None:
            minima = self.func_table.min(1, keepdims=True).repeat(
                                    self.num_strategies[0], axis=1)
            minima[self.action_weights >= 0] = 0

            maxima = self.func_table.max(1, keepdims=True).repeat(
                                    self.num_strategies[0], axis=1)
            maxima[self.action_weights <= 0] = 0

            self._max_payoffs = ((minima + maxima) *
                                 self.action_weights).sum(0).max(keepdims=True)
            self._max_payoffs.setflags(write=False)
        return self._max_payoffs.view()


    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        # TODO To add jacobian support.
        assert not jacobian, "Sym_AGG_FNA doesn't support jacobian"
        func_node_probs = mix[:,None].repeat(self.num_functions, axis=1)
        func_node_probs[np.logical_not(self.function_inputs)] = 0
        func_node_probs = func_node_probs.sum(0)

        log_input_probs = np.outer(np.log(func_node_probs + _TINY),
                                  self.configs[:-1])
        log_non_input_probs = np.outer(np.log(1 - func_node_probs + _TINY),
                                  self.num_players - self.configs[1:])
        config_probs = np.exp(log_input_probs + log_non_input_probs +
                              self.log_dev_reps)

        EVs = np.empty(self.num_strategies)
        for s in range(self.num_strategies[0]):
            # function_outputs is func_table for 0 to N-1 for functions that
            # don't have s in their neighborhood and 1 to N for those that do.
            function_outputs = np.array(self.func_table[:,:-1])
            function_outputs[self.function_inputs[s]] = \
                        self.func_table[:,1:][self.function_inputs[s]]
            node_EVs = (config_probs * function_outputs).sum(1)
            EVs[s] = np.dot(node_EVs, self.action_weights[:,s])
        return EVs


    def get_payoffs(self, profile):
        """Returns an array of profile payoffs."""
        function_inputs = (profile * self.function_inputs.T).sum(1)
        function_outputs = self.func_table[np.arange(self.num_functions),
                                           function_inputs]
        payoffs = (self.action_weights.T * function_outputs).sum(1)
        payoffs[np.logical_not(profile)] = 0
        return payoffs


    def to_rsgame(self):
        """Builds an rsgame.Game object that represents the same game."""
        profiles = self.all_profiles()
        payoffs = np.array([self.get_payoffs(p) for p in profiles])
        return rsgame.Game(self.num_players, self.num_strategies, profiles,
                           payoffs)
