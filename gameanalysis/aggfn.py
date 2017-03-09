import inspect
import itertools

import numpy as np
import scipy.stats as spt

from gameanalysis import gameio
from gameanalysis import rsgame
from gameanalysis import utils


class AgfnGame(rsgame.BaseGame):
    """Action graph with function nodes game

    Action node utilities have additive structure. Function nodes are
    contribution-independent. Graph is bipartite so that function nodes have
    in-edges only from action nodes and vise versa.

    Parameters
    ----------
    num_players : ndarray or int
    num_strategies : ndarray or int
    action_weights : ndarray, float
        Each entry specifies the incoming weight in the action graph for the
        action node (column).  Must have shape (num_functions,
        num_role_strats).
    function_inputs : ndarray, bool
        Each entry specifies whether the action node (row) is an input to the
        function node (col). Must have shape (num_role_strats, num_functions).
    function_table : ndarray, float
        Value of arbitrary functions for a number of players activating the
        function. This can either have shape (num_functions, num_all_players +
        1) or (num_functions,) + tuple(num_players + 1). The former treats
        different roles ass simply different strategy sets, the later treats
        each nodes inputs as distinct, and so each function maps from the
        number of inputs from each role.
    """

    def __init__(self, num_players, num_strategies, action_weights,
                 function_inputs, function_table):
        super().__init__(num_players, num_strategies)

        self._action_weights = np.asarray(action_weights, float)
        self._function_inputs = np.asarray(function_inputs, bool)
        self._function_table = np.asarray(function_table, float)
        self.num_functions = self._function_table.shape[0]

        # Verify proper formatting of data
        assert (self._action_weights.shape
                == (self.num_functions, self.num_role_strats))
        assert (self._function_inputs.shape
                == (self.num_role_strats, self.num_functions))
        assert self._function_table.shape[0] == self.num_functions
        assert (self._function_table.shape[1] == self.num_all_players + 1) \
            or (self._function_table.shape[1:] == tuple(self.num_players + 1))
        assert self._function_inputs.any(
            0).all(), "not every function get input"
        assert np.any(self._action_weights != 0,
                      1).all(), "not every function used"
        assert np.any(self._action_weights != 0, 0).all(
        ), "strategy doesn't get payoff"

        # Select appropriate functions for data format
        if not self.is_symmetric() and len(self._function_table.shape) == 2:
            self.deviation_payoffs = self._deviation_payoffs_sum
            self.get_payoffs = self._get_payoffs_sum

        # Compute minimum and maximum payoffs
        node_table = self._function_table.reshape((self.num_functions, -1))
        minima = node_table.min(1, keepdims=True).repeat(
            self.num_role_strats, 1)
        minima[self._action_weights <= 0] = 0
        maxima = node_table.max(1, keepdims=True).repeat(
            self.num_role_strats, 1)
        maxima[self._action_weights >= 0] = 0
        self._min_payoffs = self.role_reduce(
            np.sum((minima + maxima) * self._action_weights, 0),
            ufunc=np.minimum)
        minima = node_table.min(1, keepdims=True).repeat(
            self.num_role_strats, 1)
        minima[self._action_weights >= 0] = 0
        maxima = node_table.max(1, keepdims=True).repeat(
            self.num_role_strats, 1)
        maxima[self._action_weights <= 0] = 0
        self._max_payoffs = self.role_reduce(
            np.sum((minima + maxima) * self._action_weights, 0),
            ufunc=np.maximum)

        # Pre-compute derivative info
        self._dinputs = np.zeros(
            (self.num_role_strats, self.num_functions, self.num_roles), bool)
        self._dinputs[np.arange(self.num_role_strats),
                      :, self.role_indices] = self._function_inputs

        # Set data to read only
        self._action_weights.setflags(write=False)
        self._function_inputs.setflags(write=False)
        self._function_table.setflags(write=False)
        self._dinputs.setflags(write=False)
        self._min_payoffs.setflags(write=False)
        self._max_payoffs.setflags(write=False)

    def is_complete(self):
        return True

    def min_payoffs(self):
        """Returns a lower bound on the payoffs."""
        return self._min_payoffs

    def max_payoffs(self):
        """Returns an upper bound on the payoffs."""
        return self._max_payoffs

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        """Get the deviation payoffs"""
        role_node_probs = np.minimum(self.role_reduce(
            mix[:, None] * self._function_inputs, 0), 1)[..., None]
        table_probs = np.ones(
            (self.num_roles, self.num_functions) + tuple(self.num_players + 1),
            float)
        for i, (num_play, probs, zp) in enumerate(zip(
                self.num_players, role_node_probs, self.zero_prob)):
            role_probs = spt.binom.pmf(
                np.arange(num_play + 1), num_play, probs)
            dev_role_probs = spt.binom.pmf(
                np.arange(num_play + 1), num_play - 1, probs)

            new_shape = [self.num_functions] + [1] * self.num_roles
            new_shape[i + 1] = num_play + 1
            role_probs.shape = new_shape
            dev_role_probs.shape = new_shape

            table_probs[:i] *= role_probs
            table_probs[i] *= dev_role_probs
            table_probs[i + 1:] *= role_probs

        dev_probs = self.role_repeat(table_probs, 0)
        for role, (rinps, rdev_probs) in enumerate(zip(
                self.role_split(self._function_inputs, 0),
                self.role_split(dev_probs, 0))):
            rdev_probs[rinps] = np.roll(rdev_probs[rinps], 1, role + 1)
        dev_vals = np.reshape(dev_probs * self._function_table,
                              (self.num_role_strats, self.num_functions, -1))
        devs = np.sum(np.sum(dev_vals, -1) * self._action_weights.T, -1)

        if not jacobian:
            return devs

        deriv = np.empty((self.num_roles, self.num_roles, self.num_functions) +
                         tuple(self.num_players + 1), float)
        for i, (num_play, probs, zp) in enumerate(zip(
                self.num_players, role_node_probs, self.zero_prob)):
            configs = np.arange(num_play + 1)
            der = configs / (probs + zp) - configs[::-1] / (1 - probs + zp)
            dev_der = np.insert(configs[:-1] / (probs + zp) - configs[-2::-1] /
                                (1 - probs + zp), num_play, 0, 1)

            new_shape = [self.num_functions] + [1] * self.num_roles
            new_shape[i + 1] = num_play + 1
            der.shape = new_shape
            dev_der.shape = new_shape

            deriv[:i, i] = der
            deriv[i, i] = dev_der
            deriv[i + 1:, i] = der

        dev_deriv = self.role_repeat(np.rollaxis(deriv, 2, 1), 0)
        for role, (rinps, rdev_deriv) in enumerate(zip(
                self.role_split(self._function_inputs, 0),
                self.role_split(dev_deriv, 0))):
            rdev_deriv[rinps] = np.roll(rdev_deriv[rinps], 1, role + 2)

        dev_values = dev_probs[:, :, None] * \
            dev_deriv * self._function_table[:, None]
        dev_values.shape = (self.num_role_strats,
                            self.num_functions, self.num_roles, -1)
        jac = np.sum(
            np.sum(dev_values.sum(-1)[:, None] * self._dinputs, -1) *
            self._action_weights.T[:, None], -1)
        jac -= self.role_repeat(self.role_reduce(jac) / self.num_strategies)
        return devs, jac

    def _deviation_payoffs_sum(self, mix, assume_complete=True,
                               jacobian=False):
        role_node_probs = np.minimum(self.role_reduce(
            mix[:, None] * self._function_inputs, 0), 1)[..., None]
        role_fft = np.ones((self.num_roles, self.num_functions,
                            self.num_all_players + 1), np.complex128)
        if jacobian:
            drole_fft = np.ones(
                (self.num_roles, self.num_roles, self.num_functions,
                 self.num_all_players + 1), np.complex128)

        for i, (num_play, probs, zp) in enumerate(zip(
                self.num_players, role_node_probs, self.zero_prob)):
            configs = np.arange(num_play + 1)
            role_probs = spt.binom.pmf(configs, num_play, probs)
            dev_role_probs = spt.binom.pmf(configs[:-1], num_play - 1, probs)
            fft = np.fft.fft(role_probs, self.num_all_players + 1)
            dev_fft = np.fft.fft(dev_role_probs, self.num_all_players + 1)

            role_fft[:i] *= fft
            role_fft[i] *= dev_fft
            role_fft[i + 1:] *= fft

            if not jacobian:
                continue

            der = configs / (probs + zp) - configs[::-1] / (1 - probs + zp)
            dev_der = (configs[:-1] / (probs + zp) - configs[-2::-1] /
                       (1 - probs + zp))
            dfft = np.fft.fft(der * role_probs, self.num_all_players + 1)
            ddev_fft = np.fft.fft(dev_der * dev_role_probs,
                                  self.num_all_players + 1)

            drole_fft[:i, :i] *= fft
            drole_fft[i, :i] *= dev_fft
            drole_fft[i + 1:, :i] *= fft
            drole_fft[:i, i] *= dfft
            drole_fft[i, i] *= ddev_fft
            drole_fft[i + 1:, i] *= dfft
            drole_fft[:i, i + 1:] *= fft
            drole_fft[i, i + 1:] *= dev_fft
            drole_fft[i + 1:, i + 1:] *= fft

        dev_probs = self.role_repeat(utils.simplex_project(
            np.fft.ifft(role_fft).real), 0)
        dev_probs[self._function_inputs] = np.roll(
            dev_probs[self._function_inputs], 1, 1)
        devs = np.sum(np.sum(dev_probs * self._function_table, -1) *
                      self._action_weights.T, -1)

        if not jacobian:
            return devs

        dev_deriv = self.role_repeat(np.rollaxis(
            np.fft.ifft(drole_fft).real, 2, 1), 0)
        dev_deriv[self._function_inputs] = np.roll(
            dev_deriv[self._function_inputs], 1, 2)

        dev_values = dev_deriv * self._function_table[:, None]
        jac = np.sum(
            np.sum(dev_values.sum(-1)[:, None] * self._dinputs, -1) *
            self._action_weights.T[:, None], -1)
        jac -= self.role_repeat(self.role_reduce(jac) / self.num_strategies)
        return devs, jac

    def get_payoffs(self, profile):
        """Returns an array of profile payoffs."""
        function_inputs = self.role_reduce(
            profile[:, None] * self._function_inputs, 0)
        inds = (np.arange(self.num_functions),) + tuple(function_inputs)
        function_outputs = self._function_table[inds]
        payoffs = function_outputs.dot(self._action_weights)
        payoffs[profile == 0] = 0
        return payoffs

    def _get_payoffs_sum(self, profile):
        function_inputs = profile.dot(self._function_inputs)
        function_outputs = self._function_table[np.arange(self.num_functions),
                                                function_inputs]
        payoffs = function_outputs.dot(self._action_weights)
        payoffs[profile == 0] = 0
        return payoffs

    def to_rsgame(self):
        """Builds an rsgame.Game object that represents the same game."""
        profiles = self.all_profiles()
        payoffs = np.empty(profiles.shape, float)
        for prof, pay in zip(profiles, payoffs):
            np.copyto(pay, self.get_payoffs(prof))
        return rsgame.game(self.num_players, self.num_strategies, profiles,
                           payoffs)

    def __repr__(self):
        return '{old}, {nfuncs:d})'.format(
            old=super().__repr__()[:-1],
            nfuncs=self.num_functions)

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.all(self.num_strategies == other.num_strategies) and
                np.all(self.num_players == other.num_players) and
                np.all(self._function_inputs == other._function_inputs) and
                np.allclose(self._action_weights, other._action_weights) and
                self._function_table.shape == other._function_table.shape and
                np.allclose(self._function_table, other._function_table))


def aggfn(num_players, num_strategies, action_weights, function_inputs,
          function_table):
    """Static constructor for AgfnGame

    Parameters
    ----------
    num_players : ndarray or int
    num_strategies : ndarray or int
    action_weights : ndarray, float
        Each entry specifies the incoming weight in the action graph for the
        action node (column).  Must have shape (num_functions,
        num_role_strats).
    function_inputs : ndarray, bool
        Each entry specifies whether the action node (row) is an input to the
        function node (col). Must have shape (num_role_strats, num_functions).
    function_table : ndarray, float
        Value of arbitrary functions for a number of players activating the
        function. This can either have shape (num_functions, num_all_players +
        1) or (num_functions,) + tuple(num_players + 1). The former treats
        different roles ass simply different strategy sets, the later treats
        each nodes inputs as distinct, and so each function maps from the
        number of inputs from each role.
    """
    return AgfnGame(num_players, num_strategies, action_weights,
                    function_inputs, function_table)


def aggfn_copy(copy_game, action_weights, function_inputs, function_table):
    """Static constructor for AgfnGame

    Parameters
    ----------
    copy_game : BaseGame
        Copies players and strategies from base game.
    action_weights : ndarray, float
        Each entry specifies the incoming weight in the action graph for the
        action node (column).  Must have shape (num_functions,
        num_role_strats).
    function_inputs : ndarray, bool
        Each entry specifies whether the action node (row) is an input to the
        function node (col). Must have shape (num_role_strats, num_functions).
    function_table : ndarray, float
        Value of arbitrary functions for a number of players activating the
        function. This can either have shape (num_functions, num_all_players +
        1) or (num_functions,) + tuple(num_players + 1). The former treats
        different roles ass simply different strategy sets, the later treats
        each nodes inputs as distinct, and so each function maps from the
        number of inputs from each role.
    """
    return aggfn(copy_game.num_players, copy_game.num_strategies,
                 action_weights, function_inputs, function_table)


# TODO Add from_rsgame that works via least squares fit. This isn't exactly
# trivial as you likely want sparsity in action_weights and function_inputs.
# Doing so probably undoes any nice optimization things.


def aggfn_funcs(num_players, num_strategies, action_weights, function_inputs,
                functions):
    """Static constructor for AgfnGame with functions

    This is generally less efficient than just constructing the function table
    using vectorized operations.

    Parameters
    ----------
    num_players : ndarray or int
    num_strategies : ndarray or int
    action_weights : ndarray, float
        Each entry specifies the incoming weight in the action graph for the
        action node (column).  Must have shape (num_functions,
        num_role_strats).
    function_inputs : ndarray, bool
        Each entry specifies whether the action node (row) is an input to the
        function node (col). Must have shape (num_role_strats, num_functions).
    functions : [f(n, ...) -> float]
        List of functions that either map total player activations or player
        per role activations to a single value.
    """
    assert functions, "must have at least one function"
    base = rsgame.basegame(num_players, num_strategies)
    num_functions = len(functions)
    num_params = _num_args_safe(functions[0])
    assert all(num_params == _num_args_safe(f) for f in functions)

    if num_params == 1:
        # sum format
        function_table = np.empty(
            (num_functions, base.num_all_players + 1), float)
        for func, tab in zip(functions, function_table):
            for p in range(base.num_all_players + 1):
                tab[p] = func(p)
    else:
        # role format
        assert num_params == base.num_roles
        function_table = np.empty(
            (num_functions,) + tuple(base.num_players + 1),
            float)
        for func, tab in zip(functions, function_table):
            for p in itertools.product(*map(range, base.num_players + 1)):
                tab[p] = func(*p)

    return aggfn(num_players, num_strategies, action_weights, function_inputs,
                 function_table)


def aggfn_funcs_copy(copy_game, action_weights, function_inputs, functions):
    """Static constructor for AgfnGame with functions

    This is generally less efficient than just constructing the function table
    using vectorized operations.

    Parameters
    ----------
    copy_game : BaseGame
        The base game to take player and strategy counts from.
    action_weights : ndarray, float
        Each entry specifies the incoming weight in the action graph for the
        action node (column).  Must have shape (num_functions,
        num_role_strats).
    function_inputs : ndarray, bool
        Each entry specifies whether the action node (row) is an input to the
        function node (col). Must have shape (num_role_strats, num_functions).
    functions : [f(n, ...) -> float]
        List of functions that either map total player activations or player
        per role activations to a single value.
    """
    return aggfn_funcs(copy_game.num_players, copy_game.num_strategies,
                       action_weights, function_inputs, functions)


class AgfnGameSerializer(gameio.GameSerializer):
    """A serializer for agfn games

    Parameters
    ----------
    role_names : [str]
        Names of each role.
    strat_names : [[str]]
        Names of each strategy for each role.
    function_names : [str]
        Names of each function in order.
    """

    def __init__(self, role_names, strat_names, function_names):
        super().__init__(role_names, strat_names)
        self.function_names = tuple(function_names)
        self._function_index = {f: i for i,
                                f in enumerate(self.function_names)}
        self.num_functions = len(self.function_names)

    def function_index(self, func_name):
        return self._function_index[func_name]

    def from_agfngame_json(self, game):
        base = self.from_basegame_json(game)

        function_inputs = np.zeros(
            (self.num_role_strats, self.num_functions), bool)
        for func, roles in game['function_inputs'].items():
            fi = self.function_index(func)
            for role, strats in roles.items():
                for strat in strats:
                    function_inputs[self.role_strat_index(
                        role, strat), fi] = True

        action_weights = np.zeros(
            (self.num_functions, self.num_role_strats), float)
        for role, strats in game['action_weights'].items():
            for strat, funcs in strats.items():
                rsi = self.role_strat_index(role, strat)
                for func, val in funcs.items():
                    action_weights[self.function_index(func), rsi] = val

        function_list = [None] * self.num_functions
        for func, table in game['function_tables'].items():
            function_list[self.function_index(func)] = np.asarray(table, float)

        return aggfn_copy(base, action_weights, function_inputs,
                          np.asarray(function_list, float))

    def to_agfngame_json(self, game):
        assert isinstance(game, AgfnGame)
        res = self.to_basegame_json(game)
        res['function_names'] = self.function_names

        finputs = {}
        for func, finp in zip(self.function_names, game._function_inputs.T):
            finputs[func] = {
                role: [s for s, inp in zip(strats, rinp) if inp]
                for role, strats, rinp
                in zip(self.role_names, self.strat_names,
                       self.role_split(finp))
                if rinp.any()}
        res['function_inputs'] = finputs

        act_weights = {}
        for role, strats, role_acts in zip(
                self.role_names, self.strat_names,
                self.role_split(game._action_weights)):
            if not np.allclose(role_acts, 0):
                act_weights[role] = {
                    strat: {f: w for f, w
                            in zip(self.function_names, strat_acts)
                            if not np.isclose(w, 0)}
                    for strat, strat_acts in zip(strats, role_acts.T)
                    if not np.allclose(strat_acts, 0)}
        res['action_weights'] = act_weights

        res['function_tables'] = dict(zip(
            self.function_names, (tab.tolist() for tab in
                                  game._function_table)))

        return res

    def __repr__(self):
        return '{}, {})'.format(super().__repr__(), self.function_names)

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.role_names == other.role_names and
                self.strat_names == other.strat_names and
                self.function_names == other.function_names)


def aggfnserializer(role_names, strat_names, function_names):
    """Static constructor for AgfnGameSerializer

    Parameters
    ----------
    role_names : [str]
    strat_names : [[str]]
    function_names : [str]
    """
    return AgfnGameSerializer(role_names, strat_names, function_names)


def aggfnserializer_json(json):
    """Static constructor for AgfnGameSerializer

    Takes a game that would be loaded from json and determines field names.

    Parameters
    ----------
    json : json
        A json format of a base AgfnGame. One standard output is the one output
        by to_agfngame_json. {strategies: {<role>: [<strat>]}, function_names:
        [<func>]}
    """
    serial = gameio.gameserializer_json(json)
    function_names = json['function_names']
    return AgfnGameSerializer(serial.role_names, serial.strat_names,
                              function_names)


def read_agfngame(json):
    serial = aggfnserializer_json(json)
    return serial.from_agfngame_json(json), serial


def _num_args(func):
    return sum(1 for p in inspect.signature(func).parameters.values()
               if p.default is p.empty)


def _num_args_safe(func):
    try:
        return _num_args(func)
    except AttributeError:  # pragma: no cover
        return _num_args(func.__call__)
