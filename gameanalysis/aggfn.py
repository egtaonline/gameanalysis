"""An action graph game with additive function nodes"""
import functools
import inspect
import itertools

import numpy as np
import scipy.stats as spt

from gameanalysis import rsgame
from gameanalysis import utils


class AgfnGame(rsgame.CompleteGame):
    """Action graph with function nodes game

    Action node utilities have additive structure. Function nodes are
    contribution-independent. Graph is bipartite so that function nodes have
    in-edges only from action nodes and vise versa.

    Parameters
    ----------
    role_names : (str,)
        The name of each role.
    strat_names : ((str,),)
        The name of each strategy per role.
    num_role_players : ndarray
        The number of players for each role.
    function_names : (str,)
        The name of each function. Must be sorted, unique, and non-empty.
    action_weights : ndarray
        Each entry specifies the incoming weight in the action graph for the
        action node (column).  Must have shape (num_functions, num_strats). The
        action weights for a particular function can't be all zero, otherwise
        that function should not exist.
    function_inputs : ndarray
        Each entry specifies whether the action node (row) is an input to the
        function node (col). Must have shape (num_strats, num_functions).
    function_table : ndarray
        Value of arbitrary functions for a number of players activating the
        function. This can either have shape (num_functions, num_players + 1)
        or (num_functions,) + tuple(num_role_players + 1). The former treats
        different roles ass simply different strategy sets, the later treats
        each nodes inputs as distinct, and so each function maps from the
        number of inputs from each role.
    """
    # FIXME Currently the only way to apply an affine transformation to payoffs
    # is to add a constant function, i.e. one that's not a function of the
    # number of players. While a fine solution, it also means that now we must
    # allow there to be a function with either all true or all false function
    # inputs, and only one usable entry in the function table. In terms of
    # expressing the space compactly, it probably makes more sense for Aggfns
    # to disallow function_inputs that are all true or all false, and
    # function_table rows that are all identical, and instead have a final
    # option which is a payoff offset that gets added to all payoffs. This will
    # make normalization and assertion checking much simpler and more
    # representative of the space of Aggfns.

    def __init__(self, role_names, strat_names, num_role_players,
                 function_names, action_weights, function_inputs,
                 function_table):
        super().__init__(role_names, strat_names, num_role_players)
        self.function_names = function_names
        self.num_functions = len(function_names)
        self._action_weights = action_weights
        self._action_weights.setflags(write=False)
        self._function_inputs = function_inputs
        self._function_inputs.setflags(write=False)
        self._function_table = function_table
        self._function_table.setflags(write=False)

        # Pre-compute derivative info
        self._dinputs = np.zeros(
            (self.num_strats, self.num_functions, self.num_roles), bool)
        self._dinputs[np.arange(self.num_strats), :, self.role_indices] = (
            self._function_inputs)
        self._dinputs.setflags(write=False)

        self._function_index = {f: i for i, f in enumerate(function_names)}

    def function_index(self, func_name):
        """Get the index of a function by name"""
        return self._function_index[func_name]

    @functools.lru_cache(maxsize=1)
    def profiles(self):
        return self.all_profiles()

    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        return self.get_payoffs(self.profiles())

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns a lower bound on the payoffs."""
        node_table = self._function_table.reshape((self.num_functions, -1))
        minima = node_table.min(1, keepdims=True)
        maxima = node_table.max(1, keepdims=True)
        eff_min = np.where(self._action_weights > 0, minima, maxima)
        # Don't use min/max if function is effectively constant
        always = self._function_inputs.all(0)
        if always.any():
            eff_min[always] = node_table[always, -1]
        never = ~self._function_inputs.any(0)
        if never.any():
            eff_min[never] = node_table[never, 0]
        mins = np.sum(eff_min * self._action_weights, 0)
        mins.setflags(write=False)
        return mins.view()

    @utils.memoize
    def max_strat_payoffs(self):
        """Returns an upper bound on the payoffs."""
        node_table = self._function_table.reshape((self.num_functions, -1))
        minima = node_table.min(1, keepdims=True)
        maxima = node_table.max(1, keepdims=True)
        eff_max = np.where(self._action_weights > 0, maxima, minima)
        # Don't use min/max if function is effectively constant
        always = self._function_inputs.all(0)
        if always.any():
            eff_max[always] = node_table[always, -1]
        never = ~self._function_inputs.any(0)
        if never.any():
            eff_max[never] = node_table[never, 0]
        maxs = np.sum(eff_max * self._action_weights, 0)
        maxs.setflags(write=False)
        return maxs.view()

    def normalize(self):
        """Return a normalized AgfnGame"""
        # To normalize an aggfn, we need the help of a constant function. This
        # first step attempts to identify one if it already exists, or adds a
        # new function if it doesn't.
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0)] = 1

        norm_ftab = self._function_table.reshape((self.num_functions, -1))
        # We could use allclose here to capture almost constant functions, but
        # it's not that expensive to just add one more, so this seems "better"
        eq_value = np.all(norm_ftab[:, 0, None] == norm_ftab[:, 1:], 1)
        eq_inp = np.all(self._function_inputs[0] == self._function_inputs[1:],
                        0)
        if eq_value.any() or eq_inp.any():  # existing constant function
            if eq_value.any():  # Function with constant value exists
                const_func = eq_value.nonzero()[0][0]
                ind = 0
            else:  # Only one value from func_table is used
                const_func = eq_inp.nonzero()[0][0]
                ind = -1 if self._function_inputs[0, const_func] else 0
            # Here we do some normalization for the constant function
            action_weights = self._action_weights.copy()
            action_weights[const_func] *= norm_ftab[const_func, ind]
            function_inputs = self._function_inputs
            function_table = self._function_table.copy()
            function_table[const_func] = 1
        else:  # Must add constant function
            const_func = self.num_functions
            action_weights = np.insert(self._action_weights, const_func, 0, 0)
            function_inputs = np.insert(
                self._function_inputs, const_func, False, 1)
            function_table = np.insert(self._function_table, const_func, 1, 0)

        action_weights[const_func] -= self.min_role_payoffs().repeat(
            self.num_role_strats)
        return aggfn_replace(
            self, action_weights / scale.repeat(self.num_role_strats),
            function_inputs, function_table)

    def subgame(self, subgame_mask):
        subgame_mask = np.asarray(subgame_mask, bool)
        base = super().subgame(subgame_mask)
        action_weights = self._action_weights[:, subgame_mask]
        func_mask = np.any(~np.isclose(action_weights, 0), 1)
        if func_mask.all():
            func_names = self.function_names
        else:
            func_names = tuple(
                n for n, m in zip(self.function_names, func_mask) if m)
        return AgfnGame(base.role_names, base.strat_names,
                        base.num_role_players, func_names,
                        action_weights[func_mask],
                        self._function_inputs[:, func_mask][subgame_mask],
                        self._function_table[func_mask])

    def to_json(self):
        res = super().to_json()

        finputs = {}
        for func, finp in zip(self.function_names, self._function_inputs.T):
            finputs[func] = {
                role: [s for s, inp in zip(strats, rinp) if inp]
                for role, strats, rinp
                in zip(self.role_names, self.strat_names,
                       np.split(finp, self.role_starts[1:]))
                if rinp.any()}
        res['function_inputs'] = finputs

        act_weights = {}
        for role, strats, role_acts in zip(
                self.role_names, self.strat_names,
                np.split(self._action_weights, self.role_starts[1:], 1)):
            act_weights[role] = {
                strat: {f: float(w) for f, w
                        in zip(self.function_names, strat_acts)
                        if not np.isclose(w, 0)}
                for strat, strat_acts in zip(strats, role_acts.T)
                if not np.allclose(strat_acts, 0)}
        res['action_weights'] = act_weights

        res['function_tables'] = dict(zip(
            self.function_names, (tab.tolist() for tab in
                                  self._function_table)))
        res['type'] = 'aggfn.1'
        return res

    def __repr__(self):
        return '{old}, {nfuncs:d})'.format(
            old=super().__repr__()[:-1],
            nfuncs=self.num_functions)

    def __eq__(self, other):
        selfp = np.lexsort(
            self._function_table.reshape((self.num_functions, -1)).T)
        otherp = np.lexsort(
            other._function_table.reshape((other.num_functions, -1)).T)
        return (super().__eq__(other) and
                self.function_names == other.function_names and
                self._function_table.shape == other._function_table.shape and
                np.all(self._function_inputs[:, selfp]
                       == other._function_inputs[:, otherp]) and
                np.allclose(self._action_weights[selfp],
                            other._action_weights[otherp]) and
                np.allclose(self._function_table[selfp],
                            other._function_table[otherp]))

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_functions))


# TODO a role Aggfn is a slightly less compressed version of the SumAggfn where
# the diagonals of the role are all the same value. i.e. [0, 2] == [1, 1] ==
# [2, 0]. Potentially Sum should be removed in favor of role? It is slightly
# less compressed, and potentially less performant than the sum in these
# circumstances.
class RoleAgfnGame(AgfnGame):
    """Aggfn with functions as function of players per role"""

    def __init__(self, role_names, strat_names, num_role_players,
                 function_names, action_weights, function_inputs,
                 function_table):
        super().__init__(role_names, strat_names, num_role_players,
                         function_names, action_weights, function_inputs,
                         function_table)
        self._basis = np.insert(
            np.cumprod(self.num_role_players[:0:-1] + 1)[::-1],
            self.num_roles - 1, 1)
        self._func_offset = (np.arange(self.num_functions) *
                             np.prod(self.num_role_players + 1))

    def get_payoffs(self, profile):
        """Returns an array of profile payoffs."""
        profile = np.asarray(profile, int)
        function_inputs = np.add.reduceat(
            profile[..., None, :] * self._function_inputs.T,
            self.role_starts, -1)
        inds = function_inputs.dot(self._basis) + self._func_offset
        function_outputs = self._function_table.ravel()[inds]
        payoffs = function_outputs.dot(self._action_weights)
        payoffs[profile == 0] = 0
        return payoffs

    def deviation_payoffs(self, mix, *, jacobian=False):
        """Get the deviation payoffs"""
        mix = np.asarray(mix, float)
        role_node_probs = np.minimum(
            np.add.reduceat(mix[:, None] * self._function_inputs,
                            self.role_starts), 1)[..., None]
        table_probs = np.ones(
            (self.num_roles, self.num_functions) +
            tuple(self.num_role_players + 1),
            float)
        for i, (num_play, probs, zp) in enumerate(zip(
                self.num_role_players, role_node_probs, self.zero_prob)):
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

        dev_probs = table_probs.repeat(self.num_role_strats, 0)
        for role, (rinps, rdev_probs) in enumerate(zip(
                np.split(self._function_inputs, self.role_starts[1:], 0),
                np.split(dev_probs, self.role_starts[1:], 0))):
            rdev_probs[rinps] = np.roll(rdev_probs[rinps], 1, role + 1)
        dev_vals = np.reshape(dev_probs * self._function_table,
                              (self.num_strats, self.num_functions, -1))
        devs = np.sum(np.sum(dev_vals, -1) * self._action_weights.T, -1)

        if not jacobian:
            return devs

        deriv = np.empty((self.num_roles, self.num_roles, self.num_functions) +
                         tuple(self.num_role_players + 1), float)
        for i, (num_play, probs, zp) in enumerate(zip(
                self.num_role_players, role_node_probs, self.zero_prob)):
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

        dev_deriv = np.rollaxis(deriv, 2, 1).repeat(self.num_role_strats, 0)
        for role, (rinps, rdev_deriv) in enumerate(zip(
                np.split(self._function_inputs, self.role_starts[1:], 0),
                np.split(dev_deriv, self.role_starts[1:], 0))):
            rdev_deriv[rinps] = np.roll(rdev_deriv[rinps], 1, role + 2)

        dev_values = dev_probs[:, :, None] * \
            dev_deriv * self._function_table[:, None]
        dev_values.shape = (self.num_strats,
                            self.num_functions, self.num_roles, -1)
        jac = np.sum(
            np.sum(dev_values.sum(-1)[:, None] * self._dinputs, -1) *
            self._action_weights.T[:, None], -1)
        jac -= np.repeat(np.add.reduceat(jac, self.role_starts, 1) /
                         self.num_role_strats, self.num_role_strats, 1)
        return devs, jac

    def subgame(self, subgame_mask):
        aggfn = super().subgame(subgame_mask)
        return RoleAgfnGame(aggfn.role_names, aggfn.strat_names,
                            aggfn.num_role_players, aggfn.function_names,
                            aggfn._action_weights, aggfn._function_inputs,
                            aggfn._function_table)

    @utils.memoize
    def __hash__(self):
        return super().__hash__()


class SumAgfnGame(AgfnGame):
    """An Aggfn with functions as a result of total player counts"""

    def get_payoffs(self, profile):
        profile = np.asarray(profile, int)
        function_inputs = profile.dot(self._function_inputs)
        function_outputs = self._function_table[np.arange(self.num_functions),
                                                function_inputs]
        payoffs = function_outputs.dot(self._action_weights)
        payoffs[profile == 0] = 0
        return payoffs

    def deviation_payoffs(self, mix, *, jacobian=False):
        mix = np.asarray(mix, float)
        role_node_probs = np.minimum(
            np.add.reduceat(mix[:, None] * self._function_inputs,
                            self.role_starts), 1)[..., None]
        role_fft = np.ones((self.num_roles, self.num_functions,
                            self.num_players + 1), np.complex128)
        if jacobian:
            drole_fft = np.ones(
                (self.num_roles, self.num_roles, self.num_functions,
                 self.num_players + 1), np.complex128)

        for i, (num_play, probs, zp) in enumerate(zip(
                self.num_role_players, role_node_probs, self.zero_prob)):
            configs = np.arange(num_play + 1)
            role_probs = spt.binom.pmf(configs, num_play, probs)
            dev_role_probs = spt.binom.pmf(configs[:-1], num_play - 1, probs)
            fft = np.fft.fft(role_probs, self.num_players + 1)
            dev_fft = np.fft.fft(dev_role_probs, self.num_players + 1)

            role_fft[:i] *= fft
            role_fft[i] *= dev_fft
            role_fft[i + 1:] *= fft

            if not jacobian:
                continue

            der = configs / (probs + zp) - configs[::-1] / (1 - probs + zp)
            dev_der = (configs[:-1] / (probs + zp) - configs[-2::-1] /
                       (1 - probs + zp))
            dfft = np.fft.fft(der * role_probs, self.num_players + 1)
            ddev_fft = np.fft.fft(dev_der * dev_role_probs,
                                  self.num_players + 1)

            drole_fft[:i, :i] *= fft
            drole_fft[i, :i] *= dev_fft
            drole_fft[i + 1:, :i] *= fft
            drole_fft[:i, i] *= dfft
            drole_fft[i, i] *= ddev_fft
            drole_fft[i + 1:, i] *= dfft
            drole_fft[:i, i + 1:] *= fft
            drole_fft[i, i + 1:] *= dev_fft
            drole_fft[i + 1:, i + 1:] *= fft

        dev_probs = utils.simplex_project(
            np.fft.ifft(role_fft).real).repeat(self.num_role_strats, 0)
        dev_probs[self._function_inputs] = np.roll(
            dev_probs[self._function_inputs], 1, 1)
        devs = np.sum(np.sum(dev_probs * self._function_table, -1) *
                      self._action_weights.T, -1)

        if not jacobian:
            return devs

        dev_deriv = np.rollaxis(
            np.fft.ifft(drole_fft).real, 2, 1).repeat(self.num_role_strats, 0)
        dev_deriv[self._function_inputs] = np.roll(
            dev_deriv[self._function_inputs], 1, 2)

        dev_values = dev_deriv * self._function_table[:, None]
        jac = np.sum(
            np.sum(dev_values.sum(-1)[:, None] * self._dinputs, -1) *
            self._action_weights.T[:, None], -1)
        jac -= np.repeat(np.add.reduceat(jac, self.role_starts, 1) /
                         self.num_role_strats, self.num_role_strats, 1)
        return devs, jac

    def subgame(self, subgame_mask):
        aggfn = super().subgame(subgame_mask)
        return SumAgfnGame(aggfn.role_names, aggfn.strat_names,
                           aggfn.num_role_players, aggfn.function_names,
                           aggfn._action_weights, aggfn._function_inputs,
                           aggfn._function_table)

    @utils.memoize
    def __hash__(self):
        return super().__hash__()


def aggfn(num_role_players, num_role_strats, action_weights, function_inputs,
          function_table):
    """Create an Aggfn with default names

    Parameters
    ----------
    num_role_players : ndarray
        The number of players per role.
    num_role_strats : ndarray
        The number of strategies per role.
    action_weights : ndarray, float
        The action weights.
    function_inputs : ndarray, bool
        The input mask for each function.
    function_table : ndarray, float
        The function value relative to number of incoming edges.
    """
    return aggfn_replace(rsgame.emptygame(num_role_players, num_role_strats),
                         action_weights, function_inputs, function_table)


def aggfn_names(role_names, num_role_players, strat_names, function_names,
                action_weights, function_inputs, function_table):
    """Create an Aggfn with specified names

    Parameters
    ----------
    role_names : [str]
        The name of each role.
    num_role_players : ndarray
        The number of players for each role.
    strat_names : [[str]]
        The name of each strategy for each role.
    function_names : [str]
        The name of each function.
    action_weights : ndarray
        The mapping of each function to the strategy weight for a player.
    function_inpits : ndarray
        The mask indicating which strategies are inputs to which function.
    """
    return aggfn_names_replace(
        rsgame.emptygame_names(role_names, num_role_players, strat_names),
        function_names, action_weights, function_inputs, function_table)


# TODO Make aggfn_copy method that will clone the aggfn game if it is one,
# else, it will regress on profiles to compute one.


def aggfn_replace(copy_game, action_weights, function_inputs, function_table):
    """Replace an existing game with default function names

    Parameters
    ----------
    copy_game : RsGame
        The game to take game structure from.
    action_weights : ndarray-like
        The weights of each function to player payoffs.
    function_inputs : ndarray-like
        The mask of each strategy to function.
    function_table : ndarray-like
        The lookup table of number of incoming edges to function value.
    """
    # XXX This does a little checking of names that we know are valid, but it's
    # much simplier, so it seems okay.
    function_names = tuple(utils.prefix_strings('f', len(action_weights)))
    return aggfn_names_replace(copy_game, function_names, action_weights,
                               function_inputs, function_table)


def aggfn_names_replace(copy_game, function_names, action_weights,
                        function_inputs, function_table):
    """Replace an existing game with an Aggfn

    Parameters
    ----------
    copy_game : RsGame
        The game to take game structure from.
    function_names : [str]
        The name of each function.
    action_weights : ndarray-like
        The weights of each function to player payoffs.
    function_inputs : ndarray-like
        The mask of each strategy to function.
    function_table : ndarray-like
        The lookup table of number of incoming edges to function value.
    """
    function_names = tuple(function_names)
    action_weights = np.asarray(action_weights, float)
    function_inputs = np.asarray(function_inputs, bool)
    function_table = np.asarray(function_table, float)
    num_funcs = len(function_names)

    assert function_names, \
        "must have at least one function"
    assert (action_weights.shape == (num_funcs, copy_game.num_strats)), \
        "action_weights must have shape (num_functions, num_strats)"
    assert (function_inputs.shape == (copy_game.num_strats, num_funcs)), \
        "function_inputs must have shape (num_strats, num_functions)"
    assert ((function_table.shape == (num_funcs, copy_game.num_players + 1))
            or (function_table.shape == (num_funcs,) +
                tuple(copy_game.num_role_players + 1))), \
        "function_table must have shape (num_functions, num_players + 1) or " \
        "(num_functions, ... num_role_players + 1)"
    assert not np.isclose(action_weights, 0).all(1).any(), \
        "a function can't have actions weights of all zero"
    assert all(isinstance(f, str) for f in function_names), \
        "all function names must be strs"
    assert utils.is_sorted(function_names, strict=True), \
        "function_names must be sorted"
    # For symmetric games where the function table has dimension 2, then Role
    # and Sum are identical, but Role's math is a little simpler so we default
    # to that.
    if not copy_game.is_symmetric() and function_table.ndim == 2:
        return SumAgfnGame(copy_game.role_names, copy_game.strat_names,
                           copy_game.num_role_players, function_names,
                           action_weights, function_inputs, function_table)
    else:
        return RoleAgfnGame(copy_game.role_names, copy_game.strat_names,
                            copy_game.num_role_players, function_names,
                            action_weights, function_inputs, function_table)


def aggfn_funcs(num_role_players, num_role_strats, action_weights,
                function_inputs, functions):
    """Construct and Aggfn with functions

    This is generally less efficient than just constructing the function table
    using vectorized operations or an existing function table.

    Parameters
    ----------
    num_role_players : ndarray
        The number of players per role.
    num_role_strats : ndarray
        The number of strategies per role.
    action_weights : ndarray, float
        The action weights.
    function_inputs : ndarray, bool
        The input mask for each function.
    functions : [(n, ...) -> float]
        List of functions that either map total player activations or player
        per role activations to a single value. The number of ordered arguments
        will be inferred from each function.
    """
    assert functions, "must have at least one function"
    num_functions = len(functions)
    num_params = _num_args_safe(functions[0])
    assert all(num_params == _num_args_safe(f) for f in functions), \
        "all functions must take the same number of parameters"

    base = rsgame.emptygame(num_role_players, num_role_strats)
    if num_params == 1:
        # sum format
        function_table = np.empty(
            (num_functions, base.num_players + 1), float)
        for func, tab in zip(functions, function_table):
            for p in range(base.num_players + 1):
                tab[p] = func(p)
    else:
        # role format
        assert num_params == base.num_roles
        function_table = np.empty(
            (num_functions,) + tuple(base.num_role_players + 1),
            float)
        for func, tab in zip(functions, function_table):
            for p in itertools.product(*map(range, base.num_role_players + 1)):
                tab[p] = func(*p)

    return aggfn_replace(base, action_weights, function_inputs, function_table)


def aggfn_json(json):
    """Read an Aggfn from json

    Json versions of the game will generally have 'type': 'aggfn...' in them,
    but as long as the proper fields exist, this will succeed."""
    base = rsgame.emptygame_json(json)
    function_names = sorted(json['function_tables'])
    findex = {f: i for i, f in enumerate(function_names)}
    num_functions = len(function_names)

    function_inputs = np.zeros(
        (base.num_strats, num_functions), bool)
    for func, roles in json['function_inputs'].items():
        fi = findex[func]
        for role, strats in roles.items():
            for strat in strats:
                rsi = base.role_strat_index(role, strat)
                function_inputs[rsi, fi] = True

    action_weights = np.zeros(
        (num_functions, base.num_strats), float)
    for role, strats in json['action_weights'].items():
        for strat, funcs in strats.items():
            rsi = base.role_strat_index(role, strat)
            for func, val in funcs.items():
                action_weights[findex[func], rsi] = val

    function_list = [None] * num_functions
    for func, table in json['function_tables'].items():
        function_list[findex[func]] = np.asarray(table, float)

    return aggfn_names_replace(
        base, function_names, action_weights, function_inputs,
        np.asarray(function_list, float))


def _num_args(func):
    """Helper to get the number of args of a function"""
    return sum(1 for p in inspect.signature(func).parameters.values()
               if p.default is p.empty)


def _num_args_safe(func):
    """Get the number of arguments of a function or function object"""
    try:
        return _num_args(func)
    except AttributeError:  # pragma: no cover
        return _num_args(func.__call__)
