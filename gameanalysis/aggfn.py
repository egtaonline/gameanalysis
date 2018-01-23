"""An action graph game with additive function nodes"""
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
    def __init__(self, role_names, strat_names, function_names,
                 num_role_players, action_weights, function_inputs,
                 function_table, offsets):
        super().__init__(role_names, strat_names, num_role_players)
        self.function_names = function_names
        self.num_functions = len(function_names)
        self.action_weights = action_weights
        self.action_weights.setflags(write=False)
        self.function_inputs = function_inputs
        self.function_inputs.setflags(write=False)
        self.function_table = function_table
        self.function_table.setflags(write=False)
        self.offsets = offsets
        self.offsets.setflags(write=False)

        # Pre-compute derivative info
        self._dinputs = np.zeros(
            (self.num_strats, self.num_functions, self.num_roles), bool)
        self._dinputs[np.arange(self.num_strats), :, self.role_indices] = (
            self.function_inputs)
        self._dinputs.setflags(write=False)

        self._function_index = {f: i for i, f in enumerate(function_names)}

        # Compute other bookmarking stuff
        self._basis = np.insert(
            np.cumprod(self.num_role_players[:0:-1] + 1)[::-1],
            self.num_roles - 1, 1)
        self._func_offset = (np.arange(self.num_functions) *
                             np.prod(self.num_role_players + 1))

    def function_index(self, func_name):
        """Get the index of a function by name"""
        return self._function_index[func_name]

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns a lower bound on the payoffs."""
        node_table = self.function_table.reshape((self.num_functions, -1))
        minima = node_table.min(1, keepdims=True)
        maxima = node_table.max(1, keepdims=True)
        eff_min = np.where(self.action_weights > 0, minima, maxima)
        mins = np.einsum(
            'ij,ij->j', eff_min, self.action_weights) + self.offsets
        mins.setflags(write=False)
        return mins.view()

    @utils.memoize
    def max_strat_payoffs(self):
        """Returns an upper bound on the payoffs."""
        node_table = self.function_table.reshape((self.num_functions, -1))
        minima = node_table.min(1, keepdims=True)
        maxima = node_table.max(1, keepdims=True)
        eff_max = np.where(self.action_weights > 0, maxima, minima)
        maxs = np.einsum(
            'ij,ij->j', eff_max, self.action_weights) + self.offsets
        maxs.setflags(write=False)
        return maxs.view()

    def get_payoffs(self, profile):
        """Returns an array of profile payoffs."""
        profile = np.asarray(profile, int)
        function_inputs = np.add.reduceat(
            profile[..., None, :] * self.function_inputs.T,
            self.role_starts, -1)
        inds = function_inputs.dot(self._basis) + self._func_offset
        function_outputs = self.function_table.ravel()[inds]
        payoffs = function_outputs.dot(self.action_weights) + self.offsets
        payoffs[profile == 0] = 0
        return payoffs

    # TODO override get_dev_payoffs to be more efficient, i.e. only compute the
    # dev payoff.

    def deviation_payoffs(self, mix, *, jacobian=False):
        """Get the deviation payoffs"""
        mix = np.asarray(mix, float)
        role_node_probs = np.minimum(
            np.add.reduceat(mix[:, None] * self.function_inputs,
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
                np.split(self.function_inputs, self.role_starts[1:], 0),
                np.split(dev_probs, self.role_starts[1:], 0))):
            rdev_probs[rinps] = np.roll(rdev_probs[rinps], 1, role + 1)
        dev_vals = np.reshape(dev_probs * self.function_table,
                              (self.num_strats, self.num_functions, -1))
        devs = (np.einsum('ijk,ji->i', dev_vals, self.action_weights) +
                self.offsets)

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
                np.split(self.function_inputs, self.role_starts[1:], 0),
                np.split(dev_deriv, self.role_starts[1:], 0))):
            rdev_deriv[rinps] = np.roll(rdev_deriv[rinps], 1, role + 2)

        dev_values = dev_probs[:, :, None] * \
            dev_deriv * self.function_table[:, None]
        dev_values.shape = (self.num_strats,
                            self.num_functions, self.num_roles, -1)
        jac = np.einsum('iklm,jkl,ki->ij', dev_values, self._dinputs,
                        self.action_weights)
        return devs, jac

    def normalize(self):
        """Return a normalized AgfnGame"""
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0)] = 1
        scale = scale.repeat(self.num_role_strats)
        offsets = (self.offsets - self.min_role_payoffs().repeat(
            self.num_role_strats)) / scale

        return AgfnGame(
            self.role_names, self.strat_names, self.function_names,
            self.num_role_players, self.action_weights / scale,
            self.function_inputs, self.function_table, offsets)

    def restrict(self, rest):
        rest = np.asarray(rest, bool)
        base = rsgame.emptygame_copy(self).restrict(rest)
        action_weights = self.action_weights[:, rest]
        func_mask = np.any(~np.isclose(action_weights, 0), 1)
        func_names = tuple(
            n for n, m in zip(self.function_names, func_mask) if m)
        return AgfnGame(
            base.role_names, base.strat_names, func_names,
            base.num_role_players, action_weights[func_mask],
            self.function_inputs[:, func_mask][rest],
            self.function_table[func_mask], self.offsets[rest])

    def to_json(self):
        res = super().to_json()

        res['function_inputs'] = {
            func: self.restriction_to_json(finp) for func, finp
            in zip(self.function_names, self.function_inputs.T)}

        res['action_weights'] = {
            func: self.payoff_to_json(ws) for func, ws
            in zip(self.function_names, self.action_weights)}

        # XXX This will fail if a role has the name "value", do we care?
        res['function_tables'] = {
            name: [dict(zip(self.role_names, (c.item() for c in counts)),
                        value=val)
                   for val, *counts in zip(
                           tab.ravel(), *np.indices(tab.shape).reshape(
                               self.num_roles, -1))
                   if val != 0]
            for name, tab in zip(self.function_names, self.function_table)}

        if not np.allclose(self.offsets, 0):
            res['offsets'] = self.payoff_to_json(self.offsets)

        res['type'] = 'aggfn.2'
        return res

    def __repr__(self):
        return '{old}, {nfuncs:d})'.format(
            old=super().__repr__()[:-1],
            nfuncs=self.num_functions)

    def __eq__(self, other):
        if not (super().__eq__(other) and
                self.function_names == other.function_names and
                self.function_table.shape == other.function_table.shape and
                np.allclose(self.offsets, other.offsets)):
            return False

        selfp = np.lexsort(
            self.function_table.reshape((self.num_functions, -1)).T)
        otherp = np.lexsort(
            other.function_table.reshape((other.num_functions, -1)).T)
        return (np.all(self.function_inputs[:, selfp]
                       == other.function_inputs[:, otherp]) and
                np.allclose(self.action_weights[selfp],
                            other.action_weights[otherp]) and
                np.allclose(self.function_table[selfp],
                            other.function_table[otherp]))

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_functions))


def aggfn(num_role_players, num_role_strats, action_weights, function_inputs,
          function_table, offsets=None):
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
    offsets : ndarray, float, optional
        A constant offset for each strategies payoff. Constant functions are
        not allowed in the function table as they are clutter, instead,
        constant functions can be specified here.
    """
    return aggfn_replace(rsgame.emptygame(num_role_players, num_role_strats),
                         action_weights, function_inputs, function_table,
                         offsets)


def aggfn_names(role_names, num_role_players, strat_names, function_names,
                action_weights, function_inputs, function_table, offsets=None):
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
    offsets : ndarray, float, optional
        A constant offset for each strategies payoff. Constant functions are
        not allowed in the function table as they are clutter, instead,
        constant functions can be specified here.
    """
    return aggfn_names_replace(
        rsgame.emptygame_names(role_names, num_role_players, strat_names),
        function_names, action_weights, function_inputs, function_table,
        offsets)


# TODO Make aggfn_copy method that will clone the aggfn game if it is one,
# else, it will regress on profiles to compute one.


def aggfn_replace(copy_game, action_weights, function_inputs, function_table,
                  offsets=None):
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
    offsets : ndarray, float, optional
        A constant offset for each strategies payoff. Constant functions are
        not allowed in the function table as they are clutter, instead,
        constant functions can be specified here.
    """
    if hasattr(copy_game, 'function_names'):
        function_names = copy_game.function_names
    else:
        function_names = tuple(utils.prefix_strings('f', len(action_weights)))
    return aggfn_names_replace(copy_game, function_names, action_weights,
                               function_inputs, function_table, offsets)


def aggfn_names_replace(copy_game, function_names, action_weights,
                        function_inputs, function_table, offsets=None):
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
    offsets : ndarray, float, optional
        A constant offset for each strategies payoff. Constant functions are
        not allowed in the function table as they are clutter, instead,
        constant functions can be specified here.
    """
    if offsets is None:
        offsets = np.zeros(copy_game.num_strats)

    function_names = tuple(function_names)
    action_weights = np.asarray(action_weights, float)
    function_inputs = np.asarray(function_inputs, bool)
    function_table = np.asarray(function_table, float)
    offsets = np.asarray(offsets, float)
    num_funcs = len(function_names)

    assert function_names, \
        "must have at least one function"
    assert (action_weights.shape == (num_funcs, copy_game.num_strats)), \
        "action_weights must have shape (num_functions, num_strats)"
    assert (function_inputs.shape == (copy_game.num_strats, num_funcs)), \
        "function_inputs must have shape (num_strats, num_functions)"
    assert not function_inputs.all(0).any(), \
        "can't have a function with input from every strategy"
    assert function_inputs.any(0).all(), \
        "every function must take input from at least one strategy"
    assert (function_table.shape == (num_funcs,) +
            tuple(copy_game.num_role_players + 1)), \
        "function_table must have shape (num_functions, ... num_role_players + 1)"  # noqa
    assert not np.isclose(
        function_table.reshape((num_funcs, -1))[:, 0, None],
        function_table.reshape((num_funcs, -1))).all(1).any(), \
        "a function can't be constant (all identical values)"
    assert not np.isclose(action_weights, 0).all(1).any(), \
        "a function can't have actions weights of all zero"
    assert all(isinstance(f, str) for f in function_names), \
        "all function names must be strs"
    assert utils.is_sorted(function_names, strict=True), \
        "function_names must be sorted"

    return AgfnGame(
        copy_game.role_names, copy_game.strat_names, function_names,
        copy_game.num_role_players, action_weights, function_inputs,
        function_table, offsets)


def aggfn_funcs(num_role_players, num_role_strats, action_weights,
                function_inputs, functions, offsets=None):
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
    functions : [(nr1, nr2, ...) -> float]
        List of functions that maps the player per role activations to a single
        value. The number of ordered arguments will be inferred from each
        function.
    """
    assert functions, "must have at least one function"
    num_functions = len(functions)

    base = rsgame.emptygame(num_role_players, num_role_strats)
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

    version = json.get('type', '.1').split('.', 1)[1]

    function_inputs = np.empty((base.num_strats, num_functions), bool)
    action_weights = np.empty((num_functions, base.num_strats))
    function_table = np.empty((num_functions,) +
                              tuple(base.num_role_players + 1))
    offsets = np.empty(base.num_strats)

    for func, inps in json['function_inputs'].items():
        base.restriction_from_json(inps, function_inputs[:, findex[func]],
                                   verify=False)

    base.payoff_from_json(json.get('offsets', {}), offsets)

    if version == '1':
        action_weights.fill(0)
        for role, strats in json['action_weights'].items():
            for strat, funcs in strats.items():
                rsi = base.role_strat_index(role, strat)
                for func, val in funcs.items():
                    action_weights[findex[func], rsi] = val

        for func, jtable in json['function_tables'].items():
            atable = np.asarray(jtable, float)
            if base.num_roles > 1 and atable.ndim == 1:
                # Convert old sum format to role format
                tab = function_table[findex[func]]
                inds = np.indices(tab.shape)
                tab[tuple(inds)] = atable[inds.sum(0)]
            else:
                function_table[findex[func]] = atable

        # Find constant functions and remove them. Constant functions either
        # have identical function values for every possible input, or are never
        # activated or always activated.
        flat_funcs = function_table.reshape(num_functions, -1)
        vals = flat_funcs[:, 0].copy()
        consts = np.isclose(vals[:, None], flat_funcs).all(1)
        consts |= ~function_inputs.any(0)
        always = function_inputs.all(0)
        vals[always] = flat_funcs[always, -1]
        consts |= always

        offsets += vals[consts].dot(action_weights[consts])
        function_inputs = function_inputs[:, ~consts]
        action_weights = action_weights[~consts]
        function_table = function_table[~consts]
        flat_funcs = flat_funcs[~consts]
        function_names = [n for n, c in zip(function_names, consts) if not c]

    elif version == '2':
        for func, weights in json['action_weights'].items():
            base.payoff_from_json(weights, action_weights[findex[func]])

        function_table.fill(0)
        for func, jtable in json['function_tables'].items():
            table = function_table[findex[func]]
            for elem in jtable:
                copy = elem.copy()
                value = copy.pop('value')
                table[tuple(int(i) for i in base.role_from_json(copy))] = value

    else:
        assert False, "unknown version \"{}\"".format(version)

    return aggfn_names_replace(
        base, function_names, action_weights, function_inputs, function_table,
        offsets)
