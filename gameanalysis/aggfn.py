"""An action graph game with additive function nodes"""
import itertools

import numpy as np
import scipy.stats as spt

from gameanalysis import rsgame
from gameanalysis import utils


class _AgfnGame(rsgame._CompleteGame): # pylint: disable=too-many-instance-attributes,protected-access
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

    def __init__( # pylint: disable=too-many-arguments
            self, role_names, strat_names, num_role_players, action_weights,
            function_inputs, function_table, offsets):
        super().__init__(role_names, strat_names, num_role_players)
        self.num_functions, *_ = function_table.shape
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

        # Compute other bookmarking stuff
        self._basis = np.insert(
            np.cumprod(self.num_role_players[:0:-1] + 1)[::-1],
            self.num_roles - 1, 1)
        self._func_offset = (np.arange(self.num_functions) *
                             np.prod(self.num_role_players + 1))

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

    def get_payoffs(self, profiles):
        """Returns an array of profile payoffs."""
        profiles = np.asarray(profiles, int)
        function_inputs = np.add.reduceat(
            profiles[..., None, :] * self.function_inputs.T,
            self.role_starts, -1)
        inds = function_inputs.dot(self._basis) + self._func_offset
        function_outputs = self.function_table.ravel()[inds]
        payoffs = function_outputs.dot(self.action_weights) + self.offsets
        payoffs[profiles == 0] = 0
        return payoffs

    # TODO override get_dev_payoffs to be more efficient, i.e. only compute the
    # dev payoff.

    def deviation_payoffs(self, mixture, *, jacobian=False, **_): # pylint: disable=too-many-locals
        """Get the deviation payoffs"""
        mixture = np.asarray(mixture, float)
        role_node_probs = np.minimum(
            np.add.reduceat(mixture[:, None] * self.function_inputs,
                            self.role_starts), 1)[..., None]
        table_probs = np.ones(
            (self.num_roles, self.num_functions) +
            tuple(self.num_role_players + 1),
            float)
        for i, (num_play, probs) in enumerate(zip(self.num_role_players,
                                                  role_node_probs)):
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
        for i, (num_play, probs, zprob) in enumerate(zip(
                self.num_role_players, role_node_probs, self.zero_prob)):
            # TODO This zprob threshold causes large errors in the jacobian
            # when we look at sparse mixtures. This should probably be
            # addressed, but it's unclear how without making this significantly
            # slower.
            configs = np.arange(num_play + 1)
            der = (configs / (probs + zprob) -
                   configs[::-1] / (1 - probs + zprob))
            dev_der = np.insert(
                configs[:-1] / (probs + zprob) -
                configs[-2::-1] / (1 - probs + zprob), num_play, 0, 1)

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

    def _add_constant(self, constant):
        off = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        return _AgfnGame(
            self.role_names, self.strat_names, self.num_role_players,
            self.action_weights, self.function_inputs, self.function_table,
            self.offsets + off)

    def _multiply_constant(self, constant):
        mul = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        return _AgfnGame(
            self.role_names, self.strat_names, self.num_role_players,
            self.action_weights * mul, self.function_inputs,
            self.function_table, self.offsets * mul)

    def _add_game(self, othr):
        try:
            return _AgfnGame(
                self.role_names, self.strat_names, self.num_role_players,
                np.concatenate([self.action_weights, othr.action_weights]),
                np.concatenate([self.function_inputs, othr.function_inputs],
                               1),
                np.concatenate([self.function_table, othr.function_table]),
                self.offsets + othr.offsets)
        except AttributeError:
            return NotImplemented

    def restrict(self, restriction):
        restriction = np.asarray(restriction, bool)
        base = rsgame.empty_copy(self).restrict(restriction)
        action_weights = self.action_weights[:, restriction]
        func_mask = np.any(~np.isclose(action_weights, 0), 1)
        return _AgfnGame(
            base.role_names, base.strat_names, base.num_role_players,
            action_weights[func_mask],
            self.function_inputs[:, func_mask][restriction],
            self.function_table[func_mask], self.offsets[restriction])

    def to_json(self):
        res = super().to_json()

        res['function_inputs'] = [
            self.restriction_to_json(finp) for finp in self.function_inputs.T]

        res['action_weights'] = [
            self.payoff_to_json(ws) for ws in self.action_weights]

        # XXX This will fail if a role has the name 'value', do we care?
        res['function_tables'] = [
            [dict(zip(self.role_names, (c.item() for c in counts)),
                  value=val)
             for val, *counts in zip(
                 tab.ravel(),
                 *np.indices(tab.shape).reshape(self.num_roles, -1))
             if val != 0]
            for tab in self.function_table]

        if not np.allclose(self.offsets, 0):
            res['offsets'] = self.payoff_to_json(self.offsets)

        res['type'] = 'aggfn.3'
        return res

    def __repr__(self):
        return '{old}, {nfuncs:d})'.format(
            old=super().__repr__()[:-1],
            nfuncs=self.num_functions)

    def __eq__(self, othr):
        return (super().__eq__(othr) and
                self.num_functions == othr.num_functions and
                np.allclose(self.offsets, othr.offsets) and
                utils.allclose_perm(
                    np.concatenate(
                        [self.action_weights, self.function_inputs.T,
                         self.function_table.reshape(self.num_functions, -1)],
                        1),
                    np.concatenate(
                        [othr.action_weights, othr.function_inputs.T,
                         othr.function_table.reshape(othr.num_functions, -1)],
                        1)))

    @utils.memoize
    def __hash__(self):
        return hash((
            super().__hash__(),
            np.sort(utils.axis_to_elem(self.function_inputs.T)).tobytes()))


def aggfn( # pylint: disable=too-many-arguments
        num_role_players, num_role_strats, action_weights, function_inputs,
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
    return aggfn_replace(
        rsgame.empty(num_role_players, num_role_strats), action_weights,
        function_inputs, function_table, offsets)


def aggfn_names( # pylint: disable=too-many-arguments
        role_names, num_role_players, strat_names, action_weights,
        function_inputs, function_table, offsets=None):
    """Create an Aggfn with specified names

    Parameters
    ----------
    role_names : [str]
        The name of each role.
    num_role_players : ndarray
        The number of players for each role.
    strat_names : [[str]]
        The name of each strategy for each role.
    action_weights : ndarray
        The mapping of each function to the strategy weight for a player.
    function_inpits : ndarray
        The mask indicating which strategies are inputs to which function.
    offsets : ndarray, float, optional
        A constant offset for each strategies payoff. Constant functions are
        not allowed in the function table as they are clutter, instead,
        constant functions can be specified here.
    """
    return aggfn_replace(
        rsgame.empty_names(role_names, num_role_players, strat_names),
        action_weights, function_inputs, function_table, offsets)


# TODO Make aggfn_copy method that will clone the aggfn game if it is one,
# else, it will regress on profiles to compute one.


def aggfn_replace(copy_game, action_weights, function_inputs, function_table,
                  offsets=None):
    """Replace an existing game with an Aggfn

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
    if offsets is None:
        offsets = np.zeros(copy_game.num_strats)

    action_weights = np.asarray(action_weights, float)
    function_inputs = np.asarray(function_inputs, bool)
    function_table = np.asarray(function_table, float)
    offsets = np.asarray(offsets, float)
    num_funcs, *one_plays = function_table.shape

    utils.check(num_funcs > 0, 'must have at least one function')
    utils.check(
        action_weights.shape == (num_funcs, copy_game.num_strats),
        'action_weights must have shape (num_functions, num_strats) but got '
        '{}', action_weights.shape)
    utils.check(
        function_inputs.shape == (copy_game.num_strats, num_funcs),
        'function_inputs must have shape (num_strats, num_functions) but got '
        '{}', function_inputs.shape)
    utils.check(
        not function_inputs.all(0).any(),
        "can't have a function with input from every strategy")
    utils.check(
        function_inputs.any(0).all(),
        'every function must take input from at least one strategy')
    utils.check(
        one_plays == list(copy_game.num_role_players + 1),
        'function_table must have shape '
        '(num_functions, ... num_role_players + 1) but got {}',
        function_table.shape)
    utils.check(
        not np.isclose(
            function_table.reshape((num_funcs, -1))[:, 0, None],
            function_table.reshape((num_funcs, -1))).all(1).any(),
        "a function can't be constant (all identical values)")
    utils.check(
        not np.isclose(action_weights, 0).all(1).any(),
        "a function can't have actions weights of all zero")
    utils.check(
        offsets.shape == (copy_game.num_strats,),
        'offsets must have shape (num_strats,) but got {}', offsets.shape)

    return _AgfnGame(
        copy_game.role_names, copy_game.strat_names,
        copy_game.num_role_players, action_weights, function_inputs,
        function_table, offsets)


def aggfn_funcs( # pylint: disable=too-many-arguments
        num_role_players, num_role_strats, action_weights, function_inputs,
        functions, offsets=None):
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
    utils.check(functions, 'must have at least one function')
    num_functions = len(functions)

    base = rsgame.empty(num_role_players, num_role_strats)
    function_table = np.empty(
        (num_functions,) + tuple(base.num_role_players + 1),
        float)
    for func, tab in zip(functions, function_table):
        for play in itertools.product(*map(range, base.num_role_players + 1)):
            tab[play] = func(*play)

    return aggfn_replace(
        base, action_weights, function_inputs, function_table, offsets)


def aggfn_json(json): # pylint: disable=too-many-locals
    """Read an Aggfn from json

    Json versions of the game will generally have 'type': 'aggfn...' in them,
    but as long as the proper fields exist, this will succeed."""
    base = rsgame.empty_json(json)

    _, version = json.get('type', '.3').split('.', 1)
    utils.check(
        version == '3', 'parsing versions below 3 is currently unsupported')

    num_functions = len(json['function_tables'])
    function_inputs = np.empty((base.num_strats, num_functions), bool)
    action_weights = np.empty((num_functions, base.num_strats))
    function_table = np.empty(
        (num_functions,) + tuple(base.num_role_players + 1))
    offsets = np.empty(base.num_strats)

    base.payoff_from_json(json.get('offsets', {}), offsets)

    for inps, jinps in zip(function_inputs.T, json['function_inputs']):
        base.restriction_from_json(jinps, inps, verify=False)

    for weights, jweights in zip(action_weights, json['action_weights']):
        base.payoff_from_json(jweights, weights)

    function_table.fill(0)
    for table, jtable in zip(function_table, json['function_tables']):
        for elem in jtable:
            copy = elem.copy()
            value = copy.pop('value')
            table[tuple(int(i) for i in base.role_from_json(copy))] = value

    return aggfn_replace(
        base, action_weights, function_inputs, function_table, offsets)
