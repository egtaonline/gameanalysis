import numpy as np

from gameanalysis import aggfn
from gameanalysis import rsgame
from gameanalysis import utils


# TODO These parameters could be more intelligently set with player knowledge
def random_poly_dist(degree_dist,
                     coef_dist=lambda d: np.random.normal(0, 10. ** (1 - d))):
    """Generate a function table distribution from a polynomial

    Parameters
    ----------
    degree_dist : ndarray
        Probability that a polynomial will have a specific degree starting from
        zero. [0.1, 0.4, 0.5] means each function has a 10% chance of being
        constant, 40% chance of being linear, and a 50% chance of being
        quadratic.
    coef_dist : ndarray int -> ndarray float
        Distribution that maps an array of degrees for a polynomial to a
        coefficient of that degree, e.g. `lambda d: np.random.normal(0,
        10.0**(1 - d))`

    Notes
    -----
    Table distributions take a shape and return a random matrix of that shape,
    where the first axis is the number of functions, and the other axes are the
    number of players from each role."""
    degree_dist = np.asarray(degree_dist, float)
    assert np.all(degree_dist >= 0) and np.sum(degree_dist[:-1]) <= 1
    degrees = np.arange(degree_dist.size)
    degree_dist = np.insert(degree_dist[:-1], 0, 0).cumsum()

    def table_func(shape):
        funcs, *players = shape
        table = np.ones(shape, float)
        for d, play in enumerate(players):
            polys = coef_dist(np.broadcast_to(degrees, (funcs, degrees.size)))
            dmask = np.random.random(funcs)[:, None] > degree_dist
            values = np.dot(polys * dmask, np.arange(play) ** degrees[:, None])
            values.shape += (1,) * (len(players) - 1)
            table *= np.rollaxis(values, 1, d + 2)
        return table
    return table_func


def random_sin_dist(width_dist=np.random.random, coef_dist=np.random.random,
                    offset_dist=None):
    """Create a table function for random sinusoidal functions

    Functions will be `coef * sin(offset + width * num_players)`

    Parameters
    ----------
    width_dist : (shape) -> array, optional
        Distribution to generate function widths.
    coef_dist : (shape) -> array, optional
        Distribution to generate function coefficients.
    offset_dist : (shape) -> array, optional
        Distribution to generate function offsets. If unspecified, it will be
        the same as width_dist.
    """
    if offset_dist is None:
        offset_dist = width_dist

    def table_func(shape):
        funcs, *players = shape
        table = np.ones(shape, float)
        for d, play in enumerate(players):
            widths = width_dist((funcs, play))
            offsets = offset_dist((funcs, play))
            coefs = coef_dist((funcs,))[:, None]
            values = coefs * np.sin(widths * np.arange(play) + offsets)
            values.shape += (1,) * (len(players) - 1)
            table *= np.rollaxis(values, 1, d + 2)
        return table
    return table_func


def random_aggfn(num_role_players, num_role_strats, num_functions,
                 input_dist=lambda s: utils.random_con_bitmask(.2, s),
                 weight_dist=lambda s: np.random.normal(0, 1, s) *
                 utils.random_con_bitmask(.2, s),
                 func_dist=lambda s: np.random.normal(0, 1, s),
                 by_role=False):
    """Generate a random AgfnGame

    Parameters
    ----------
    num_role_players : int or ndarray
    num_role_strats : int or ndarray
    num_functions : int
    input_dist : f(shape) -> ndarray, bool, optional
        Function that takes a shape and redurns a boolean ndarray with the same
        shape representing the function inputs.
    weight_dist : f(shape) -> ndarray, float, optional
        Function that takes a shape and returns a float array with the same
        shape representing the action weights.
    func_dist : f(shape) ndarray, float, optional
        Function that takes a shape and returns a float array with the same
        shape representing the function table. To create polynomial functions
        use `random_poly_dist`.
    by_role : bool, optional
        Generate a role form AgfnGame. A role form game uses functions of the
        number of activations for each role, instead of just the total number
        of activations."""
    base = rsgame.emptygame(num_role_players, num_role_strats)
    weights = weight_dist((num_functions, base.num_strats))
    inputs = input_dist((base.num_strats, num_functions))
    shape = ((num_functions,) + tuple(base.num_role_players + 1)
             if by_role else (num_functions, base.num_players + 1))
    func_table = func_dist(shape)
    return aggfn.aggfn_replace(base, weights, inputs, func_table)


def congestion(num_players, num_facilities, num_required, degree=2,
               coef_dist=lambda d: -np.random.exponential(10. ** (1 - d))):
    """Generate a congestion game

    A congestion game is a symmetric game, where there are a given number of
    facilities, and each player must choose to use some amount of them. The
    payoff for each facility generally goes down as more players use it, and a
    players utility is the sum of the utilities for every facility.

    In this formulation, facility payoffs are random polynomials of the number
    of people using said facility.

    Parameters
    ----------
    num_players : int
    num_facilities : int
    num_required : int
    degree : int, optional
        Degree of payoff polynomials
    coef_dist : f(int) -> float, optional
        Numpy compatible function for generating random coefficients
        conditioned on degree. Leading degree must be negative to generate a
        true congestion game.
    """
    function_inputs = utils.acomb(num_facilities, num_required)
    table_dist = random_poly_dist(np.insert(np.zeros(degree), 2, 1), coef_dist)
    functions = table_dist((num_facilities, num_players + 1))
    facs = tuple(utils.prefix_strings('', num_facilities))
    strats = tuple('_'.join(facs[i] for i, m in enumerate(mask) if m)
                   for mask in function_inputs)
    return aggfn.aggfn_names(['all'], num_players, [strats], facs,
                             function_inputs.T, function_inputs, functions)


def local_effect(num_players, num_strategies, edge_prob=.2,
                 self_dist=random_poly_dist(
                     [0, 1], lambda d: -np.random.exponential(10. ** (1 - d))),
                 other_dist=random_poly_dist(
                     [0, 0, 1], lambda d: np.random.normal(0, 10. ** (-d)))):
    """Generate a local effect game

    In a local effect game, strategies are connected by a graph, and utilities
    are a function of the number of players playing our strategy and the number
    of players playing a neighboring strategy, hence local effect.

    In this formulation, payoffs for others playing our strategy are negative
    quadratics, and payoffs for playing other strategies are positive cubics.

    Parameters
    ----------
    num_players : int
    num_strategies : int
    edge_prob : float, optional
    self_dist : f(shape) -> ndarray, float, optional
        Table distribution for self functions, e.g. payoff effect for being on
        the same node.
    other_dist : f(shape) -> ndarray, float, optional
        Table distribution for functions affecting other strategies.
    """
    local_effect_graph = np.random.rand(
        num_strategies, num_strategies) < edge_prob
    np.fill_diagonal(local_effect_graph, False)
    num_functions = local_effect_graph.sum() + num_strategies

    action_weights = np.eye(num_functions, num_strategies, dtype=float)
    function_inputs = np.eye(num_strategies, num_functions, dtype=bool)
    in_act, out_act = local_effect_graph.nonzero()
    func_inds = np.arange(num_strategies, num_functions)
    function_inputs[in_act, func_inds] = True
    action_weights[func_inds, out_act] = 1

    function_table = np.empty((num_functions, num_players + 1), float)
    function_table[:num_strategies] = self_dist(
        (num_strategies, num_players + 1))
    function_table[num_strategies:] = self_dist(
        (num_functions - num_strategies, num_players + 1))
    return aggfn.aggfn(num_players, num_strategies, action_weights,
                       function_inputs, function_table)
