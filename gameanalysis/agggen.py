import numpy as np

from gameanalysis import aggfn
from gameanalysis import rsgame
from gameanalysis import utils


def random_poly_dist(coef_dist, degree_dist):
    """Generate a function table distribution from a polynomial

    Parameters
    ----------
    coef_dist : ndarray int -> ndarray float
        Distribution that maps an array of degrees for a polynomial to a
        coefficient of that degree, e.g. `lambda d: np.random.normal(0,
        10.0**(1 - d))`
    degree_dist : ndarray
        Probability that a polynomial will have a specific degree starting from
        zero. [0.1, 0.4, 0.5] means each function has a 10% chance of being
        constant, 40% chance of being linear, and a 50% chance of being
        quadratic.

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


def random_aggfn(num_players, num_strategies, num_functions,
                 input_dist=lambda s: utils.random_con_bitmask(.2, s),
                 weight_dist=lambda s: np.random.normal(0, 1, s) *
                 utils.random_con_bitmask(.2, s),
                 func_dist=lambda s: np.random.normal(0, 1, s),
                 by_role=False):
    """Generate a random AgfnGame

    Parameters
    ----------
    num_players : int or ndarray
    num_strategies : int or ndarray
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
    base = rsgame.basegame(num_players, num_strategies)
    weights = weight_dist((num_functions, base.num_role_strats))
    inputs = input_dist((base.num_role_strats, num_functions))
    shape = ((num_functions,) + tuple(base.num_players + 1)
             if by_role else (num_functions, base.num_all_players + 1))
    func_table = func_dist(shape)
    return aggfn.aggfn_copy(base, weights, inputs, func_table)


def congestion(num_players, num_facilities, num_required, degree=2,
               coef_dist=lambda d: -np.random.exponential(10. ** (1 - d))):
    """Generate a congestion game

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
    table_dist = random_poly_dist(coef_dist, np.insert(np.zeros(degree), 2, 1))
    functions = table_dist((num_facilities, num_players + 1))
    return aggfn.aggfn(num_players, function_inputs.shape[0],
                       function_inputs.T, function_inputs, functions)


def local_effect(num_players, num_strategies, edge_prob=.2,
                 self_dist=random_poly_dist(
                     lambda d: -np.random.exponential(10. ** (1 - d)), [0, 1]),
                 other_dist=random_poly_dist(
                     lambda d: np.random.normal(0, 10. ** (-d)), [0, 0, 1])):
    """Generate a local effect game

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


def serializer(game):
    """Generate a random serializer from an AgfnGame"""
    role_names = ['all'] if game.is_symmetric(
    ) else utils.prefix_strings('r', game.num_roles)
    strat_names = [utils.prefix_strings('s', s) for s in game.num_strategies]
    function_names = utils.prefix_strings('f', game.num_functions)
    return aggfn.aggfnserializer(role_names, strat_names, function_names)


def function_serializer(game):
    """Generate a random serializer from an AgfnGame

    Generates strategy names that describe the fucntions they input to. Useful
    for congestion games"""
    role_names = ['all'] if game.is_symmetric(
    ) else utils.prefix_strings('r', game.num_roles)
    function_names = utils.prefix_strings('f', game.num_functions)
    strat_names = [['_'.join(f for f, i in zip(function_names, inp) if i)
                    for inp in role_inps]
                   for role_inps in game.role_split(game._function_inputs)]
    return aggfn.aggfnserializer(role_names, strat_names, function_names)
