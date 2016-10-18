from itertools import combinations
from functools import partial
import numpy as np
from gameanalysis import AGGFN


default_degr_dist = partial(np.random.choice, [1,2,3,4], p=[.4,.3,.2,.1])
default_coef_dist = lambda d: np.random.normal(0, 10**(1-d))

def random_polynomial(degree=default_degr_dist, coef_distr=default_coef_dist):
    if hasattr(degree, "__call__"):
        degree = degree()
    assert isinstance(degree, int) or isinstance(degree, np.integer), \
                "degree must be an int or return one"
    return np.poly1d([coef_distr(d) for d in range(degree, -1, -1)])


congest_coef_dist = lambda d: -np.random.exponential(10**(1-d))

def congestion_game(num_players, num_facilities, num_required, degree=2,
                    coef_dist=congest_coef_dist):
    # Used for both AGGFNA and rsgame representations
    facilities = np.arange(num_facilities)
    strategies = list(combinations(facilities, num_required))
    num_strategies = len(strategies)
    function_inputs = np.zeros([num_strategies, num_facilities], dtype=bool)
    for s,strat in enumerate(strategies):
        for f in strat:
            function_inputs[s, f] = True
    congestion_functions = [random_polynomial(degree, coef_dist) for _ in
                            range(num_facilities)]

    agg = AGGFN.Sym_AGG_FNA(num_players, num_strategies, function_inputs.T,
                 function_inputs, congestion_functions)

    function_names = ["f" + str(f) for f in facilities]
    strategy_names = ["+".join("f" + str(s) for s in strat) for
                      strat in strategies]
    return agg, agg.to_json(strategy_names, function_names)


def random_bipartite_graph(source_set_size, dest_set_size, p=.2, min_inputs=1,
                           min_outputs=1):
    adj_matrix = np.random.binomial(1, p, source_set_size * dest_set_size)
    adj_matrix = adj_matrix.reshape(source_set_size, dest_set_size)

    assert min_inputs <= source_set_size, \
            "can't have more in edges than source nodes"
    assert min_outputs <= dest_set_size, \
            "can't have more out edges than dest nodes"

    for source in range(source_set_size):
        while sum(adj_matrix[source]) < min_outputs:
            adj_matrix[source, np.random.randint(dest_set_size)] = 1
    for dest in range(dest_set_size):
        while sum(adj_matrix[:,dest]) < min_inputs:
            adj_matrix[np.random.randint(source_set_size), dest] = 1

    return adj_matrix


def randomize_weights(adjacency_matrix, weight_distr):
    weight_matrix = [weight_distr() if e else 0 for e in adjacency_matrix.flat]
    return np.array(weight_matrix).reshape(adjacency_matrix.shape)


def random_AGGFNA(num_players, num_strategies, num_funcs,
                  weight_distr=partial(np.random.normal, 0, 1),
                  func_distr=random_polynomial,
                  act_edge_distr=random_bipartite_graph,
                  func_edge_distr=random_bipartite_graph):
    if hasattr(num_players, "__call__"):
        num_players = num_players()
    assert isinstance(num_players, int) or isinstance(num_players, \
                np.integer), "num_players must be an int or return one"
    if hasattr(num_strategies, "__call__"):
        num_strategies = num_strategies()
    assert isinstance(num_strategies, int) or isinstance(num_strategies, \
                np.integer), "num_strategies must be an int or return one"
    if hasattr(num_funcs, "__call__"):
        num_funcs = num_funcs()
    assert isinstance(num_funcs, int) or isinstance(num_funcs, np.integer), \
                "num_players must be an int or return one"
    node_functions = [func_distr() for _ in range(num_funcs)]
    action_edges = act_edge_distr(num_funcs, num_strategies)
    action_weights = randomize_weights(action_edges, weight_distr)
    function_inputs = func_edge_distr(num_strategies, num_funcs)
    return AGGFN.Sym_AGG_FNA(num_players, num_strategies, action_weights,
                             function_inputs, node_functions)
