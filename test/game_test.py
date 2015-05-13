import random
import math
import numpy as np

from gameanalysis import randgames, funcs
from test import utils

TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps


def random_game(max_players, max_strategies):
    players = random.randint(2, max_players)
    strategies = random.randint(3, max_strategies)
    game_type = random.choice((randgames.independent_game,
                               randgames.symmetric_game))
    return game_type(players, strategies)


def approx_dev_reps(game):
    if game._dev_reps.dtype == object:
        return game._dev_reps
    approx = np.array(np.round(game._dev_reps), dtype=object)
    view = approx.ravel()
    for i, x in enumerate(view):
        view[i] = int(x)
    return approx


def exact_dev_reps(game):
    '''Uses python ints to compute dev reps. Much slower'''
    counts = game._counts
    dev_reps = np.empty_like(counts, dtype=object)
    strat_counts = list(game.players.values())
    fcount = [math.factorial(x) for x in strat_counts]
    for dev_prof, count_prof in zip(dev_reps, counts):
        total = funcs.prod(fc // funcs.prod(math.factorial(x) for x in cs)
                           for fc, cs in zip(fcount, count_prof))
        for dev_role, counts_role, strat_count \
                in zip(dev_prof, count_prof, strat_counts):
            for s, count in enumerate(counts_role):
                dev_role[s] = total * int(count) // strat_count
    return dev_reps


# The lambda: 0 means that the payoffs are all zero, since they don't matter
@utils.apply([x()] for x in (
    lambda: randgames.symmetric_game(2, 2, lambda: 0),
    lambda: randgames.symmetric_game(2, 5, lambda: 0),
    lambda: randgames.symmetric_game(5, 2, lambda: 0),
    lambda: randgames.symmetric_game(5, 5, lambda: 0),
    lambda: randgames.independent_game(2, 2, lambda: 0),
    lambda: randgames.independent_game(2, 5, lambda: 0),
    lambda: randgames.independent_game(5, 2, lambda: 0),
    lambda: randgames.independent_game(5, 5, lambda: 0),
    lambda: randgames.role_symmetric_game(2, 2, 2, lambda: 0),
    lambda: randgames.role_symmetric_game(2, 2, 5, lambda: 0),
    lambda: randgames.role_symmetric_game(2, 5, 2, lambda: 0),
    lambda: randgames.role_symmetric_game(2, 5, 5, lambda: 0),
    # Big Games
    lambda: randgames.symmetric_game(5, 40, lambda: 0),
    lambda: randgames.symmetric_game(3, 160, lambda: 0),
    lambda: randgames.symmetric_game(50, 2, lambda: 0),
    lambda: randgames.symmetric_game(20, 5, lambda: 0),
    # Limit of approximate dev reps
    lambda: randgames.symmetric_game(170, 2, lambda: 0),
    lambda: randgames.symmetric_game(1000, 2, lambda: 0),
    lambda: randgames.symmetric_game(90, 5, lambda: 0),
    lambda: randgames.role_symmetric_game(2, 2, 40, lambda: 0),
    lambda: randgames.symmetric_game(12, 12, lambda: 0)
))
def devreps_approx_test(game):
    approx = approx_dev_reps(game)
    exact = exact_dev_reps(game)
    diff = (approx - exact) / (exact + TINY)
    assert np.all(np.abs(diff) < EPS), \
        'dev reps were not close enough'
