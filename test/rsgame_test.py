import random
import math
import numpy as np

from gameanalysis import randgames, funcs
import test
from test import utils

TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps

# The lambda: 0 means that the payoffs are all zero, since they don't matter
SMALL_GAMES = [
    [randgames.symmetric_game(2, 2, lambda: 0)],
    [randgames.symmetric_game(2, 5, lambda: 0)],
    [randgames.symmetric_game(5, 2, lambda: 0)],
    [randgames.symmetric_game(5, 5, lambda: 0)],
    [randgames.independent_game(2, 2, lambda: 0)],
    [randgames.independent_game(2, 5, lambda: 0)],
    [randgames.independent_game(5, 2, lambda: 0)],
    [randgames.independent_game(5, 5, lambda: 0)],
    [randgames.role_symmetric_game(2, [1, 2], [2, 1], lambda: 0)],
    [randgames.role_symmetric_game(2, 2, 2, lambda: 0)],
    [randgames.role_symmetric_game(2, 2, 5, lambda: 0)],
    [randgames.role_symmetric_game(2, 5, 2, lambda: 0)],
    [randgames.role_symmetric_game(2, 5, 5, lambda: 0)]
]


def generate_games():
    '''Returns a generator for game testing'''
    for game in SMALL_GAMES:
        yield game

    if test._CONFIG['big_tests']:  # Big Games
        # First is the limit of approximate dev reps
        yield [randgames.symmetric_game(170, 2, lambda: 0)]
        yield [randgames.symmetric_game(1000, 2, lambda: 0)]
        yield [randgames.symmetric_game(5, 40, lambda: 0)]
        yield [randgames.symmetric_game(3, 160, lambda: 0)]
        yield [randgames.symmetric_game(50, 2, lambda: 0)]
        yield [randgames.symmetric_game(20, 5, lambda: 0)]
        yield [randgames.symmetric_game(90, 5, lambda: 0)]
        yield [randgames.role_symmetric_game(2, 2, 40, lambda: 0)]
        yield [randgames.symmetric_game(12, 12, lambda: 0)]


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


@utils.apply(generate_games())
def devreps_approx_test(game):
    approx = approx_dev_reps(game)
    exact = exact_dev_reps(game)
    diff = (approx - exact) / (exact + TINY)
    assert np.all(np.abs(diff) < EPS), \
        'dev reps were not close enough'


@utils.apply(generate_games())
def uniform_mixture_test(game):
    mix = game.uniform_mixture(as_array=True)
    # Check that it's a proper mixture
    assert np.allclose(mix.sum(1), 1), 'Uniform mixture wasn\'t a mixture'
    # Check that they're all the same
    masked = np.ma.masked_array(mix, mix == 0)
    assert np.allclose(np.diff(masked), 0), 'Uniform mixture wasn\'t uniform'


@utils.apply(generate_games())
def random_mixture_test(game):
    mix = game.random_mixture(as_array=True)
    assert np.allclose(mix.sum(1), 1), 'Random mixture wasn\'t a mixture'


@utils.apply(generate_games())
def biased_mixture_test(game):
    bias = 0.6
    mixes = game.biased_mixtures(as_array=True, bias=bias)
    saw_bias = np.zeros_like(game._mask, dtype=bool)
    count = 0
    for mix in mixes:
        count += 1
        saw_bias |= mix == bias

    num_strats = game._mask.sum(1)
    assert np.prod(num_strats[num_strats > 1] + 1) == count, \
        'Didn\'t generate the proper number of mixtures'
    assert np.all(
        saw_bias  # observed a bias
        | (~game._mask)  # couldn't have observed one
        | (game._mask.sum(1) == 1)[:, np.newaxis]  # Only one strat so can't bias
    ), 'Didn\'t bias every strategy'
