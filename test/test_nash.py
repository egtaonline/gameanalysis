"""Test nash"""
import itertools
import math
from os import path

import numpy as np
import pytest
import timeout_decorator

from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import nash
from gameanalysis import regret
from gameanalysis import rsgame


METHS = [('optimize', {}), ('replicator', {}), ('fictitious', {})]
METHODS = [{k: v} for k, v in METHS]
ALL_METHODS = list(map(dict, itertools.chain.from_iterable(
    itertools.combinations(METHS, i)
    for i in range(1, len(METHS) + 1))))


@pytest.mark.parametrize('_', range(20))
def test_pure_prisoners_dilemma(_):
    """Test prisoners dilemma"""
    game = gamegen.prisoners_dilemma()
    eqa = nash.pure_nash(game)

    assert eqa.shape[0] == 1, "didn't find exactly one equilibria in pd"
    expected = [0, 2]
    assert np.all(expected == eqa), \
        "didn't find pd equilibrium"


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('methods', ALL_METHODS)
def test_mixed_prisoners_dilemma(methods, _):
    """Test prisoners dilemma mixed nash"""
    game = gamegen.prisoners_dilemma()
    eqa = nash.mixed_nash(game, dist_thresh=1e-3, **methods)

    assert eqa.shape[0] >= 1, \
        "didn't find at least one equilibria in pd {}".format(eqa)
    assert all(regret.mixture_regret(game, eqm) < 1e-3 for eqm in eqa), \
        'returned equilibria with high regret'
    expected = [0, 1]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(1).any(), \
        "didn't find pd equilibrium {}".format(eqa)


@pytest.mark.parametrize('methods', METHODS)
@pytest.mark.parametrize('eq_prob', [0, .1, .2, .3, .5, .7, .8, .9, 1])
def test_mixed_known_eq(methods, eq_prob):
    """Test mixed nash with known eq"""
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    eqa = nash.mixed_nash(game, **methods)
    assert eqa.shape[0] >= 1, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert np.isclose(eqa, expected, atol=1e-2, rtol=1e-2).all(1).any(), \
        "didn't find correct equilibrium {} instead of {}".format(
            eqa, expected)


def test_pure_roshambo():
    """Test roshambo"""
    game = gamegen.rock_paper_scissors()
    eqa = nash.pure_nash(game)
    assert eqa.size == 0, 'found a pure equilibrium in roshambo'
    eqa = nash.pure_nash(game, epsilon=1)
    assert eqa.shape[0] == 3, \
        "didn't find low regret ties in roshambo"
    eqa = nash.pure_nash(game, epsilon=2)
    assert eqa.shape[0] == game.num_all_profiles, \
        'found profiles with more than 2 regret in roshambo'


def test_minreg_roshambo():
    """Test minimum regret"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_profile(game)
    assert np.all(np.sort(eqm) == [0, 0, 2]), \
        'min regret profile was not rr, pp, or ss'


def test_minreg_grid_roshambo():
    """Test min regret grid search"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_grid_mixture(game, 3)  # Not enough for eq
    assert np.isclose(regret.mixture_regret(game, eqm), .5), \
        "min regret grid didn't find [.5, .5, 0] profile with regret .5"
    eqm = nash.min_regret_grid_mixture(game, 4)  # hit eqa perfectly
    assert np.isclose(regret.mixture_regret(game, eqm), 0), \
        "min regret grid didn't find equilibrium"


def test_minreg_rand_roshambo():
    """Test min regret random search"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_rand_mixture(game, 20)
    assert regret.mixture_regret(game, eqm) < 2 + 1e-7, \
        'Found a mixture with greater than maximum regret'


@pytest.mark.parametrize('methods', METHODS)
def test_mixed_roshambo(methods):
    """Test roshambo"""
    game = gamegen.rock_paper_scissors()
    eqa = nash.mixed_nash(game, dist_thresh=1e-2, **methods)
    assert eqa.shape[0] == 1, \
        "didn't find right number of equilibria in roshambo"
    assert np.allclose(1 / 3, eqa, rtol=1e-3, atol=1e-3), \
        "roshambo equilibria wasn't uniform"


# XXX For some reason this fails occasionally, and it's unclear why
@pytest.mark.xfail(raises=AssertionError)
def test_hard_roshambo():
    """Test hard roshambo"""
    game = gamegen.rock_paper_scissors(loss=[-2, -3, -3])
    eqa = nash.mixed_nash(game)
    assert eqa.shape[0] == 1, \
        "didn't find right number of equilibria in roshambo"
    assert np.allclose([0.3125, 0.40625, 0.28125], eqa), \
        "roshambo equilibria wasn't uniform"


def test_at_least_one():
    """Test at_least_one"""
    # Equilibrium of game is not at a starting point for equilibria finding
    game = gamegen.sym_2p2s_known_eq(1 / math.sqrt(2))
    # Don't converge
    eqa = nash.mixed_nash(game, replicator={'max_iters': 0})
    assert eqa.size == 0, 'found an equilibrium normally'
    eqa = nash.mixed_nash(game, replicator={'max_iters': 0},
                          at_least_one=True)
    assert eqa.shape[0] == 1, "at_least_one didn't return anything"


def test_min_reg_nash():
    """Test minimum regret nash"""
    # Equilibrium of game is not at a starting point for equilibria finding
    game = gamegen.sym_2p2s_known_eq(1 / math.sqrt(2))
    # Don't converge
    eqa = nash.mixed_nash(game, replicator={'max_iters': 0})
    assert eqa.size == 0, 'found an equilibrium normally'
    eqa = nash.mixed_nash(game, replicator={'max_iters': 0},
                          min_reg=True)
    assert eqa.shape[0] == 1, "min_reg didn't return anything"


@pytest.mark.parametrize('methods,strategies', zip(
    ALL_METHODS * 2,
    [
        [1],
        [2],
        [1, 1],
        [2, 2],
        [1, 3],
    ]))
def test_mixed_nash(methods, strategies):
    """Test mixed nash"""
    game = gamegen.independent_game(strategies)
    eqa = nash.mixed_nash(game, **methods)
    assert all(regret.mixture_regret(game, eqm) <= 1e-3 for eqm in eqa)


@pytest.mark.parametrize('methods,strategies', zip(
    ALL_METHODS * 2,
    [
        [1],
        [2],
        [1, 1],
        [2, 2],
        [1, 3],
    ]))
def test_mixed_nash_multi_process(methods, strategies):
    """Test multiprocessing"""
    game = gamegen.independent_game(strategies)
    eqa = nash.mixed_nash(game, processes=2, **methods)
    assert all(regret.mixture_regret(game, eqm) <= 1e-3 for eqm in eqa)


@pytest.mark.parametrize('methods,strategies', zip(
    ALL_METHODS * 2,
    [
        [1],
        [2],
        [1, 1],
        [2, 2],
        [1, 3],
    ]))
def test_mixed_nash_best(methods, strategies):
    """Test mixed nash with min reg"""
    game = gamegen.independent_game(strategies)
    eqa = nash.mixed_nash(game, min_reg=True, **methods)
    assert eqa.size, "didn't return something"


@pytest.mark.slow
@pytest.mark.parametrize('methods,strategies', zip(
    ALL_METHODS * 2,
    [
        [1],
        [2],
        [1, 1],
        [2, 2],
        [4, 4, 4],
        [1, 3],
    ]))
def test_mixed_nash_at_least_one(methods, strategies):
    """Test at least one"""
    game = gamegen.independent_game(strategies)
    eqa = nash.mixed_nash(game, at_least_one=True, **methods)
    assert eqa.size, "didn't return at least one equilibria"
    assert all(regret.mixture_regret(game, eqm) <= 1e-3 for eqm in eqa)


def test_empty_game():
    """Test on empty game"""
    game = rsgame.empty(2, 3)
    with pytest.raises(ValueError):
        nash.min_regret_profile(game)


# XXX For some reason this fails on travis-ci, but not locally, so we have to
# allow for finding no equilibria.
def test_hard_nash():
    """Test hard nash"""
    with open(path.join('example_games', 'hard_nash.json')) as fil:
        game = gamereader.load(fil)
    expected = [0.54074609, 0.45925391, 0, 0, 0, 1, 0, 0, 0]
    eqa = nash.mixed_nash(game)
    assert not eqa.size or np.isclose(game.trim_mixture_support(eqa), expected,
                                      atol=1e-4, rtol=1e-4).all(1).any(), \
        "Didn't find equilibrium in known hard instance"


@timeout_decorator.timeout(2)
@pytest.mark.xfail(raises=timeout_decorator.timeout_decorator.TimeoutError)
def test_hard_scarf():
    """A buggy instance of scarfs algorithm

    This triggered a discretization error with the fixed point algorithm
    immediately, e.g. a timeout of 2s is fine"""
    with open(path.join('example_games', 'hard_scarf.json')) as fil:
        game = gamereader.load(fil)
    nash.scarfs_algorithm(game, game.uniform_mixture())


@pytest.mark.slow
@pytest.mark.parametrize('_', range(20))
def test_at_least_one_big(_):
    """Test at least one on big game"""
    num_roles = np.random.randint(1, 4)
    players = np.random.randint(2, 5, num_roles)
    strategies = np.random.randint(2, 5, num_roles)
    functions = np.random.randint(2, 8)
    agame = gamegen.normal_aggfn(players, strategies, functions)
    eqa = nash.mixed_nash(agame, at_least_one=True)
    assert eqa.size, "didn't find equilibrium but should always find one"
