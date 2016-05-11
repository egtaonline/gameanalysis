import itertools
import json
import math

import numpy as np
from nose import tools

from gameanalysis import gamegen
from gameanalysis import nash
from gameanalysis import regret
from gameanalysis import rsgame
from test import testutils


METHODS = [('optimize', {}), ('replicator', {})]
ALL_METHODS = list(map(dict, itertools.chain.from_iterable(
    itertools.combinations(METHODS, i)
    for i in range(1, len(METHODS) + 1))))


@testutils.apply(repeat=20)
def pure_prisoners_dilemma_test():
    game = gamegen.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    eqa = list(nash.pure_nash(game, as_array=True))

    assert len(eqa) == 1, "didn't find exactly one equilibria in pd"
    expected = np.array([0, 2])
    assert np.all(expected == eqa[0]), \
        "didn't find pd equilibrium"


@testutils.apply(zip(ALL_METHODS), repeat=20)
def mixed_prisoners_dilemma_test(methods):
    game = gamegen.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    eqa = list(nash.mixed_nash(game, dist_thresh=5e-2, as_array=True,
                               processes=1, **methods))

    assert len(eqa) >= 1, \
        "didn't find at least one equilibria in pd {}".format(eqa)
    assert all(regret.mixture_regret(game, eqm) < 1e-3 for eqm in eqa), \
        "returned equilibria with high regret"
    expected = [0., 1.]
    assert any(np.allclose(eqm, expected, atol=1e-4, rtol=1e-4)
               for eqm in eqa), \
        "didn't find pd equilibrium {}".format(eqa)


@testutils.apply(itertools.product(ALL_METHODS, (p/10 for p in range(11))))
def mixed_known_eq_test(methods, eq_prob):
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    eqa = list(nash.mixed_nash(game, as_array=True, processes=1, **methods))
    assert len(eqa) >= 1, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert any(np.allclose(eqm, expected, atol=1e-3) for eqm in eqa), \
        "didn't find correct equilibrium {} instead of {}".format(
            eqa, expected)


@testutils.apply(zip(p/10 for p in range(11)))
def optimization_stable_point_test(eq_prob):
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    opt = nash.RegretOptimizer(game)
    val, grad = opt.grad(np.array([eq_prob, 1 - eq_prob]))
    assert np.isclose(val, 0), \
        "value at equilibrium was not close to zero: {}".format(val)
    assert np.allclose(grad, 0), \
        "grad at equilibrium was not close to zero: {}".format(grad)


def pure_roshambo_test():
    game = gamegen.rock_paper_scissors()
    eqa = list(nash.pure_nash(game))
    assert len(eqa) == 0, "found a pure equilibrium in roshambo"
    eqa = list(nash.pure_nash(game, 1))
    assert len(eqa) == 3, \
        "didn't find low regret ties in roshambo"
    eqa = list(nash.pure_nash(game, 2))
    assert len(eqa) == game.size, \
        "found profiles with more than 2 regret in roshambo"


def minreg_roshambo_test():
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_profile(game)
    assert 2 == next(iter(next(iter(eqm.values())).values())), \
        "min regret profile was not rr, pp, or ss"


def minreg_grid_roshambo_test():
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_grid_mixture(game, 3)  # Not enough for eq
    assert abs(regret.mixture_regret(game, eqm) - .5) < 1e-7, \
        "min regret grid didn't find [.5, .5, 0] profile with regret .5"
    eqm = nash.min_regret_grid_mixture(game, 4)  # hit eqa perfectly
    assert abs(regret.mixture_regret(game, eqm) - 0) < 1e-7, \
        "min regret grid didn't find equilibrium"


def minreg_rand_roshambo_test():
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_rand_mixture(game, 20)
    assert regret.mixture_regret(game, eqm) < 2 + 1e-7, \
        "Found a mixture with greater than maximum regret"


@testutils.apply(zip(ALL_METHODS))
def mixed_roshambo_test(methods):
    game = gamegen.rock_paper_scissors()
    eqa = list(nash.mixed_nash(game, dist_thresh=1e-2, processes=1, **methods))
    assert len(eqa) == 1, "didn't find right number of equilibria in roshambo"
    assert np.allclose(1/3, game.as_mixture(eqa[0], as_array=True)), \
        "roshambo equilibria wasn't uniform"


def at_least_one_test():
    # Equilibrium of game is not at a starting point for equilibria finding
    game = gamegen.sym_2p2s_known_eq(1/math.sqrt(2))
    # Don't converge
    opts = {'max_iters': 0}
    eqa = list(nash.mixed_nash(game, processes=1, replicator=opts))
    assert len(eqa) == 0, "found an equilibrium normally"
    eqa = list(nash.mixed_nash(game, replicator=opts, processes=1,
                               at_least_one=True))
    assert len(eqa) == 1, "at_least_one didn't return anything"


@testutils.apply(zip(
    ALL_METHODS,
    [
        (1, 1, [1]),
        (1, 2, [2]),
        (2, 1, [1, 1]),
        (2, 2, [2, 2]),
        (3, 4, [4, 4, 4]),
        (2, [1, 3], [1, 3]),
    ]))
def mixed_nash_test(methods, game_def):
    players, strategies, exp_strats = game_def
    game = gamegen.independent_game(players, strategies)
    eqa = list(nash.mixed_nash(game, at_least_one=True, processes=1,
                               **methods))
    assert eqa, "Didn't find an equilibria with at_least_one on"


@tools.raises(ValueError)
def empty_game_test():
    game = rsgame.Game.from_game(gamegen.empty_role_symmetric_game(1, 2, 3))
    nash.min_regret_profile(game)


def hard_nash_test():
    with open('test/hard_nash_game_1.json') as f:
        game = rsgame.Game.from_json(json.load(f))
    eqa = nash.mixed_nash(game, as_array=True, processes=1)
    expected = game.as_array({
        'background': {
            'markov:rmin_30000_rmax_30000_thresh_0.001_priceVarEst_1e6':
            0.5407460907477768,
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9':
            0.45925390925222315
        },
        'hft': {
            'trend:trendLength_5_profitDemanded_50_expiration_50': 1.0
        }
    }, float)
    assert any(np.allclose(game.trim_mixture_array_support(eqm), expected,
                atol=1e-4, rtol=1e-4) for eqm in eqa), \
        "Didn't find equilibrium in known hard instance"
