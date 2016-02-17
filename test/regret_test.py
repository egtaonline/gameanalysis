import math

import numpy as np

from gameanalysis import randgames
from gameanalysis import regret
from gameanalysis import rsgame
from test import testutils


@testutils.apply(repeat=20)
def pure_prisoners_dilemma_test():
    game = randgames.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    role = next(iter(game.strategies))
    strats = list(game.strategies[role])
    eqm = {role: {strats[1]: 2}}

    assert regret.pure_strategy_regret(game, eqm) == 0, \
        "Known equilibrium was not zero regret"


@testutils.apply(repeat=20)
def mixed_prisoners_dilemma_test():
    game = randgames.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    role = next(iter(game.strategies))
    strats = list(game.strategies[role])
    eqm = {role: {strats[1]: 1}}

    assert regret.mixture_regret(game, eqm) == 0, \
        "Known symmetric mixed was not zero regret"


def mixed_incomplete_data_test():
    game = rsgame.Game({'r': 2}, {'r': ['1', '2']},
                       np.array([[[2, 0]],
                                 [[1, 1]]]),
                       np.array([[[4.3, 0]],
                                 [[6.2, 6.7]]]))
    dg = regret.mixture_deviation_gains(game, {'r': {'1': 1}}, as_array=True)
    assert np.allclose(dg, np.array([[0.0, 2.4]])), "mixture gains wrong"
    dg = regret.mixture_deviation_gains(game, game.uniform_mixture(), True)
    assert np.isnan(dg).all(), "had data for mixture without data"


def mixed_incomplete_data_test_2():
    game = rsgame.Game({'r': 2}, {'r': ['1', '2']},
                       np.array([[[2, 0]]]),
                       np.array([[[1.0, 0.0]]]))
    dg = regret.mixture_deviation_gains(game, {'r': {'1': 1}})
    assert dg['r']['1'] == 0, "nonzero regret for mixture"
    assert math.isnan(dg['r']['2']), \
        "deviation without payoff didn't return nan"


@testutils.apply(zip(range(6)), repeat=20)
def two_player_zero_sum_pure_wellfare_test(strategies):
    game = randgames.zero_sum_game(6)
    for prof in game.all_profiles():
        assert abs(regret.pure_social_welfare(game, prof)) < 1e-5, \
            "zero sum profile wasn't zero sum"


def nonzero_profile_welfare_test():
    game = rsgame.Game.from_matrix({'a': ['s'], 'b': ['s']},
                                   np.array([[[3.5, 2.5]]]))
    assert abs(6 - regret.pure_social_welfare(
        game, {'a': {'s': 1}, 'b': {'s': 1}})) < 1e-5, \
        "Didn't properly sum welfare"


@testutils.apply(zip(range(6)), repeat=20)
def two_player_zero_sum_mixed_wellfare_test(strategies):
    game = randgames.zero_sum_game(6)
    for _ in range(20):
        prof = game.random_mixture()
        assert abs(regret.mixed_social_welfare(game, prof)) < 1e-5, \
            "zero sum profile wasn't zero sum"


def nonzero_mixed_welfare_test():
    game = rsgame.Game.from_matrix({'a': ['s'], 'b': ['s']},
                                   np.array([[[3.5, 2.5]]]))
    assert abs(6 - regret.mixed_social_welfare(
        game, {'a': {'s': 1}, 'b': {'s': 1}})) < 1e-5, \
        "Didn't properly sum welfare"
