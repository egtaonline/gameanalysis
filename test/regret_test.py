import numpy as np
import pytest

from gameanalysis import gamegen
from gameanalysis import regret
from gameanalysis import rsgame


@pytest.mark.parametrize('_', range(20))
def test_pure_prisoners_dilemma(_):
    game = gamegen.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    eqm = [0, 2]

    assert regret.pure_strategy_regret(game, eqm) == 0, \
        "Known equilibrium was not zero regret"


@pytest.mark.parametrize('_', range(20))
def test_mixed_prisoners_dilemma(_):
    game = gamegen.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    eqm = [0, 1]

    assert regret.mixture_regret(game, eqm) == 0, \
        "Known symmetric mixed was not zero regret"


def test_mixed_incomplete_data():
    profiles = [[2, 0],
                [1, 1]]
    payoffs = [[4.3, 0],
               [6.2, 6.7]]
    game = rsgame.Game(2, 2, profiles, payoffs)
    dg = regret.mixture_deviation_gains(game, [1, 0])
    expected_gains = [0.0, 2.4]
    assert np.allclose(dg, expected_gains), \
        "mixture gains wrong {} instead of {}".format(dg, expected_gains)
    dg = regret.mixture_deviation_gains(game, game.uniform_mixture())
    assert np.isnan(dg).all(), "had data for mixture without data"


def test_mixed_incomplete_data_2():
    profiles = [[2, 0]]
    payoffs = [[1.0, 0.0]]
    game = rsgame.Game(2, 2, profiles, payoffs)
    dg = regret.mixture_deviation_gains(game, [1, 0])
    assert np.allclose(dg, [0, np.nan], equal_nan=True), \
        "nonzero regret or deviation without payoff didn't return nan"


def test_pure_incomplete_data():
    profiles = [[2, 0]]
    payoffs = [[1.0, 0.0]]
    game = rsgame.Game(2, 2, profiles, payoffs)
    reg = regret.pure_strategy_regret(game, [2, 0])
    assert np.isnan(reg), "regret of missing profile not nan"


@pytest.mark.parametrize('strategies', list(range(1, 7)) * 20)
def test_two_player_zero_sum_pure_wellfare(strategies):
    game = gamegen.two_player_zero_sum_game(strategies)
    for prof in game.profiles:
        assert np.isclose(regret.pure_social_welfare(game, prof), 0), \
            "zero sum profile wasn't zero sum"


def test_nonzero_profile_welfare():
    game = rsgame.Game([[[3.5, 2.5]]])
    assert np.isclose(regret.pure_social_welfare(game, [1, 1]), 6), \
        "Didn't properly sum welfare"


@pytest.mark.parametrize('strategies', list(range(1, 7)) * 20)
def test_two_player_zero_sum_mixed_wellfare(strategies):
    game = gamegen.two_player_zero_sum_game(strategies)
    for prof in game.random_mixtures(20):
        assert np.isclose(regret.mixed_social_welfare(game, prof), 0), \
            "zero sum profile wasn't zero sum"


def test_nonzero_mixed_welfare():
    game = rsgame.Game([[[3.5, 2.5]]])
    assert np.isclose(regret.mixed_social_welfare(game, [1, 1]), 6), \
        "Didn't properly sum welfare"


@pytest.mark.parametrize('players,strategies', [
    ([1], 1),
    ([1], 2),
    ([2], 1),
    ([2], 2),
    (2 * [1], 1),
    (2 * [1], 2),
    (2 * [2], 1),
    (2 * [2], 2),
    ([1, 2], 2),
    (2, [1, 2]),
    ([1, 2], [1, 2]),
    ([3, 4], [2, 3]),
])
# Test that for complete games, there are never any nan deviations.
def test_nan_deviations(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    for mix in game.random_mixtures(20, 0.05):
        mix = game.trim_mixture_support(mix)
        gains = regret.mixture_deviation_gains(game, mix)
        assert not np.isnan(gains).any(), \
            "deviation gains in complete game were nan"


def test_max_mixed_profile():
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[1, 0],
               [3, 3],
               [0, 1]]
    game = rsgame.Game(2, 2, profiles, payoffs)
    mix1 = regret.max_mixed_social_welfare(game, processes=1)[1]
    mix2 = regret.max_mixed_social_welfare(game)[1]
    assert np.allclose(mix1, [0.5, 0.5])
    assert np.allclose(mix2, [0.5, 0.5])


def test_max_pure_profile():
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[3, 0],
               [4, 4],
               [0, 1]]
    game = rsgame.Game(2, 2, profiles, payoffs)
    prof = regret.max_pure_social_welfare(game)[1]
    assert np.all(prof == [2, 0])

    game = rsgame.Game(rsgame.BaseGame(2, 2))
    sw, prof = regret.max_pure_social_welfare(game)
    assert np.isnan(sw)
    assert prof is None
