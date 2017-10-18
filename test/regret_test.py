import numpy as np
import pytest

from gameanalysis import gamegen
from gameanalysis import matgame
from gameanalysis import regret
from gameanalysis import paygame
from gameanalysis import rsgame


def test_pure_strategy_deviation_gains():
    profiles = [[2, 0, 2, 0],
                [2, 0, 1, 1],
                [2, 0, 0, 2],
                [1, 1, 2, 0],
                [1, 1, 1, 1],
                [1, 1, 0, 2],
                [0, 2, 2, 0],
                [0, 2, 1, 1],
                [0, 2, 0, 2]]
    payoffs = [[1, 0, 2, 0],
               [3, 0, 4, 5],
               [6, 0, 0, 7],
               [8, 9, 10, 0],
               [11, 12, 13, 14],
               [15, 16, 0, 17],
               [0, 18, 19, 0],
               [0, 20, 21, 22],
               [0, 23, 0, 24]]
    game = paygame.game(2, [2, 2], profiles, payoffs)

    gains = regret.pure_strategy_deviation_gains(game, [2, 0, 2, 0])
    assert np.allclose(gains, [8, 0, 3, 0])
    gains = regret.pure_strategy_deviation_gains(game, [1, 1, 1, 1])
    assert np.allclose(gains, [9, -9, 4, -4])


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
    game = paygame.game(2, 2, profiles, payoffs)
    dg = regret.mixture_deviation_gains(game, [1, 0])
    expected_gains = [0.0, 2.4]
    assert np.allclose(dg, expected_gains), \
        "mixture gains wrong {} instead of {}".format(dg, expected_gains)
    dg = regret.mixture_deviation_gains(game, game.uniform_mixture())
    assert np.isnan(dg).all(), "had data for mixture without data"


def test_mixed_incomplete_data_2():
    profiles = [[2, 0]]
    payoffs = [[1.0, 0.0]]
    game = paygame.game(2, 2, profiles, payoffs)
    dg = regret.mixture_deviation_gains(game, [1, 0])
    assert np.allclose(dg, [0, np.nan], equal_nan=True), \
        "nonzero regret or deviation without payoff didn't return nan"


def test_pure_incomplete_data():
    profiles = [[2, 0]]
    payoffs = [[1.0, 0.0]]
    game = paygame.game(2, 2, profiles, payoffs)
    reg = regret.pure_strategy_regret(game, [2, 0])
    assert np.isnan(reg), "regret of missing profile not nan"


@pytest.mark.parametrize('strategies', list(range(1, 7)) * 20)
def test_two_player_zero_sum_pure_wellfare(strategies):
    game = gamegen.two_player_zero_sum_game(strategies)
    for prof in game.profiles():
        assert np.isclose(regret.pure_social_welfare(game, prof), 0), \
            "zero sum profile wasn't zero sum"


def test_nonzero_profile_welfare():
    game = matgame.matgame([[[3.5, 2.5]]])
    assert np.isclose(regret.pure_social_welfare(game, [1, 1]), 6), \
        "Didn't properly sum welfare"


@pytest.mark.parametrize('strategies', list(range(1, 7)) * 20)
def test_two_player_zero_sum_mixed_wellfare(strategies):
    game = gamegen.two_player_zero_sum_game(strategies)
    for prof in game.random_mixtures(20):
        assert np.isclose(regret.mixed_social_welfare(game, prof), 0), \
            "zero sum profile wasn't zero sum"


def test_nonzero_mixed_welfare():
    game = matgame.matgame([[[3.5, 2.5]]])
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
    for mix in game.random_mixtures(20, alpha=0.05):
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
    game = paygame.game(2, 2, profiles, payoffs)
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
    game = paygame.game(2, 2, profiles, payoffs)
    prof = regret.max_pure_social_welfare(game)[1]
    assert np.all(prof == [1, 1])

    game = rsgame.emptygame(2, 2)
    sw, prof = regret.max_pure_social_welfare(game)
    assert np.isnan(sw)
    assert prof is None

    (sw,), (prof,) = regret.max_pure_social_welfare(game, by_role=True)
    assert np.isnan(sw)
    assert prof is None


def test_max_pure_profile_profile_game():
    """Test that game are correct when profiles have incomplete data"""
    profiles = [[2, 0, 2, 0],
                [1, 1, 2, 0],
                [1, 1, 1, 1]]
    payoffs = [[np.nan, 0, 5, 0],  # Max role 2
               [2, 3, np.nan, 0],  # Max role 1
               [1, 1, 1, 1]]       # Max total
    game = paygame.game([2, 2], [2, 2], profiles, payoffs)
    welfare, profile = regret.max_pure_social_welfare(game)
    assert welfare == 4
    assert np.all(profile == [1, 1, 1, 1])
    welfares, profiles = regret.max_pure_social_welfare(game, by_role=True)
    assert np.allclose(welfares, [5, 10])
    expected = [[1, 1, 2, 0],
                [2, 0, 2, 0]]
    assert np.all(profiles == expected)
