import numpy as np
import pytest

from gameanalysis import bootstrap
from gameanalysis import gamegen

SMALL_GAMES = [
    ([1], 1),
    ([1], 2),
    ([2], 1),
    ([2], 2),
    ([2], 5),
    ([5], 2),
    ([5], 5),
    (2 * [1], 1),
    (2 * [1], 2),
    (2 * [2], 1),
    (2 * [2], 2),
    (5 * [1], 2),
]
GAMES = SMALL_GAMES + [
    (2 * [1], 5),
    (2 * [2], 5),
    (2 * [5], 2),
    (2 * [5], 5),
    (3 * [3], 3),
    (5 * [1], 5),
    ([170], 2),
    ([180], 2),
    ([1, 2], 2),
    ([1, 2], [2, 1]),
    (2, [1, 2]),
    ([3, 4], [2, 3]),
    ([2, 3, 4], [4, 3, 2]),
]


@pytest.mark.parametrize('players,strategies', GAMES)
def test_mixture_welfare(players, strategies):
    num_mixes = 5
    num_boots = 200
    game = gamegen.add_noise(gamegen.role_symmetric_game(players, strategies),
                             1, 3)
    mixes = game.random_mixtures(num_mixes)
    boots = bootstrap.mixture_welfare(game, mixes, num_boots, processes=1)
    assert boots.shape == (num_mixes, num_boots)


@pytest.mark.parametrize('players,strategies', SMALL_GAMES)
def test_mixture_regret(players, strategies):
    num_mixes = 5
    num_boots = 200
    game = gamegen.add_noise(gamegen.role_symmetric_game(players, strategies),
                             1, 3)
    mixes = game.random_mixtures(num_mixes)
    boots = bootstrap.mixture_regret(game, mixes, num_boots, processes=1)
    assert boots.shape == (num_mixes, num_boots)
    assert np.all(boots >= 0)

    perc_boots = bootstrap.mixture_regret(game, mixes, num_boots,
                                          percentiles=[2.5, 97.5], processes=1)
    assert perc_boots.shape == (num_mixes, 2)
    assert np.all(perc_boots >= 0)


@pytest.mark.parametrize('players,strategies', GAMES)
def test_mixture_regret_single_mix(players, strategies):
    num_boots = 200
    game = gamegen.add_noise(gamegen.role_symmetric_game(players, strategies),
                             1, 3)
    mix = game.random_mixtures()
    boots = bootstrap.mixture_regret(game, mix, num_boots, processes=1)
    assert boots.shape == (1, num_boots)
    assert np.all(boots >= 0)


def test_mixture_regret_parallel():
    num_mixes = 5
    num_boots = 200
    game = gamegen.add_noise(gamegen.role_symmetric_game([4, 3], [3, 4]), 1,
                             3)
    mixes = game.random_mixtures(num_mixes)
    boots = bootstrap.mixture_regret(game, mixes, num_boots)
    assert boots.shape == (num_mixes, num_boots)
    assert np.all(boots >= 0)