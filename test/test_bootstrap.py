"""Test bootstrap"""
import numpy as np
import pytest

from gameanalysis import bootstrap
from gameanalysis import gamegen
from test import utils # pylint: disable=wrong-import-order


@pytest.mark.parametrize('num_mixes', [5])
@pytest.mark.parametrize('num_boots', [200])
@pytest.mark.parametrize('base', utils.basic_games())
def test_mixture_welfare(base, num_mixes, num_boots):
    """Test bootstrap mixture welfare"""
    game = gamegen.samplegame_replace(base)
    mixes = game.random_mixtures(num_mixes)
    boots = bootstrap.mixture_welfare(game, mixes, num_boots, processes=1)
    assert boots.shape == (num_mixes, num_boots)


@pytest.mark.parametrize('num_mixes', [5])
@pytest.mark.parametrize('num_boots', [200])
@pytest.mark.parametrize('base', utils.basic_games())
def test_mixture_regret(base, num_mixes, num_boots):
    """Test bootstrap mixture regret"""
    game = gamegen.samplegame_replace(base)
    mixes = game.random_mixtures(num_mixes)
    boots = bootstrap.mixture_regret(game, mixes, num_boots, processes=1)
    assert boots.shape == (num_mixes, num_boots)
    assert np.all(boots >= 0)

    perc_boots = bootstrap.mixture_regret(game, mixes, num_boots,
                                          percentiles=[2.5, 97.5], processes=1)
    assert perc_boots.shape == (num_mixes, 2)
    assert np.all(perc_boots >= 0)


@pytest.mark.parametrize('num_boots', [200])
@pytest.mark.parametrize('base', utils.basic_games())
def test_mixture_regret_single_mix(base, num_boots):
    """Test bootstrap regret with a single mix"""
    game = gamegen.samplegame_replace(base)
    mix = game.random_mixture()
    boots = bootstrap.mixture_regret(game, mix, num_boots, processes=1)
    assert boots.shape == (1, num_boots)
    assert np.all(boots >= 0)


@pytest.mark.parametrize('num_mixes', [5])
@pytest.mark.parametrize('num_boots', [200])
def test_mixture_regret_parallel(num_mixes, num_boots):
    """Test mixture regret run on multiple processors"""
    game = gamegen.samplegame([4, 3], [3, 4])
    mixes = game.random_mixtures(num_mixes)
    boots = bootstrap.mixture_regret(game, mixes, num_boots)
    assert boots.shape == (num_mixes, num_boots)
    assert np.all(boots >= 0)
