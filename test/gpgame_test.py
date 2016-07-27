import itertools

import numpy as np

from gameanalysis import gamegen
from gameanalysis import gpgame
from gameanalysis import rsgame

from test import testutils


@testutils.apply(itertools.product(testutils.game_sizes(), range(5)))
def test_nearby_profiles(game_params, num_devs):
    # TODO There is probably a better way to test this, but it requires moving
    # nearyby_profs out of a game the requires enough data for x-validation
    base = rsgame.BaseGame(*game_params)
    game_data = gamegen.add_profiles(base, min(base.num_all_profiles,
                                               3 * base.num_strategies.max()))
    if np.any(np.sum(game_data.profiles > 0, 0) < 3):
        # We need at least 3 profiles per strategy for x-validation
        return
    game = gpgame.NeighborGPGame(game_data)
    mix = game.random_mixtures()[0]
    prof = game.random_profiles(mix)[0]
    nearby = game.nearby_profs(prof, num_devs)
    diff = nearby - prof
    devs_from = game.role_reduce((diff < 0) * -diff)
    devs_to = game.role_reduce((diff > 0) * diff)
    assert np.all(devs_to.sum(1) <= num_devs)
    assert np.all(devs_from.sum(1) <= num_devs)
    assert np.all(devs_to == devs_from)
    assert np.all(game.verify_profile(nearby))


def test_basic_functions():
    """Test that all functions can be called without breaking"""
    base = rsgame.BaseGame([4, 3], [3, 4])
    game = gamegen.add_profiles(base, 200)
    gpbase = gpgame.BaseGPGame(game)
    mix = game.random_mixtures()[0]
    assert np.all(gpbase.min_payoffs() == game.min_payoffs())
    assert np.all(gpbase.max_payoffs() == game.max_payoffs())

    gppoint = gpgame.PointGPGame(gpbase)
    gppoint.deviation_payoffs(mix)

    gpsample = gpgame.SampleGPGame(gpbase, num_samples=100)
    gpsample.deviation_payoffs(mix)

    gpneighbor = gpgame.NeighborGPGame(gpbase)
    gpneighbor.deviation_payoffs(mix)

    gpdpr = gpgame.DprGPGame(gpbase)
    gpdpr.deviation_payoffs(mix)

    gpfull = gpgame.FullGPGame(gpbase)
    gpfull.deviation_payoffs(mix)
