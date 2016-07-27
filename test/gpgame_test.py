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
    profs = base.random_profiles(base.uniform_mixture(),
                                 3 * base.num_strategies.max())
    _, keep = np.unique(base.profile_id(profs), return_index=True)
    profs = profs[keep]
    if np.any(np.sum(profs > 0, 0) < 3):
        # We need at least 3 profiles per strategy for x-validation
        return
    game_data = gamegen.add_noise(rsgame.Game(
        base, profs, np.zeros(profs.shape)), 1)
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
