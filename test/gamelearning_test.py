# XXX pytest fails to load keras dynamically, so we must explicitly import it
# here for testing. This way it's still only loaded from gameanalysis when an
# actual model is used.
import keras  # noqa
import numpy as np
import pytest

from gameanalysis import gamegen
from gameanalysis import gamelearning
from gameanalysis import rsgame


@pytest.mark.parametrize('reg_method', ['gp', 'nn'])
@pytest.mark.parametrize('ev_method', ['full', 'sample', 'point', 'neighbor'])
def test_basic_functions(reg_method, ev_method):
    """Test that all functions can be called without breaking"""
    game = gamegen.add_profiles(rsgame.emptygame([4, 3], [3, 4]), 200)
    reggame = gamelearning.RegressionGame(game, reg_method, ev_method)

    assert np.all(reggame.min_strat_payoffs() == game.min_strat_payoffs())
    assert np.all(reggame.max_strat_payoffs() == game.max_strat_payoffs())

    assert reggame.is_complete()

    dev = reggame.deviation_payoffs(game.random_mixtures())
    assert not np.isnan(dev).any()

    pays = reggame.get_payoffs(game.random_profiles())
    assert not np.isnan(pays).any()

    # Test copy is effective
    # FIXME This is a poor api, requiring you to respecify some things...
    copy = gamelearning.RegressionGame(reggame, reg_method, 'point')
    profs = reggame.random_profiles(20)
    pays = reggame.get_payoffs(profs)
    cpays = copy.get_payoffs(profs)
    assert np.allclose(pays, cpays)
