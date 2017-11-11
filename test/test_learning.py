import warnings

import numpy as np
import pytest
from sklearn import gaussian_process as gp

from gameanalysis import gamegen
from gameanalysis import learning
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import subgame


def test_rbfgame_members():
    """Test that all functions can be called without breaking"""
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    reggame = learning.rbfgame_train(game)

    prof = reggame.random_profiles()
    assert prof.shape == (game.num_strats,)
    pay = reggame.get_payoffs(prof)
    assert prof.shape == pay.shape
    assert np.all((prof > 0) | (pay == 0))

    profs = reggame.random_profiles(20)
    assert profs.shape == (20, game.num_strats)
    pays = reggame.get_payoffs(profs)
    assert profs.shape == pays.shape
    assert np.all((profs > 0) | (pays == 0))

    mix = reggame.random_mixtures()
    dev_prof = reggame.random_dev_profiles(mix)
    assert dev_prof.shape == (game.num_roles, game.num_strats)
    pay = reggame.get_mean_dev_payoffs(dev_prof)
    assert pay.shape == (game.num_strats,)

    dev_profs = reggame.random_dev_profiles(mix, 20)
    assert dev_profs.shape == (20, game.num_roles, game.num_strats)
    pay = reggame.get_mean_dev_payoffs(dev_profs)
    assert pay.shape == (game.num_strats,)

    assert reggame.is_complete()

    assert len(reggame._regressors) == game.num_strats
    assert reggame._offset.shape == (reggame.num_strats,)
    assert reggame._scale.shape == (reggame.num_strats,)
    assert reggame._min_payoffs.shape == (reggame.num_strats,)
    assert reggame._max_payoffs.shape == (reggame.num_strats,)
    assert reggame._sub_mask.shape == (reggame.num_strats,)
    assert reggame._sub_mask.all()


def test_nntrain():
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reggame = learning.nngame_train(game)
    assert np.all((reggame.profiles() > 0) | (reggame.payoffs() == 0))
    assert not np.any(np.isnan(reggame.get_payoffs(
        reggame.random_profiles())))
    assert not np.any(np.isnan(reggame.get_mean_dev_payoffs(
        reggame.random_dev_profiles(reggame.random_mixtures()))))


def test_nntrain_no_dropout():
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    with warnings.catch_warnings():
        # Keras has some warning associated with loading tensorflow
        warnings.simplefilter('ignore')
        reggame = learning.nngame_train(game, dropout=0)
    assert np.all((reggame.profiles() > 0) | (reggame.payoffs() == 0))
    assert not np.any(np.isnan(reggame.get_payoffs(
        reggame.random_profiles())))
    assert not np.any(np.isnan(reggame.get_mean_dev_payoffs(
        reggame.random_dev_profiles(reggame.random_mixtures()))))


def test_skltrain():
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    model = gp.GaussianProcessRegressor(
        1.0 * gp.kernels.RBF(2, [1, 3]) + gp.kernels.WhiteKernel(1))
    reggame = learning.sklgame_train(game, model)
    assert np.all((reggame.profiles() > 0) | (reggame.payoffs() == 0))
    assert not np.any(np.isnan(reggame.get_payoffs(
        reggame.random_profiles())))
    assert not np.any(np.isnan(reggame.get_mean_dev_payoffs(
        reggame.random_dev_profiles(reggame.random_mixtures()))))


@pytest.mark.parametrize('_', range(20))
def test_rbfgame_subgame(_):
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    reggame = learning.rbfgame_train(game)

    sub_mask = game.random_subgames()
    subreg = reggame.subgame(sub_mask)

    subpays = subreg.payoffs()
    fullpays = reggame.get_payoffs(subgame.translate(
        subreg.profiles(), sub_mask))[:, sub_mask]
    assert np.allclose(subpays, fullpays)

    mix = subreg.random_mixtures()
    sub_dev_profs = subreg.random_dev_profiles(mix, 20)
    sub_pays = subreg.get_mean_dev_payoffs(sub_dev_profs)
    pays = reggame.get_mean_dev_payoffs(subgame.translate(
        sub_dev_profs, sub_mask))[sub_mask]
    assert np.allclose(sub_pays, pays)

    assert len(subreg._regressors) == subreg.num_strats
    assert subreg._offset.shape == (subreg.num_strats,)
    assert subreg._scale.shape == (subreg.num_strats,)
    assert subreg._min_payoffs.shape == (subreg.num_strats,)
    assert subreg._max_payoffs.shape == (subreg.num_strats,)
    assert subreg._sub_mask.shape == (game.num_strats,)

    subsubmask = subreg.random_subgames()
    subsubreg = subreg.subgame(subsubmask)

    assert len(subsubreg._regressors) == subsubreg.num_strats
    assert subsubreg._offset.shape == (subsubreg.num_strats,)
    assert subsubreg._scale.shape == (subsubreg.num_strats,)
    assert subsubreg._min_payoffs.shape == (subsubreg.num_strats,)
    assert subsubreg._max_payoffs.shape == (subsubreg.num_strats,)
    assert subsubreg._sub_mask.shape == (game.num_strats,)


@pytest.mark.parametrize('_', range(20))
def test_rbfgame_normalize(_):
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    reggame = learning.rbfgame_train(game)
    normreg = reggame.normalize()
    assert reggame != normreg

    assert np.allclose(normreg.min_role_payoffs(), 0)
    assert np.all(np.isclose(normreg.max_role_payoffs(), 1) |
                  np.isclose(normreg.max_role_payoffs(), 0))

    scale = (reggame.max_role_payoffs() - reggame.min_role_payoffs()).repeat(
        game.num_role_strats)
    offset = reggame.min_role_payoffs().repeat(game.num_role_strats)

    reg_pays = reggame.payoffs()
    norm_pays = normreg.get_payoffs(reggame.profiles())
    norm_pays *= scale
    norm_pays += offset
    norm_pays[reggame.profiles() == 0] = 0
    assert np.allclose(reg_pays, norm_pays)

    mix = game.random_mixtures()
    dev_profs = reggame.random_dev_profiles(mix, 20)
    reg_pays = reggame.get_mean_dev_payoffs(dev_profs)
    norm_pays = normreg.get_mean_dev_payoffs(dev_profs) * scale + offset
    assert np.allclose(reg_pays, norm_pays)


@pytest.mark.parametrize('_', range(20))
def test_sample(_):
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    reggame = learning.sample(learning.rbfgame_train(game))
    full = paygame.game_copy(reggame)

    errors = np.zeros(game.num_strats)
    for i, mix in enumerate(game.grid_mixtures(11), 1):
        truth = full.deviation_payoffs(mix)
        approx = reggame.deviation_payoffs(mix)
        avg_err = np.abs(truth - approx)
        errors += (avg_err - errors) / i
    assert np.all(avg_err < 0.1)

    submask = game.random_subgames()
    subreg = reggame.subgame(submask)
    subfull = full.subgame(submask)
    assert np.allclose(subreg.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    norm = reggame.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)

    assert learning.sample(reggame, reggame._num_samples) == reggame


@pytest.mark.parametrize('_', range(20))
def test_point(_):
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    reggame = learning.point(learning.rbfgame_train(game))
    full = paygame.game_copy(reggame)

    errors = np.zeros(game.num_strats)
    for i, mix in enumerate(game.grid_mixtures(11), 1):
        truth = full.deviation_payoffs(mix)
        approx = reggame.deviation_payoffs(mix)
        avg_err = np.abs(truth - approx)
        errors += (avg_err - errors) / i
    assert np.all(avg_err < 0.1)

    submask = game.random_subgames()
    subreg = reggame.subgame(submask)
    subfull = full.subgame(submask)
    assert np.allclose(subreg.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    norm = reggame.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)

    assert learning.point(reggame) == reggame


@pytest.mark.parametrize('_', range(20))
def test_neighbor(_):
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    reggame = learning.neighbor(learning.rbfgame_train(game))
    full = paygame.game_copy(reggame)

    errors = np.zeros(game.num_strats)
    for i, mix in enumerate(game.grid_mixtures(11), 1):
        truth = full.deviation_payoffs(mix)
        approx = reggame.deviation_payoffs(mix)
        avg_err = np.abs(truth - approx)
        errors += (avg_err - errors) / i
    assert np.all(avg_err < 0.1)

    submask = game.random_subgames()
    subreg = reggame.subgame(submask)
    subfull = full.subgame(submask)
    assert np.allclose(subreg.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    norm = reggame.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)

    assert learning.neighbor(reggame, reggame._num_devs) == reggame


@pytest.mark.parametrize('_', range(20))
def test_continuous_approximation(_):
    game = gamegen.add_profiles(rsgame.emptygame([2, 3], [3, 2]), 10)
    reggame = learning.rbfgame_train(game)
    full = paygame.game_copy(reggame)

    assert np.all(full.min_strat_payoffs() >=
                  reggame.min_strat_payoffs() - 1e-4)
    assert np.all(full.max_strat_payoffs() <=
                  reggame.max_strat_payoffs() + 1e-4)

    errors = np.zeros(game.num_strats)
    mixes = game.grid_mixtures(11) * .9 + game.uniform_mixture() * .1
    for i, mix in enumerate(mixes, 1):
        truth = full.deviation_payoffs(mix)
        approx = reggame.deviation_payoffs(mix)
        avg_err = np.abs(truth - approx)
        errors += (avg_err - errors) / i
    assert np.all(avg_err < 0.1)

    submask = game.random_subgames()
    subreg = reggame.subgame(submask)
    subfull = full.subgame(submask)
    assert np.allclose(subreg.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    mixes = subreg.grid_mixtures(11) * .9 + subreg.uniform_mixture() * .1
    errors = np.zeros(subreg.num_strats)
    for i, mix in enumerate(mixes, 1):
        truth = subfull.deviation_payoffs(mix)
        approx = subreg.deviation_payoffs(mix)
        avg_err = np.abs(truth - approx)
        errors += (avg_err - errors) / i
    assert np.all(avg_err < 0.5)

    norm = reggame.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)


def test_continuous_approximation_one_players():
    game = gamegen.add_profiles(rsgame.emptygame([1, 3], [2, 2]))
    reggame = learning.rbfgame_train(game)
    full = paygame.game_copy(reggame)

    assert np.all(full.min_strat_payoffs() >=
                  reggame.min_strat_payoffs() - 1e-4)
    assert np.all(full.max_strat_payoffs() <=
                  reggame.max_strat_payoffs() + 1e-4)

    errors = np.zeros(game.num_strats)
    for i, mix in enumerate(game.grid_mixtures(11), 1):
        truth = full.deviation_payoffs(mix)
        approx = reggame.deviation_payoffs(mix)
        avg_err = np.abs(truth - approx)
        errors += (avg_err - errors) / i
    assert np.all(avg_err < 0.1)

    submask = game.random_subgames()
    subreg = reggame.subgame(submask)
    subfull = full.subgame(submask)
    assert np.allclose(subreg.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    norm = reggame.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)
