"""Test learning"""
import itertools
import json
import random
import warnings

import autograd
import autograd.numpy as anp
import numpy as np
import pytest
from sklearn import gaussian_process as gp

from gameanalysis import gamegen
from gameanalysis import learning
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame


GAMES = [
    (4, 3),
    ([3, 2], [2, 3]),
    ([2, 2, 2], 2),
]


@pytest.fixture(autouse=True)
def ignore_fit():
    """Ignore fit warnings"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'some lengths were at their bounds, this may indicate a poor fit',
            UserWarning)
        yield


@pytest.mark.parametrize('players,strats', GAMES)
def test_rbfgame_members(players, strats):
    """Test that all functions can be called without breaking"""
    # pylint: disable-msg=protected-access
    game = gamegen.sparse_game(players, strats, 10)
    reggame = learning.rbfgame_train(game)

    prof = reggame.random_profile()
    assert prof.shape == (game.num_strats,)
    pay = reggame.get_payoffs(prof)
    assert prof.shape == pay.shape
    assert np.all((prof > 0) | (pay == 0))

    profs = reggame.random_profiles(20)
    assert profs.shape == (20, game.num_strats)
    pays = reggame.get_payoffs(profs)
    assert profs.shape == pays.shape
    assert np.all((profs > 0) | (pays == 0))

    mix = reggame.random_mixture()
    dev_prof = reggame.random_role_deviation_profile(mix)
    assert dev_prof.shape == (game.num_roles, game.num_strats)
    pay = reggame.get_dev_payoffs(dev_prof)
    assert pay.shape == (game.num_strats,)

    dev_profs = reggame.random_role_deviation_profiles(20, mix)
    assert dev_profs.shape == (20, game.num_roles, game.num_strats)
    pay = reggame.get_dev_payoffs(dev_profs)
    assert pay.shape == (20, game.num_strats)

    assert reggame.is_complete()

    assert reggame._offset.shape == (reggame.num_strats,)
    assert reggame._coefs.shape == (reggame.num_strats,)
    assert reggame._min_payoffs.shape == (reggame.num_strats,)
    assert reggame._max_payoffs.shape == (reggame.num_strats,)

    jgame = json.dumps(reggame.to_json())
    copy = learning.rbfgame_json(json.loads(jgame))
    assert hash(copy) == hash(reggame)
    assert copy == reggame
    assert reggame + copy == copy + reggame


def test_rbfgame_duplicate_profiles():
    """Test learnign with duplicate profiles"""
    profs = [[2, 2],
             [2, 2]]
    pays = [[1, 2],
            [3, 4]]
    game = paygame.samplegame_flat(4, 2, profs, pays)
    learning.rbfgame_train(game)


def test_nntrain():
    """Test training neural networks"""
    game = gamegen.sparse_game([2, 3], [3, 2], 10)
    reggame = learning.nngame_train(game)
    assert np.all((reggame.profiles() > 0) | (reggame.payoffs() == 0))
    assert not np.any(np.isnan(reggame.get_payoffs(
        reggame.random_profile())))
    assert not np.any(np.isnan(reggame.get_dev_payoffs(
        reggame.random_role_deviation_profile())))


def test_nntrain_no_dropout():
    """Test no dropout"""
    game = gamegen.sparse_game([2, 3], [3, 2], 10)
    reggame = learning.nngame_train(game, dropout=0)
    assert np.all((reggame.profiles() > 0) | (reggame.payoffs() == 0))
    assert not np.any(np.isnan(reggame.get_payoffs(
        reggame.random_profile())))
    assert not np.any(np.isnan(reggame.get_dev_payoffs(
        reggame.random_role_deviation_profile())))


def test_skltrain():
    """Test scikit learn traing"""
    game = gamegen.sparse_game([2, 3], [3, 2], 10)
    model = gp.GaussianProcessRegressor(
        1.0 * gp.kernels.RBF(2, [1, 3]) + gp.kernels.WhiteKernel(1))
    reggame = learning.sklgame_train(game, model)
    assert np.all((reggame.profiles() > 0) | (reggame.payoffs() == 0))
    assert not np.any(np.isnan(reggame.get_payoffs(
        reggame.random_profile())))
    assert not np.any(np.isnan(reggame.get_dev_payoffs(
        reggame.random_role_deviation_profile())))

    with pytest.raises(ValueError):
        reggame.deviation_payoffs(game.random_mixture())
    assert game + reggame == reggame + game


@pytest.mark.parametrize('players,strats', GAMES)
@pytest.mark.parametrize('_', range(5))
def test_rbfgame_restriction(players, strats, _): # pylint: disable=too-many-locals
    """Test rbf game restriction"""
    game = gamegen.sparse_game(players, strats, 13)
    reggame = learning.rbfgame_train(game)

    rest = game.random_restriction()
    rreg = reggame.restrict(rest)

    subpays = rreg.payoffs()
    fullpays = reggame.get_payoffs(restrict.translate(
        rreg.profiles(), rest))[:, rest]
    assert np.allclose(subpays, fullpays)

    mix = rreg.random_mixture()
    sub_dev_profs = rreg.random_role_deviation_profiles(20, mix)
    sub_pays = rreg.get_dev_payoffs(sub_dev_profs)
    pays = reggame.get_dev_payoffs(restrict.translate(
        sub_dev_profs, rest))[:, rest]
    assert np.allclose(sub_pays, pays)

    for mix in rreg.random_mixtures(20):
        dev_pay = rreg.deviation_payoffs(mix)
        full_pay = reggame.deviation_payoffs(restrict.translate(
            mix, rest))[rest]
        assert np.allclose(dev_pay, full_pay)

    assert rreg.min_strat_payoffs().shape == (rreg.num_strats,)
    assert rreg.max_strat_payoffs().shape == (rreg.num_strats,)

    jgame = json.dumps(rreg.to_json())
    copy = learning.rbfgame_json(json.loads(jgame))
    assert hash(copy) == hash(rreg)
    assert copy == rreg

    rrest = rreg.random_restriction()
    rrreg = rreg.restrict(rrest)

    assert rrreg.min_strat_payoffs().shape == (rrreg.num_strats,)
    assert rrreg.max_strat_payoffs().shape == (rrreg.num_strats,)

    jgame = json.dumps(rrreg.to_json())
    copy = learning.rbfgame_json(json.loads(jgame))
    assert hash(copy) == hash(rrreg)
    assert copy == rrreg


@pytest.mark.parametrize('players,strats', GAMES)
@pytest.mark.parametrize('_', range(5))
def test_rbfgame_normalize(players, strats, _):
    """Test normalize"""
    game = gamegen.sparse_game(players, strats, 13)
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

    mix = game.random_mixture()
    dev_profs = reggame.random_role_deviation_profiles(20, mix)
    reg_pays = reggame.get_dev_payoffs(dev_profs)
    norm_pays = normreg.get_dev_payoffs(dev_profs) * scale + offset
    assert np.allclose(reg_pays, norm_pays)

    jgame = json.dumps(normreg.to_json())
    copy = learning.rbfgame_json(json.loads(jgame))
    assert copy == normreg


@pytest.mark.parametrize('_', range(20))
def test_sample(_): # pylint: disable=too-many-locals
    """Test sample game"""
    # pylint: disable-msg=protected-access
    game = gamegen.sparse_game([2, 3], [3, 2], 10)
    model = learning.sklgame_train(game, gp.GaussianProcessRegressor(
        1.0 * gp.kernels.RBF(2, [1, 3]) + gp.kernels.WhiteKernel(1),
        normalize_y=True))
    learn = learning.sample(model)
    full = paygame.game_copy(learn)

    @autograd.primitive
    def sample_profs(mix):
        """Sample profiles"""
        return game.random_role_deviation_profiles(
            learn.num_samples, mix).astype(float)

    @autograd.primitive
    def model_pays(profs):
        """Get pays from model"""
        return model.get_dev_payoffs(profs)

    @autograd.primitive
    def const_weights(profs, mix):
        """Get the weights"""
        return np.prod(mix ** profs, 2).repeat(game.num_role_strats, 1)

    @autograd.primitive
    def rep(probs):
        """Repeat an array"""
        return probs.repeat(game.num_role_strats, 1)

    def rep_vjp(_repd, _probs):
        """The jacobian of repeat"""
        return lambda grad: np.add.reduceat(grad, game.role_starts, 1)

    autograd.extend.defvjp(sample_profs, None)
    autograd.extend.defvjp(model_pays, None)
    autograd.extend.defvjp(const_weights, None, None)
    autograd.extend.defvjp(rep, rep_vjp)  # This is wrong in autograd

    def devpays(mix):
        """Compute the dev pays"""
        profs = sample_profs(mix)
        payoffs = model_pays(profs)
        numer = rep(anp.prod(mix ** profs, 2))
        denom = const_weights(profs, mix)
        weights = numer / denom / learn.num_samples
        return anp.einsum('ij,ij->j', weights, payoffs)

    devpays_jac = autograd.jacobian(devpays) # pylint: disable=no-value-for-parameter

    errors = np.zeros(game.num_strats)
    samp_errors = np.zeros(game.num_strats)
    for i, mix in enumerate(itertools.chain(
            game.random_mixtures(20), game.random_sparse_mixtures(20)), 1):
        seed = random.randint(0, 10**9)
        fdev = full.deviation_payoffs(mix)
        np.random.seed(seed)
        dev, jac = learn.deviation_payoffs(mix, jacobian=True)
        avg_err = (fdev - dev) ** 2 / (np.abs(fdev) + 1e-5)
        errors += (avg_err - errors) / i
        samp_err = ((learn.deviation_payoffs(mix) - dev) ** 2 /
                    (np.abs(dev) + 1e-5))
        samp_errors += (samp_err - samp_errors) / i

        np.random.seed(seed)
        tdev = devpays(mix)
        assert np.allclose(dev, tdev)
        np.random.seed(seed)
        tjac = devpays_jac(mix)
        assert np.allclose(jac, tjac)
    assert np.all(errors <= 200 * (samp_errors + 1e-5))

    submask = game.random_restriction()
    sublearn = learn.restrict(submask)
    subfull = full.restrict(submask)
    assert np.allclose(sublearn.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    norm = learn.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)

    assert learning.sample(learn, learn.num_samples) == learn

    learn = learning.sample(learning.rbfgame_train(game))
    jgame = json.dumps(learn.to_json())
    copy = learning.sample_json(json.loads(jgame))
    assert hash(copy) == hash(learn)
    assert copy == learn
    assert learn + copy == copy + learn

    empty = rsgame.empty_copy(learn)
    assert learn + empty == empty


@pytest.mark.parametrize('_', range(20))
def test_point(_): # pylint: disable=too-many-locals
    """Test point

    We increase player number so point is a more accurate estimator.
    """
    # pylint: disable-msg=protected-access
    game = gamegen.sparse_game(1000, 2, 10)
    model = learning.rbfgame_train(game)
    learn = learning.point(model)
    full = paygame.game_copy(learn)
    red = np.eye(game.num_roles).repeat(game.num_role_strats, 0)
    size = np.eye(game.num_strats).repeat(model._sizes, 0)

    def devpays(mix):
        """The deviation payoffs"""
        profile = learn._dev_players * mix
        dev_profiles = anp.dot(size, anp.dot(red, profile))
        vec = ((dev_profiles - model._profiles) /
               model._lengths.repeat(model._sizes, 0))
        rbf = anp.einsum('...ij,...ij->...i', vec, vec)
        exp = anp.exp(-rbf / 2) * model._alpha
        return model._offset + model._coefs * anp.dot(exp, size)

    devpays_jac = autograd.jacobian(devpays) # pylint: disable=no-value-for-parameter

    errors = np.zeros(game.num_strats)
    for i, mix in enumerate(itertools.chain(
            game.random_mixtures(20), game.random_sparse_mixtures(20)), 1):
        fdev = full.deviation_payoffs(mix)
        dev, jac = learn.deviation_payoffs(mix, jacobian=True)
        err = (fdev - dev) ** 2 / (np.abs(dev) + 1e-5)
        errors += (err - errors) / i
        tdev = devpays(mix)
        tjac = devpays_jac(mix)
        assert np.allclose(learn.deviation_payoffs(mix), dev)
        assert np.allclose(dev, tdev)
        assert np.allclose(jac, tjac)

    # Point is a very biased estimator, so errors are large
    assert np.all(errors < 10)

    submask = game.random_restriction()
    sublearn = learn.restrict(submask)
    subfull = full.restrict(submask)
    assert np.allclose(sublearn.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    norm = learn.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)

    assert learning.point(learn) == learn

    jgame = json.dumps(learn.to_json())
    copy = learning.point_json(json.loads(jgame))
    assert hash(copy) == hash(learn)
    assert copy == learn
    assert learn + copy == copy + learn

    empty = rsgame.empty_copy(learn)
    assert learn + empty == empty


@pytest.mark.parametrize('_', range(20))
def test_neighbor(_): # pylint: disable=too-many-locals
    """Test neighbor games"""
    game = gamegen.sparse_game([2, 3], [3, 2], 10)
    model = gp.GaussianProcessRegressor(
        1.0 * gp.kernels.RBF(2, [1, 3]) + gp.kernels.WhiteKernel(1),
        normalize_y=True)
    learn = learning.neighbor(learning.sklgame_train(game, model))
    full = paygame.game_copy(learn)

    errors = np.zeros(game.num_strats)
    for i, mix in enumerate(itertools.chain(
            game.random_mixtures(20), game.random_sparse_mixtures(20)), 1):
        tdev = full.deviation_payoffs(mix)
        dev, _ = learn.deviation_payoffs(mix, jacobian=True)
        err = (tdev - dev) ** 2 / (np.abs(dev) + 1e-5)
        errors += (err - errors) / i
    assert np.all(errors < 5)

    submask = game.random_restriction()
    sublearn = learn.restrict(submask)
    subfull = full.restrict(submask)
    assert np.allclose(sublearn.get_payoffs(subfull.profiles()),
                       subfull.payoffs())

    norm = learn.normalize()
    assert np.allclose(norm.min_role_payoffs(), 0)
    assert np.allclose(norm.max_role_payoffs(), 1)

    assert learning.neighbor(learn, learn.num_neighbors) == learn

    learn = learning.neighbor(learning.rbfgame_train(game))
    jgame = json.dumps(learn.to_json())
    copy = learning.neighbor_json(json.loads(jgame))
    assert hash(copy) == hash(learn)
    assert copy == learn
    assert learn + copy == copy + learn

    empty = rsgame.empty_copy(learn)
    assert learn + empty == empty


@pytest.mark.parametrize('players,strats', [
    [[1, 5], [2, 2]],
    [[2, 3], [3, 2]],
])
@pytest.mark.parametrize('_', range(20))
def test_rbfgame_min_max_payoffs(players, strats, _):
    """Test min and max payoffs of rbf game"""
    game = gamegen.sparse_game(players, strats, 11)
    reggame = learning.rbfgame_train(game)
    full = paygame.game_copy(reggame)

    assert np.all(full.min_strat_payoffs() >=
                  reggame.min_strat_payoffs() - 1e-4)
    assert np.all(full.max_strat_payoffs() <=
                  reggame.max_strat_payoffs() + 1e-4)


def test_rbfgame_equality():
    """Test all branches of equality test"""
    # pylint: disable-msg=protected-access
    game = gamegen.sparse_game([2, 3], [3, 2], 10)
    regg = learning.rbfgame_train(game)
    copy = regg.restrict(np.ones(game.num_strats, bool))
    copy._alpha.setflags(write=True)
    copy._profiles.setflags(write=True)
    copy._lengths.setflags(write=True)
    copy._coefs.setflags(write=True)
    copy._offset.setflags(write=True)
    assert regg == copy

    for alph, prof in zip(np.split(copy._alpha, copy._size_starts[1:]),
                          np.split(copy._profiles, copy._size_starts[1:])):
        perm = np.random.permutation(alph.size)
        np.copyto(alph, alph[perm])
        np.copyto(prof, prof[perm])
    assert regg == copy

    copy._alpha.fill(0)
    assert regg != copy
    copy._profiles.fill(0)
    assert regg != copy
    copy._lengths.fill(0)
    assert regg != copy
    copy._coefs.fill(0)
    assert regg != copy
    copy._offset.fill(0)
    assert regg != copy


@pytest.mark.parametrize('players,strats,num', [
    (10, 3, 15),
    ([2, 3], [3, 2], 15),
    ([1, 3], [2, 2], 8),
])
@pytest.mark.parametrize('_', range(10)) # pylint: disable=too-many-locals
def test_continuous_approximation(players, strats, num, _): # pylint: disable=too-many-locals
    """Test continuous approximation"""
    # pylint: disable-msg=protected-access
    game = gamegen.sparse_game(players, strats, num)
    learn = learning.rbfgame_train(game)
    full = paygame.game_copy(learn)
    red = np.eye(game.num_roles).repeat(game.num_role_strats, 0)
    size = np.eye(game.num_strats).repeat(learn._sizes, 0)

    def devpays(mix):
        """Compute dev pays"""
        players = learn._dev_players.repeat(game.num_role_strats, 1)
        avg_prof = players * mix
        diag = 1 / (learn._lengths ** 2 + avg_prof)
        diag_sizes = anp.dot(size, diag)
        diff = learn._profiles - anp.dot(size, avg_prof)
        det = 1 / (1 - learn._dev_players * anp.dot(mix ** 2 * diag, red))
        det_sizes = anp.dot(size, det)
        cov_diag = anp.einsum('ij,ij,ij->i', diff, diff, diag_sizes)
        cov_outer = anp.dot(mix * diag_sizes * diff, red)
        sec_term = anp.einsum(
            'ij,ij,ij,ij->i', learn._dev_players.repeat(learn._sizes, 0),
            det_sizes, cov_outer, cov_outer)
        exp = anp.exp(-(cov_diag + sec_term) / 2)
        coef = anp.prod(learn._lengths, 1) * anp.sqrt(
            anp.prod(diag, 1) * anp.prod(det, 1))
        avg = anp.dot(learn._alpha * exp, size)
        return learn._coefs * coef * avg + learn._offset

    devpays_jac = autograd.jacobian(devpays) # pylint: disable=no-value-for-parameter

    for mix in itertools.chain(game.random_mixtures(20),
                               game.random_sparse_mixtures(20)):
        dev = full.deviation_payoffs(mix)
        adev, ajac = learn.deviation_payoffs(mix, jacobian=True)
        assert np.allclose(adev, dev, rtol=0.1, atol=0.2)
        tdev = devpays(mix)
        tjac = devpays_jac(mix)
        assert np.allclose(adev, tdev)
        assert np.allclose(ajac, tjac)
