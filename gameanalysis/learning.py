"""Package for learning complete games from data

The API of this individual module is still unstable and may change as
improvements or refinements are made.

There are two general game types in this module: learned games and deviation
games. Learned games vary by the method, but generally expose methods for
computing payoffs and may other features. Deviation games use learned games and
different functions to compute deviation payoffs via various methods.
"""
import itertools

import numpy as np
import sklearn
from sklearn import gaussian_process as gp

from . import paygame
from . import rsgame
from . import subgame
from . import utils


_TINY = np.finfo(float).tiny


class DevRegressionGame(rsgame.CompleteGame):
    """A game regression model that learns deviation payoffs

    This model functions as a game, but doesn't have a default way of computing
    deviation payoffs. It must be wrapped with another game that uses payoff
    data to compute deviation payoffs.
    """

    def __init__(self, game, regressors, offset, scale, min_payoffs,
                 max_payoffs, sub_mask):
        super().__init__(game.role_names, game.strat_names,
                         game.num_role_players)
        self._regressors = regressors
        self._offset = offset
        self._offset.setflags(write=False)
        self._scale = scale
        self._scale.setflags(write=False)
        self._min_payoffs = min_payoffs
        self._min_payoffs.setflags(write=False)
        self._max_payoffs = max_payoffs
        self._max_payoffs.setflags(write=False)
        self._sub_mask = sub_mask
        self._sub_mask.setflags(write=False)

    def get_payoffs(self, profiles):
        assert self.is_profile(profiles).all(), "must pass valid profiles"
        payoffs = np.zeros(profiles.shape)
        for i, (o, s, reg) in enumerate(zip(
                self._offset, self._scale, self._regressors)):
            mask = profiles[..., i] > 0
            profs = profiles[mask]
            profs[:, i] -= 1
            if profs.size:
                payoffs[mask, i] = reg.predict(subgame.translate(
                    profs, self._sub_mask)).ravel() * s + o
        return payoffs

    def get_dev_payoffs(self, profiles):
        """Compute the payoff for deviating

        This implementation is more efficient than the default since we don't
        need to compute the payoff for non deviators."""
        prof_view = np.rollaxis(subgame.translate(profiles.reshape(
            (-1, self.num_roles, self.num_strats)), self._sub_mask), 1, 0)
        payoffs = np.empty(profiles.shape[:-2] + (self.num_strats,))
        pay_view = payoffs.reshape((-1, self.num_strats)).T
        for pays, profs, reg in zip(
                pay_view, utils.repeat(prof_view, self.num_role_strats),
                self._regressors):
            np.copyto(pays, reg.predict(profs))
        return payoffs * self._scale + self._offset

    def max_strat_payoffs(self):
        return self._max_payoffs.view()

    def min_strat_payoffs(self):
        return self._min_payoffs.view()

    def subgame(self, sub_mask):
        base = super().subgame(sub_mask)
        new_mask = self._sub_mask.copy()
        new_mask[new_mask] = sub_mask
        regs = tuple(reg for reg, m in zip(self._regressors, sub_mask) if m)
        return DevRegressionGame(
            base, regs, self._offset[sub_mask], self._scale[sub_mask],
            self._min_payoffs[sub_mask],
            self._max_payoffs[sub_mask], new_mask)

    def normalize(self):
        scale = (self.max_role_payoffs() - self.min_role_payoffs())
        scale[np.isclose(scale, 0)] = 1
        scale = scale.repeat(self.num_role_strats)
        offset = self.min_role_payoffs().repeat(self.num_role_strats)
        return DevRegressionGame(
            self, self._regressors, (self._offset - offset) / scale,
            self._scale / scale, (self._min_payoffs - offset) / scale,
            (self._max_payoffs - offset) / scale, self._sub_mask)

    def __eq__(self, other):
        return (super().__eq__(other) and
                self._regressors == other._regressors and
                np.allclose(self._offset, other._offset) and
                np.allclose(self._scale, other._scale) and
                np.all(self._sub_mask == other._sub_mask))

    def __hash__(self):
        return super().__hash__()


def _dev_profpay(game):
    """Iterate over deviation profiles and payoffs"""
    sgame = paygame.samplegame_copy(game)
    profiles = sgame.flat_profiles()
    payoffs = sgame.flat_payoffs()

    for i, pays in enumerate(payoffs.T):
        mask = (profiles[:, i] > 0) & ~np.isnan(pays)
        assert mask.any(), \
            "couldn't find deviation data for a strategy"
        profs = profiles[mask]
        profs[:, i] -= 1
        yield i, profs, pays[mask]


# FIXME Remove train, add deprecated version for backward compatibility
def nngame_train(game, epochs=100, layer_sizes=(32, 32), dropout=0.2,
                 verbosity=0, optimizer='sgd', loss='mean_squared_error'):
    """Train a neural network regression model

    This mostly exists as a proof of concept, individual testing should be done
    to make sure it is working sufficiently. This API will likely change to
    support more general architectures and training.
    """
    assert layer_sizes, "must have at least one layer"
    assert 0 <= dropout < 1, "dropout must be a valid probability"
    # This is for delayed importing inf tensor flow
    from keras import models, layers

    model = models.Sequential()
    lay_iter = iter(layer_sizes)
    model.add(layers.Dense(
        next(lay_iter), input_shape=[game.num_strats], activation='relu'))
    for units in lay_iter:
        model.add(layers.Dense(units, activation='relu'))
        if dropout:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))

    regs = []
    offsets = np.empty(game.num_strats)
    scales = np.empty(game.num_strats)
    for i, profs, pays in _dev_profpay(game):
        # XXX Payoff normalization specific to sigmoid. If we accept alternate
        # models, we need a way to compute how to potentially normalize
        # payoffs.
        min_pay = pays.min()
        offsets[i] = min_pay
        max_pay = pays.max()
        scale = 1 if np.isclose(max_pay, min_pay) else max_pay - min_pay
        scales[i] = scale
        reg = models.clone_model(model)
        reg.compile(optimizer=optimizer, loss=loss)
        reg.fit(profs, (pays - min_pay) / scale, epochs=epochs,
                verbose=verbosity)
        regs.append(reg)

    return DevRegressionGame(
        game, tuple(regs), offsets, scales, game.min_strat_payoffs(),
        game.max_strat_payoffs(), np.ones(game.num_strats, bool))


def sklgame_train(game, estimator):
    """Create a regression game from an arbitrary sklearn estimator

    Parameters
    ----------
    game : RsGame
        The game to learn, must have at least one payoff per strategy.
    estimator : sklearn estimator
        An estimator that supports clone, fit, and predict via the stand
        scikit-learn estimator API.
    """
    regs = []
    for i, profs, pays in _dev_profpay(game):
        reg = sklearn.base.clone(estimator)
        reg.fit(profs, pays)
        regs.append(reg)
    return DevRegressionGame(
        game, tuple(regs), np.zeros(game.num_strats), np.ones(game.num_strats),
        game.min_strat_payoffs(), game.max_strat_payoffs(),
        np.ones(game.num_strats, bool))


class RbfGpGame(rsgame.CompleteGame):
    """A regression game using RBF Gaussian processes

    This regression game has a build in deviation payoff based off of a
    continuous approximation of the multinomial distribution.
    """

    def __init__(self, game, offset, min_payoffs, max_payoffs, coefs, lengths,
                 train_data, alphas, zero_diags):
        super().__init__(game.role_names, game.strat_names,
                         game.num_role_players)
        self._offset = offset
        self._offset.setflags(write=False)
        self._coefs = coefs
        self._coefs.setflags(write=False)
        self._lengths = lengths
        self._lengths.setflags(write=False)
        self._train_data = train_data
        self._alphas = alphas
        self._zero_diags = zero_diags
        self._min_payoffs = min_payoffs
        self._min_payoffs.setflags(write=False)
        self._max_payoffs = max_payoffs
        self._max_payoffs.setflags(write=False)

        self._dev_players = self.num_role_players - np.eye(
            self.num_roles, dtype=int)
        self._dev_players.setflags(write=False)

    def get_payoffs(self, profiles):
        assert self.is_profile(profiles).all(), "must pass valid profiles"
        payoffs = np.zeros(profiles.shape)
        for s, (off, coef, length, data, alpha, zero) in enumerate(zip(
                self._offset, self._coefs, self._lengths, self._train_data,
                self._alphas, self._zero_diags)):
            mask = profiles[..., s] > 0
            profs = profiles[mask]
            profs[:, s] -= 1
            if profs.size:
                vec = (profs[:, None] - data) / length
                rbf = zero + np.einsum('ijk,ijk->ij', vec, vec)
                payoffs[mask, s] = coef * np.exp(-rbf / 2).dot(alpha) + off
        return payoffs

    def get_dev_payoffs(self, profiles, *, jacobian=False):
        prof_view = np.rollaxis(profiles.reshape(
            (-1, self.num_roles, self.num_strats)), 1, 0)
        pay_view = np.empty((self.num_strats, prof_view.shape[1]))
        if jacobian:
            jac_view = np.empty((self.num_strats, prof_view.shape[1],
                                 self.num_strats))
        else:
            jac_view = itertools.repeat(None, self.num_strats)

        for pays, profs, jac, length, data, alpha, zero in zip(
                pay_view, utils.repeat(prof_view, self.num_role_strats),
                jac_view, self._lengths, self._train_data, self._alphas,
                self._zero_diags):
            vec = (profs[:, None] - data) / length
            rbf = zero + np.einsum('ijk,ijk->ij', vec, vec)
            exp = np.exp(-rbf / 2)
            exp.dot(alpha, out=pays)
            if jacobian:
                np.einsum('j,ij,ijk->ik', alpha, exp, vec / length, out=jac)

        payoffs = (
            pay_view.T.reshape((profiles.shape[:-2] + (self.num_strats,))) *
            self._coefs + self._offset)

        if jacobian:
            jac = -np.rollaxis(jac_view, 1, 0).reshape(
                (profiles.shape[:-2] + (self.num_strats,) * 2))
            return payoffs, jac * self._coefs[:, None]
        else:
            return payoffs

    def max_strat_payoffs(self):
        return self._max_payoffs.view()

    def min_strat_payoffs(self):
        return self._min_payoffs.view()

    def deviation_payoffs(self, mix, *, jacobian=False):
        payoffs = np.empty(self.num_strats)
        if jacobian:
            jac = np.empty((self.num_strats,) * 2)
        for i, (players, scale, x, kinvy, zdiag) in enumerate(zip(
                self._dev_players.repeat(self.num_role_strats, 0),
                self._lengths, self._train_data, self._alphas,
                self._zero_diags)):
            avg_prof = players.repeat(self.num_role_strats) * mix
            diag = 1 / (scale ** 2 + avg_prof)
            diff = x - avg_prof
            det = 1 / (1 - players * np.add.reduceat(
                mix ** 2 * diag, self.role_starts))
            cov_diag = np.einsum('ij,ij,j->i', diff, diff, diag) + zdiag
            cov_outer = np.add.reduceat(
                mix * diag * diff, self.role_starts, 1)
            exp = np.exp(-.5 * (cov_diag + np.einsum(
                'j,j,ij,ij->i', players, det, cov_outer, cov_outer)))
            coef = np.exp(np.log(scale).sum() + .5 * np.log(diag).sum() +
                          .5 * np.log(det).sum())
            avg = kinvy.dot(exp)
            payoffs[i] = coef * avg

            if jacobian:
                beta = 1 - players.repeat(self.num_role_strats) * mix * diag
                jac_coef = ((beta ** 2 - 1) * det.repeat(self.num_role_strats)
                            * avg)
                delta = np.repeat(cov_outer * det, self.num_role_strats, 1)
                jac_exp = exp[:, None] * (
                    (delta - 1) ** 2 - (delta * beta - diff * diag - 1) ** 2)
                jac_avg = (players.repeat(self.num_role_strats) *
                           kinvy.dot(jac_exp))
                jac[i] = -0.5 * coef * (jac_coef + jac_avg)
                # Normalize jacobian to be in mixture subspace
                jac[i] -= np.repeat(
                    np.add.reduceat(jac[i], self.role_starts) /
                    self.num_role_strats, self.num_role_strats)

        payoffs *= self._coefs
        payoffs += self._offset
        if jacobian:
            jac *= self._coefs[:, None]
            return payoffs, jac
        else:
            return payoffs

    # TODO Add function that creates sample game which draws payoffs from the
    # gp distribution

    def subgame(self, sub_mask):
        sub_mask = np.asarray(sub_mask, bool)
        base = super().subgame(sub_mask)
        data = tuple(
            d[:, sub_mask] for d, m in zip(self._train_data, sub_mask) if m)
        alphas = tuple(a for a, m in zip(self._alphas, sub_mask) if m)
        zero_diags = tuple(
            c + np.einsum(
                'ij,ij,j,j->i', d[:, ~sub_mask], d[:, ~sub_mask], s, s)
            for c, d, s, m
            in zip(self._zero_diags, self._train_data,
                   1 / self._lengths[:, ~sub_mask], sub_mask) if m)
        return RbfGpGame(
            base, self._offset[sub_mask], self._min_payoffs[sub_mask],
            self._max_payoffs[sub_mask], self._coefs[sub_mask],
            self._lengths[sub_mask][:, sub_mask], data, alphas, zero_diags)

    def normalize(self):
        scale = (self.max_role_payoffs() - self.min_role_payoffs())
        scale[np.isclose(scale, 0)] = 1
        scale = scale.repeat(self.num_role_strats)
        offset = self.min_role_payoffs().repeat(self.num_role_strats)
        return RbfGpGame(
            self, (self._offset - offset) / scale,
            (self._min_payoffs - offset) / scale,
            (self._max_payoffs - offset) / scale, self._coefs / scale,
            self._lengths, self._train_data, self._alphas, self._zero_diags)

    def __eq__(self, other):
        return (super().__eq__(other) and
                np.allclose(self._offset, other._offset) and
                np.allclose(self._coefs, other._coefs) and
                np.allclose(self._lengths, other._lengths) and
                all(np.allclose(a, b) for a, b in zip(
                    self._train_data, other._train_data)) and
                all(np.allclose(a, b) for a, b in zip(
                    self._alphas, other._alphas)) and
                all(np.allclose(a, b) for a, b in zip(
                    self._zero_diags, other._zero_diags)))

    @utils.memoize
    def __hash__(self):
        return super().__hash__()


def rbfgame_train(game, num_restarts=3):
    """Train a regression game with an RBF Gaussian process

    This model is somewhat well tests and has a few added benefits over
    standard regression models due the nature of its functional form.

    Parameters
    ----------
    game : RsGame
        The game to learn. Must have at least one payoff per strategy.
    num_restarts : int, optional
        The number of random restarts to make with the optimizer. Higher
        numbers will give a better fit (in expectation), but will take
        longer.
    """
    dev_players = np.maximum(game.num_role_players - np.eye(
        game.num_roles, dtype=int), 1).repeat(
            game.num_role_strats, 0).repeat(game.num_role_strats, 1)
    bounds = np.insert(dev_players[..., None], 0, 1, 2)
    # TODO Add an alpha that is smaller for points near the edge of the
    # simplex, accounting for the importance of minimizing error at the
    # extrema.
    means = np.empty(game.num_strats)
    coefs = np.empty(game.num_strats)
    lengths = np.empty((game.num_strats, game.num_strats))
    train_data = []
    alphas = []
    mins = np.empty(game.num_strats)
    maxs = np.empty(game.num_strats)
    for (s, profs, pays), bound, dplay in zip(
            _dev_profpay(game), bounds, dev_players):
        pay_mean = pays.mean()
        pays -= pay_mean
        reg = gp.GaussianProcessRegressor(
            1.0 * gp.kernels.RBF(bound.mean(1), bound) +
            gp.kernels.WhiteKernel(1), n_restarts_optimizer=num_restarts,
            copy_X_train=False)
        reg.fit(profs, pays)

        means[s] = pay_mean
        coefs[s] = reg.kernel_.k1.k1.constant_value
        lengths[s] = reg.kernel_.k1.k2.length_scale
        # TODO If these scales are at the boundary (1 or dev_players) then it's
        # likely a poor fit and we should warn...
        train_data.append(profs)
        alphas.append(reg.alpha_)

        minw = np.exp(-.5 * np.einsum('i,i,i', dplay, dplay, 1 / lengths[s]))
        pos = reg.alpha_[reg.alpha_ > 0].sum()
        neg = reg.alpha_[reg.alpha_ < 0].sum()
        maxs[s] = coefs[s] * (pos + minw * neg) + pay_mean
        mins[s] = coefs[s] * (minw * pos + neg) + pay_mean

    return RbfGpGame(
        game, means, mins, maxs, coefs, lengths, tuple(train_data),
        tuple(alphas), (0,) * game.num_strats)


class _DeviationGame(rsgame.CompleteGame):
    """A game that adds deviation payoffs"""

    def __init__(self, model_game):
        super().__init__(model_game.role_names, model_game.strat_names,
                         model_game.num_role_players)
        assert model_game.is_complete()
        self._model = model_game

    def get_payoffs(self, profiles):
        return self._model.get_payoffs(profiles)

    def profiles(self):
        return self._model.profiles()

    def payoffs(self):
        return self._model.payoffs()

    def max_strat_payoffs(self):
        return self._model.max_strat_payoffs()

    def min_strat_payoffs(self):
        return self._model.min_strat_payoffs()


class SampleDeviationGame(_DeviationGame):
    """Deviation payoffs by sampling from mixture

    This model produces unbiased deviation payoff estimates, but they're noisy
    and random and take a while to compute. This is accurate in the limit as
    `num_samples` goes to infinity.

    Parameters
    ----------
    model : DevRegressionGame
        A payoff model
    num_samples : int, optional
        The number of samples to use for each deviation estimate. Higher means
        lower variance but higher computation time.
    """

    def __init__(self, model, num_samples=100):
        super().__init__(model)
        assert num_samples > 0
        # TODO It might be interesting to play with a sample schedule, i.e.
        # change the number of samples based off of the query number to
        # deviation payoffs (i.e. reduce variance as we get close to
        # convergence)
        self._num_samples = num_samples

    def deviation_payoffs(self, mix, *, jacobian=False):
        """Compute the deivation payoffs

        The method computes the jacobian as if we were importance sampling the
        results, i.e. the function is really always sample according to mixture
        m', but then importance sample to get the actual result."""
        profs = self.random_role_deviation_profiles(self._num_samples, mix)
        payoffs = self._model.get_dev_payoffs(profs)
        dev_pays = payoffs.mean(0)
        if not jacobian:
            return dev_pays

        weights = np.repeat(
            np.where(np.isclose(mix, 0), 0, profs / (mix + _TINY)),
            self.num_role_strats, 1)
        jac = np.einsum('ij,ijk->jk', payoffs, weights) / self._num_samples
        jac -= np.repeat(np.add.reduceat(jac, self.role_starts, 1) /
                         self.num_role_strats, self.num_role_strats, 1)
        return dev_pays, jac

    def subgame(self, sub_mask):
        return SampleDeviationGame(self._model.subgame(sub_mask),
                                   self._num_samples)

    def normalize(self):
        return SampleDeviationGame(self._model.normalize(), self._num_samples)


def sample(game, num_samples=100):
    """Create a sample game from a model

    Parameters
    ----------
    game : RsGame
        If this is a payoff model it will be used to take samples, if this is
        an existing deviation game, then this will use it's underlying model.
    num_samples : int, optional
        The number of samples to take.
    """
    if hasattr(game, '_model'):
        game = game._model
    return SampleDeviationGame(game, num_samples=num_samples)


class PointDeviationGame(_DeviationGame):
    """Deviation payoffs by point approximation

    This model computes payoffs by finding the deviation payoffs from the point
    estimate of the mixture. It's fast but biased. This is accurate in the
    limit as the number of players goes to infinity.

    For this work, the underlying implementation of get_dev_payoffs must
    support floating point profiles, which only really makes sense for
    regression games. For deviation payoffs to have a jacobian, the underlying
    model must also support a jacobian for get_dev_payoffs.

    Parameters
    ----------
    model : DevRegressionGame
        A payoff model
    """

    def __init__(self, model):
        super().__init__(model)
        self._dev_players = np.repeat(self.num_role_players - np.eye(
            self.num_roles, dtype=int), self.num_role_strats, 1)

    def deviation_payoffs(self, mix, *, jacobian=False):
        if jacobian:
            dev, jac = self._model.get_dev_payoffs(
                self._dev_players * mix, jacobian=True)
            jac *= self._dev_players.repeat(self.num_role_strats, 0)
            jac -= np.repeat(np.add.reduceat(jac, self.role_starts, 1) /
                             self.num_role_strats, self.num_role_strats, 1)
            return dev, jac
        else:
            return self._model.get_dev_payoffs(self._dev_players * mix)

    def subgame(self, sub_mask):
        return PointDeviationGame(self._model.subgame(sub_mask))

    def normalize(self):
        return PointDeviationGame(self._model.normalize())


def point(game):
    """Create a point game from a model

    Parameters
    ----------
    game : RsGame
        If this is a payoff model it will be used to take samples, if this is
        an existing deviation game, then this will use it's underlying model.
    """
    if hasattr(game, '_model'):
        game = game._model
    return PointDeviationGame(game)


class NeighborDeviationGame(_DeviationGame):
    """Create a neighbor game from a model

    This takes a normalized weighted estimate of the deviation payoffs by
    finding all profiles within `num_devs` of the maximum probability profile
    for the mixture and weighting them accordingly. This is biased, but
    accurate in the limit as `num_devs` approaches `num_players`. It also
    produces discontinuities every time the maximum probability profile
    switches.

    Parameters
    ----------
    game : RsGame
        If this is a payoff model it will be used to take samples, if this is
        an existing deviation game, then this will use it's underlying model.
    num_devs : int, optional
        The number of deviations to take.
    """

    def __init__(self, model, num_devs=2):
        super().__init__(model)
        assert 0 <= num_devs
        self._num_devs = num_devs

    def deviation_payoffs(self, mix, *, jacobian=False):
        # TODO This is not smooth because there are discontinuities when the
        # maximum probability profile jumps at the boundary. If we wanted to
        # make it smooth, one option would be to compute the smoother
        # interpolation between this and lower probability profiles. All we
        # need to ensure smoothness is that the weight at profile
        # discontinuities is 0.
        profiles = self.nearby_profiles(
            self.max_prob_prof(mix), self._num_devs)
        payoffs = self.get_payoffs(profiles)
        game = paygame.game_replace(self, profiles, payoffs)
        return game.deviation_payoffs(mix, ignore_incomplete=True,
                                      jacobian=jacobian)

    def subgame(self, sub_mask):
        return NeighborDeviationGame(
            self._model.subgame(sub_mask), self._num_devs)

    def normalize(self):
        return NeighborDeviationGame(self._model.normalize(), self._num_devs)


def neighbor(game, num_devs=2):
    """Create a neighbor game from a model

    Parameters
    ----------
    game : RsGame
        If this is a payoff model it will be used to take samples, if this is
        an existing deviation game, then this will use it's underlying model.
    num_devs : int, optional
        The number of deviations to explore out.
    """
    if hasattr(game, '_model'):
        game = game._model
    return NeighborDeviationGame(game, num_devs=num_devs)
