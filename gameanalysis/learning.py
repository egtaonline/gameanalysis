"""Package for learning complete games from data

The API of this individual module is still unstable and may change as
improvements or refinements are made.

There are two general game types in this module: learned games and deviation
games. Learned games vary by the method, but generally expose methods for
computing payoffs and may other features. Deviation games use learned games and
different functions to compute deviation payoffs via various methods.
"""
import warnings

import numpy as np
import sklearn
from sklearn import gaussian_process as gp

from gameanalysis import gamereader
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils


class DevRegressionGame(rsgame.CompleteGame):
    """A game regression model that learns deviation payoffs

    This model functions as a game, but doesn't have a default way of computing
    deviation payoffs. It must be wrapped with another game that uses payoff
    data to compute deviation payoffs.
    """

    def __init__(self, game, regressors, offset, scale, min_payoffs,
                 max_payoffs, rest):
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
        self._rest = rest
        self._rest.setflags(write=False)

    def deviation_payoffs(self, mix, *, jacobian=False):
        raise ValueError(
            "regression games don't define deviation payoffs and must be used "
            "as a model for a deviation game")

    def get_payoffs(self, profiles):
        assert self.is_profile(profiles).all(), "must pass valid profiles"
        payoffs = np.zeros(profiles.shape)
        for i, (o, s, reg) in enumerate(zip(
                self._offset, self._scale, self._regressors)):
            mask = profiles[..., i] > 0
            profs = profiles[mask]
            profs[:, i] -= 1
            if profs.size:
                payoffs[mask, i] = reg.predict(restrict.translate(
                    profs, self._rest)).ravel() * s + o
        return payoffs

    def get_dev_payoffs(self, profiles):
        """Compute the payoff for deviating

        This implementation is more efficient than the default since we don't
        need to compute the payoff for non deviators."""
        prof_view = np.rollaxis(restrict.translate(profiles.reshape(
            (-1, self.num_roles, self.num_strats)), self._rest), 1, 0)
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

    def restrict(self, rest):
        base = rsgame.emptygame_copy(self).restrict(rest)
        new_rest = self._rest.copy()
        new_rest[new_rest] = rest
        regs = tuple(reg for reg, m in zip(self._regressors, rest) if m)
        return DevRegressionGame(
            base, regs, self._offset[rest], self._scale[rest],
            self._min_payoffs[rest],
            self._max_payoffs[rest], new_rest)

    def normalize(self):
        scale = (self.max_role_payoffs() - self.min_role_payoffs())
        scale[np.isclose(scale, 0)] = 1
        scale = scale.repeat(self.num_role_strats)
        offset = self.min_role_payoffs().repeat(self.num_role_strats)
        return DevRegressionGame(
            self, self._regressors, (self._offset - offset) / scale,
            self._scale / scale, (self._min_payoffs - offset) / scale,
            (self._max_payoffs - offset) / scale, self._rest)

    def __eq__(self, other):
        return (super().__eq__(other) and
                self._regressors == other._regressors and
                np.allclose(self._offset, other._offset) and
                np.allclose(self._scale, other._scale) and
                np.all(self._rest == other._rest))

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

    def __init__(self, role_names, strat_names, num_role_players, offset,
                 coefs, lengths, sizes, profiles, alpha):
        super().__init__(role_names, strat_names, num_role_players)
        self._offset = offset
        self._offset.setflags(write=False)
        self._coefs = coefs
        self._coefs.setflags(write=False)
        self._lengths = lengths
        self._lengths.setflags(write=False)
        self._sizes = sizes
        self._sizes.setflags(write=False)
        self._size_starts = np.insert(self._sizes[:-1].cumsum(), 0, 0)
        self._size_starts.setflags(write=False)
        self._profiles = profiles
        self._profiles.setflags(write=False)
        self._alpha = alpha
        self._alpha.setflags(write=False)

        # Useful member
        self._dev_players = np.repeat(
            self.num_role_players - np.eye(self.num_roles, dtype=int),
            self.num_role_strats, 0)
        self._dev_players.setflags(write=False)

        # Compute min and max payoffs
        # TODO These are pretty conservative, and could maybe be made more
        # accurate
        sdp = self._dev_players.repeat(self.num_role_strats, 1)
        max_rbf = np.einsum('ij,ij,ij->i', sdp, sdp, 1 / self._lengths)
        minw = np.exp(-max_rbf / 2)
        mask = self._alpha > 0
        pos = np.add.reduceat(self._alpha * mask, self._size_starts)
        neg = np.add.reduceat(self._alpha * ~mask, self._size_starts)
        self._min_payoffs = self._coefs * (pos * minw + neg) + self._offset
        self._min_payoffs.setflags(write=False)
        self._max_payoffs = self._coefs * (pos + neg * minw) + self._offset
        self._max_payoffs.setflags(write=False)

    def get_payoffs(self, profiles):
        assert self.is_profile(profiles).all(), "must pass valid profiles"
        dev_profiles = np.repeat(
            profiles[..., None, :] - np.eye(self.num_strats, dtype=int),
            self._sizes, -2)
        vec = ((dev_profiles - self._profiles) /
               self._lengths.repeat(self._sizes, 0))
        rbf = np.einsum('...ij,...ij->...i', vec, vec)
        payoffs = self._offset + self._coefs * np.add.reduceat(
            np.exp(-rbf / 2) * self._alpha, self._size_starts, -1)
        payoffs[profiles == 0] = 0
        return payoffs

    def get_dev_payoffs(self, profiles, *, jacobian=False):
        dev_profiles = profiles.repeat(
            np.add.reduceat(self._sizes, self.role_starts), -2)
        vec = ((dev_profiles - self._profiles) /
               self._lengths.repeat(self._sizes, 0))
        rbf = np.einsum('...ij,...ij->...i', vec, vec)
        exp = np.exp(-rbf / 2) * self._alpha
        payoffs = self._offset + self._coefs * np.add.reduceat(
            exp, self._size_starts, -1)

        if not jacobian:
            return payoffs

        jac = -(self._coefs[:, None] / self._lengths *
                np.add.reduceat(exp[:, None] * vec, self._size_starts, 0))
        return payoffs, jac

    def max_strat_payoffs(self):
        return self._max_payoffs.view()

    def min_strat_payoffs(self):
        return self._min_payoffs.view()

    def deviation_payoffs(self, mix, *, jacobian=False):
        players = self._dev_players.repeat(self.num_role_strats, 1)
        avg_prof = players * mix
        diag = 1 / (self._lengths ** 2 + avg_prof)
        diag_sizes = diag.repeat(self._sizes, 0)
        diff = self._profiles - avg_prof.repeat(self._sizes, 0)
        det = 1 / (1 - self._dev_players * np.add.reduceat(
            mix ** 2 * diag, self.role_starts, 1))
        det_sizes = det.repeat(self._sizes, 0)
        cov_diag = np.einsum('ij,ij,ij->i', diff, diff, diag_sizes)
        cov_outer = np.add.reduceat(
            mix * diag_sizes * diff, self.role_starts, 1)
        sec_term = np.einsum(
            'ij,ij,ij,ij->i', self._dev_players.repeat(self._sizes, 0),
            det_sizes, cov_outer, cov_outer)
        exp = np.exp(-(cov_diag + sec_term) / 2)
        coef = self._lengths.prod(1) * np.sqrt(diag.prod(1) * det.prod(1))
        avg = np.add.reduceat(self._alpha * exp, self._size_starts)
        payoffs = self._coefs * coef * avg + self._offset

        if not jacobian:
            return payoffs

        beta = 1 - players * mix * diag
        jac_coef = (
            ((beta ** 2 - 1) * det.repeat(self.num_role_strats, 1) +
             players * diag) * avg[:, None])
        delta = np.repeat(cov_outer * det_sizes, self.num_role_strats, 1)
        jac_exp = -self._alpha[:, None] * exp[:, None] * (
            (delta * beta.repeat(self._sizes, 0) - diff * diag_sizes - 1) ** 2
            - (delta - 1) ** 2)
        jac_avg = (players * np.add.reduceat(jac_exp, self._size_starts, 0))
        jac = -self._coefs[:, None] * coef[:, None] * (jac_coef + jac_avg) / 2
        return payoffs, jac

    # TODO Add function that creates sample game which draws payoffs from the
    # gp distribution

    def restrict(self, rest):
        rest = np.asarray(rest, bool)
        base = rsgame.emptygame_copy(self).restrict(rest)

        size_mask = rest.repeat(self._sizes)
        sizes = self._sizes[rest]
        profiles = self._profiles[size_mask]
        lengths = self._lengths[rest]
        zeros = (profiles[:, ~rest] /
                 lengths[:, ~rest].repeat(sizes, 0))
        removed = np.exp(-np.einsum('ij,ij->i', zeros, zeros) / 2)
        new_profs, inds = utils.unique_axis(
            np.concatenate([np.arange(rest.sum()).repeat(sizes)[:, None],
                            profiles[:, rest]], 1),
            return_inverse=True)
        new_alpha = np.bincount(inds, removed * self._alpha[size_mask])
        new_sizes = np.diff(np.concatenate([
            [-1], np.flatnonzero(np.diff(new_profs[:, 0])),
            [new_alpha.size - 1]]))

        return RbfGpGame(
            base.role_names, base.strat_names, base.num_role_players,
            self._offset[rest], self._coefs[rest],
            lengths[:, rest], new_sizes, new_profs[:, 1:], new_alpha)

    def normalize(self):
        scale = (self.max_role_payoffs() - self.min_role_payoffs())
        scale[np.isclose(scale, 0)] = 1
        scale = scale.repeat(self.num_role_strats)
        offset = self.min_role_payoffs().repeat(self.num_role_strats)
        return RbfGpGame(
            self.role_names, self.strat_names, self.num_role_players,
            (self._offset - offset) / scale, self._coefs / scale,
            self._lengths, self._sizes, self._profiles, self._alpha)

    def to_json(self):
        base = super().to_json()
        base['offsets'] = self.payoff_to_json(self._offset)
        base['coefs'] = self.payoff_to_json(self._coefs)

        lengths = {}
        for role, strats, lens in zip(
                self.role_names, self.strat_names,
                np.split(self._lengths, self.role_starts[1:])):
            lengths[role] = {s: self.payoff_to_json(l)
                             for s, l in zip(strats, lens)}
        base['lengths'] = lengths

        profs = {}
        for r, (role, strats, data) in enumerate(zip(
                self.role_names, self.strat_names,
                np.split(np.split(self._profiles, self._size_starts[1:]),
                         self.role_starts[1:]))):
            profs[role] = {strat: [self.profile_to_json(p) for p in dat]
                           for strat, dat in zip(strats, data)}
        base['profiles'] = profs

        alphas = {}
        for role, strats, alphs in zip(
                self.role_names, self.strat_names,
                np.split(np.split(self._alpha, self._size_starts[1:]),
                         self.role_starts[1:])):
            alphas[role] = {s: a.tolist() for s, a in zip(strats, alphs)}
        base['alphas'] = alphas

        base['type'] = 'rbf.1'
        return base

    def __eq__(self, other):
        if not (super().__eq__(other) and
                np.allclose(self._offset, other._offset) and
                np.allclose(self._coefs, other._coefs) and
                np.allclose(self._lengths, other._lengths) and
                np.all(self._sizes == other._sizes)):
            return False

        orda = np.lexsort(np.concatenate([
            np.arange(self.num_strats).repeat(self._sizes)[None],
            self._profiles.T]))
        ordb = np.lexsort(np.concatenate([
            np.arange(other.num_strats).repeat(other._sizes)[None],
            other._profiles.T]))
        return (np.all(self._profiles[orda] == other._profiles[ordb]) and
                np.allclose(self._alpha[orda], other._alpha[ordb]))

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self._sizes.tobytes()))


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
    profiles = []
    alpha = []
    sizes = []
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
        uprofs, inds = utils.unique_axis(
            profs, return_inverse=True)
        profiles.append(uprofs)
        alpha.append(np.bincount(inds, reg.alpha_))
        sizes.append(uprofs.shape[0])

    if np.any(lengths[..., None] == bounds):
        warnings.warn("some lengths were at their bounds, "
                      "this may indicate a poor fit")

    return RbfGpGame(
        game.role_names, game.strat_names, game.num_role_players, means, coefs,
        lengths, np.array(sizes), np.concatenate(profiles),
        np.concatenate(alpha))


def rbfgame_json(json):
    """Read an rbf game from json"""
    assert json['type'].split('.', 1)[0] == 'rbf', \
        "incorrect type"
    base = rsgame.emptygame_json(json)

    offsets = base.payoff_from_json(json['offsets'])
    coefs = base.payoff_from_json(json['coefs'])

    lengths = np.empty((base.num_strats,) * 2)
    for role, strats in json['lengths'].items():
        for strat, pay in strats.items():
            ind = base.role_strat_index(role, strat)
            base.payoff_from_json(pay, lengths[ind])

    profiles = [None] * base.num_strats
    for role, strats in json['profiles'].items():
        for strat, profs in strats.items():
            ind = base.role_strat_index(role, strat)
            profiles[ind] = np.stack([
                base.profile_from_json(p, verify=False) for p in profs])

    alphas = [None] * base.num_strats
    for role, strats in json['alphas'].items():
        for strat, alph in strats.items():
            ind = base.role_strat_index(role, strat)
            alphas[ind] = np.array(alph)

    sizes = np.fromiter(  # pragma: no branch
        (a.size for a in alphas), int, base.num_strats)

    return RbfGpGame(
        base.role_names, base.strat_names, base.num_role_players, offsets,
        coefs, lengths, sizes, np.concatenate(profiles),
        np.concatenate(alphas))


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

    def to_json(self):
        base = super().to_json()
        base['model'] = self._model.to_json()
        return base


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

        supp = mix > 0
        weights = np.zeros(profs.shape)
        weights[..., supp] = profs[..., supp] / mix[supp]
        jac = np.einsum('ij,ijk->jk', payoffs, weights.repeat(
            self.num_role_strats, 1)) / self._num_samples
        return dev_pays, jac

    def restrict(self, rest):
        return SampleDeviationGame(self._model.restrict(rest),
                                   self._num_samples)

    def normalize(self):
        return SampleDeviationGame(self._model.normalize(), self._num_samples)

    def to_json(self):
        base = super().to_json()
        base['samples'] = self._num_samples
        base['type'] = 'sample.1'
        return base


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


def sample_json(json):
    """Read sample game from json"""
    assert json['type'].split('.', 1)[0] == 'sample', \
        "incorrect type"
    return SampleDeviationGame(gamereader.loadj(json['model']),
                               num_samples=json['samples'])


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
            return dev, jac
        else:
            return self._model.get_dev_payoffs(self._dev_players * mix)

    def restrict(self, rest):
        return PointDeviationGame(self._model.restrict(rest))

    def normalize(self):
        return PointDeviationGame(self._model.normalize())

    def to_json(self):
        base = super().to_json()
        base['type'] = 'point.1'
        return base


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


def point_json(json):
    """Read point game from json"""
    assert json['type'].split('.', 1)[0] == 'point', \
        "incorrect type"
    return PointDeviationGame(gamereader.loadj(json['model']))


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

    def restrict(self, rest):
        return NeighborDeviationGame(
            self._model.restrict(rest), self._num_devs)

    def normalize(self):
        return NeighborDeviationGame(self._model.normalize(), self._num_devs)

    def to_json(self):
        base = super().to_json()
        base['devs'] = self._num_devs
        base['type'] = 'neighbor.1'
        return base


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


def neighbor_json(json):
    """Read neighbor game from json"""
    assert json['type'].split('.', 1)[0] == 'neighbor', \
        "incorrect type"
    return NeighborDeviationGame(gamereader.loadj(json['model']),
                                 num_devs=json['devs'])
