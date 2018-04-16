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
from numpy.lib import recfunctions
import sklearn
from sklearn import gaussian_process as gp

from gameanalysis import gamereader
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils


class _DevRegressionGame(rsgame._CompleteGame): # pylint: disable=protected-access
    """A game regression model that learns deviation payoffs

    This model functions as a game, but doesn't have a default way of computing
    deviation payoffs. It must be wrapped with another game that uses payoff
    data to compute deviation payoffs.
    """

    def __init__( # pylint: disable=too-many-arguments
            self, game, regressors, offset, scale, min_payoffs, max_payoffs,
            rest):
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

    def deviation_payoffs(self, _, **_kw): # pylint: disable=arguments-differ
        raise ValueError(
            "regression games don't define deviation payoffs and must be "
            'used as a model for a deviation game')

    def get_payoffs(self, profiles):
        utils.check(
            self.is_profile(profiles).all(), 'must pass valid profiles')
        payoffs = np.zeros(profiles.shape)
        for i, (off, scale, reg) in enumerate(zip(
                self._offset, self._scale, self._regressors)):
            mask = profiles[..., i] > 0
            profs = profiles[mask]
            profs[:, i] -= 1
            if profs.size:
                payoffs[mask, i] = reg.predict(restrict.translate(
                    profs, self._rest)).ravel() * scale + off
        return payoffs

    def get_dev_payoffs(self, dev_profs):
        """Compute the payoff for deviating

        This implementation is more efficient than the default since we don't
        need to compute the payoff for non deviators."""
        prof_view = np.rollaxis(restrict.translate(dev_profs.reshape(
            (-1, self.num_roles, self.num_strats)), self._rest), 1, 0)
        payoffs = np.empty(dev_profs.shape[:-2] + (self.num_strats,))
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

    def restrict(self, restriction):
        base = rsgame.empty_copy(self).restrict(restriction)
        new_rest = self._rest.copy()
        new_rest[new_rest] = restriction
        regs = tuple(reg for reg, m in zip(self._regressors, restriction) if m)
        return _DevRegressionGame(
            base, regs, self._offset[restriction], self._scale[restriction],
            self._min_payoffs[restriction], self._max_payoffs[restriction],
            new_rest)

    def _add_constant(self, constant):
        off = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        return _DevRegressionGame(
            self, self._regressors, self._offset + off, self._scale,
            self._min_payoffs + off, self._max_payoffs + off, self._rest)

    def _multiply_constant(self, constant):
        mul = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        return _DevRegressionGame(
            self, self._regressors, self._offset * mul, self._scale * mul,
            self._min_payoffs * mul, self._max_payoffs * mul, self._rest)

    def _add_game(self, _):
        return NotImplemented

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return (super().__eq__(othr) and
                self._regressors == othr._regressors and
                np.allclose(self._offset, othr._offset) and
                np.allclose(self._scale, othr._scale) and
                np.all(self._rest == othr._rest))

    def __hash__(self):
        return hash((super().__hash__(), self._rest.tobytes()))


def _dev_profpay(game):
    """Iterate over deviation profiles and payoffs"""
    sgame = paygame.samplegame_copy(game)
    profiles = sgame.flat_profiles()
    payoffs = sgame.flat_payoffs()

    for i, pays in enumerate(payoffs.T):
        mask = (profiles[:, i] > 0) & ~np.isnan(pays)
        utils.check(
            mask.any(), "couldn't find deviation data for a strategy")
        profs = profiles[mask]
        profs[:, i] -= 1
        yield i, profs, pays[mask]


def nngame_train( # pylint: disable=too-many-arguments,too-many-locals
        game, epochs=100, layer_sizes=(32, 32), dropout=0.2, verbosity=0,
        optimizer='sgd', loss='mean_squared_error'):
    """Train a neural network regression model

    This mostly exists as a proof of concept, individual testing should be done
    to make sure it is working sufficiently. This API will likely change to
    support more general architectures and training.
    """
    utils.check(layer_sizes, 'must have at least one layer')
    utils.check(0 <= dropout < 1, 'dropout must be a valid probability')
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

    return _DevRegressionGame(
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
    for _, profs, pays in _dev_profpay(game):
        reg = sklearn.base.clone(estimator)
        reg.fit(profs, pays)
        regs.append(reg)
    return _DevRegressionGame(
        game, tuple(regs), np.zeros(game.num_strats), np.ones(game.num_strats),
        game.min_strat_payoffs(), game.max_strat_payoffs(),
        np.ones(game.num_strats, bool))


class _RbfGpGame(rsgame._CompleteGame): # pylint: disable=too-many-instance-attributes,protected-access
    """A regression game using RBF Gaussian processes

    This regression game has a build in deviation payoff based off of a
    continuous approximation of the multinomial distribution.
    """

    def __init__( # pylint: disable=too-many-locals,too-many-arguments
            self, role_names, strat_names, num_role_players, offset, coefs,
            lengths, sizes, profiles, alpha):
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
        minw = np.exp(-max_rbf / 2) # pylint: disable=invalid-unary-operand-type
        mask = self._alpha > 0
        pos = np.add.reduceat(self._alpha * mask, self._size_starts)
        neg = np.add.reduceat(self._alpha * ~mask, self._size_starts)
        self._min_payoffs = self._coefs * (pos * minw + neg) + self._offset
        self._min_payoffs.setflags(write=False)
        self._max_payoffs = self._coefs * (pos + neg * minw) + self._offset
        self._max_payoffs.setflags(write=False)

    def get_payoffs(self, profiles):
        utils.check(
            self.is_profile(profiles).all(), 'must pass valid profiles')
        dev_profiles = np.repeat(
            profiles[..., None, :] - np.eye(self.num_strats, dtype=int),
            self._sizes, -2)
        vec = ((dev_profiles - self._profiles) /
               self._lengths.repeat(self._sizes, 0))
        rbf = np.einsum('...ij,...ij->...i', vec, vec)
        payoffs = self._offset + self._coefs * np.add.reduceat(
            np.exp(-rbf / 2) * self._alpha, self._size_starts, -1) # pylint: disable=invalid-unary-operand-type
        payoffs[profiles == 0] = 0
        return payoffs

    def get_dev_payoffs(self, dev_profs, *, jacobian=False): # pylint: disable=arguments-differ
        dev_profiles = dev_profs.repeat(
            np.add.reduceat(self._sizes, self.role_starts), -2)
        vec = ((dev_profiles - self._profiles) /
               self._lengths.repeat(self._sizes, 0))
        rbf = np.einsum('...ij,...ij->...i', vec, vec)
        exp = np.exp(-rbf / 2) * self._alpha # pylint: disable=invalid-unary-operand-type
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

    def deviation_payoffs(self, mixture, *, jacobian=False, **_): # pylint: disable=too-many-locals
        players = self._dev_players.repeat(self.num_role_strats, 1)
        avg_prof = players * mixture
        diag = 1 / (self._lengths ** 2 + avg_prof)
        diag_sizes = diag.repeat(self._sizes, 0)
        diff = self._profiles - avg_prof.repeat(self._sizes, 0)
        det = 1 / (1 - self._dev_players * np.add.reduceat(
            mixture ** 2 * diag, self.role_starts, 1))
        det_sizes = det.repeat(self._sizes, 0)
        cov_diag = np.einsum('ij,ij,ij->i', diff, diff, diag_sizes)
        cov_outer = np.add.reduceat(
            mixture * diag_sizes * diff, self.role_starts, 1)
        sec_term = np.einsum(
            'ij,ij,ij,ij->i', self._dev_players.repeat(self._sizes, 0),
            det_sizes, cov_outer, cov_outer)
        exp = np.exp(-(cov_diag + sec_term) / 2)
        coef = self._lengths.prod(1) * np.sqrt(diag.prod(1) * det.prod(1))
        avg = np.add.reduceat(self._alpha * exp, self._size_starts)
        payoffs = self._coefs * coef * avg + self._offset

        if not jacobian:
            return payoffs

        beta = 1 - players * mixture * diag
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

    def restrict(self, restriction):
        restriction = np.asarray(restriction, bool)
        base = rsgame.empty_copy(self).restrict(restriction)

        size_mask = restriction.repeat(self._sizes)
        sizes = self._sizes[restriction]
        profiles = self._profiles[size_mask]
        lengths = self._lengths[restriction]
        zeros = (profiles[:, ~restriction] /
                 lengths[:, ~restriction].repeat(sizes, 0))
        removed = np.exp(-np.einsum('ij,ij->i', zeros, zeros) / 2) # pylint: disable=invalid-unary-operand-type
        uprofs, inds = np.unique(
            recfunctions.merge_arrays([
                np.arange(restriction.sum()).repeat(sizes).view([('s', int)]),
                utils.axis_to_elem(profiles[:, restriction])], flatten=True),
            return_inverse=True)
        new_alpha = np.bincount(inds, removed * self._alpha[size_mask])
        new_sizes = np.diff(np.concatenate([
            [-1], np.flatnonzero(np.diff(uprofs['s'])),
            [new_alpha.size - 1]]))

        return _RbfGpGame(
            base.role_names, base.strat_names, base.num_role_players,
            self._offset[restriction], self._coefs[restriction],
            lengths[:, restriction], new_sizes, uprofs['axis'], new_alpha)

    def _add_constant(self, constant):
        off = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        return _RbfGpGame(
            self.role_names, self.strat_names, self.num_role_players,
            self._offset + off, self._coefs, self._lengths, self._sizes,
            self._profiles, self._alpha)

    def _multiply_constant(self, constant):
        mul = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        return _RbfGpGame(
            self.role_names, self.strat_names, self.num_role_players,
            self._offset * mul, self._coefs * mul, self._lengths, self._sizes,
            self._profiles, self._alpha)

    def _add_game(self, _):
        return NotImplemented

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
        for role, strats, data in zip(
                self.role_names, self.strat_names,
                np.split(np.split(self._profiles, self._size_starts[1:]),
                         self.role_starts[1:])):
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

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return (super().__eq__(othr) and
                np.allclose(self._offset, othr._offset) and
                np.allclose(self._coefs, othr._coefs) and
                np.allclose(self._lengths, othr._lengths) and
                np.all(self._sizes == othr._sizes) and
                utils.allclose_perm(
                    np.concatenate([
                        np.arange(self.num_strats).repeat(
                            self._sizes)[:, None],
                        self._profiles, self._alpha[:, None]], 1),
                    np.concatenate([
                        np.arange(othr.num_strats).repeat(
                            othr._sizes)[:, None],
                        othr._profiles, othr._alpha[:, None]], 1)))

    @utils.memoize
    def __hash__(self):
        hprofs = np.sort(utils.axis_to_elem(np.concatenate([
            np.arange(self.num_strats).repeat(self._sizes)[:, None],
            self._profiles], 1))).tobytes()
        return hash((super().__hash__(), hprofs))


def rbfgame_train(game, num_restarts=3): # pylint: disable=too-many-locals
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
    for (strat, profs, pays), bound in zip(_dev_profpay(game), bounds):
        pay_mean = pays.mean()
        pays -= pay_mean
        reg = gp.GaussianProcessRegressor(
            1.0 * gp.kernels.RBF(bound.mean(1), bound) +
            gp.kernels.WhiteKernel(1), n_restarts_optimizer=num_restarts,
            copy_X_train=False)
        reg.fit(profs, pays)

        means[strat] = pay_mean
        coefs[strat] = reg.kernel_.k1.k1.constant_value
        lengths[strat] = reg.kernel_.k1.k2.length_scale
        uprofs, inds = np.unique(
            utils.axis_to_elem(profs), return_inverse=True)
        profiles.append(utils.axis_from_elem(uprofs))
        alpha.append(np.bincount(inds, reg.alpha_))
        sizes.append(uprofs.size)

    if np.any(lengths[..., None] == bounds):
        warnings.warn(
            'some lengths were at their bounds, this may indicate a poor '
            'fit')

    return _RbfGpGame(
        game.role_names, game.strat_names, game.num_role_players, means, coefs,
        lengths, np.array(sizes), np.concatenate(profiles),
        np.concatenate(alpha))


def rbfgame_json(json):
    """Read an rbf game from json"""
    utils.check(json['type'].split('.', 1)[0] == 'rbf', 'incorrect type')
    base = rsgame.empty_json(json)

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

    return _RbfGpGame(
        base.role_names, base.strat_names, base.num_role_players, offsets,
        coefs, lengths, sizes, np.concatenate(profiles),
        np.concatenate(alphas))


class _DeviationGame(rsgame._CompleteGame): # pylint: disable=abstract-method,protected-access
    """A game that adds deviation payoffs"""

    def __init__(self, model_game):
        super().__init__(model_game.role_names, model_game.strat_names,
                         model_game.num_role_players)
        utils.check(
            model_game.is_complete(),
            'deviation models must be complete games')
        self.model = model_game

    def get_payoffs(self, profiles):
        return self.model.get_payoffs(profiles)

    def profiles(self):
        return self.model.profiles()

    def payoffs(self):
        return self.model.payoffs()

    def max_strat_payoffs(self):
        return self.model.max_strat_payoffs()

    def min_strat_payoffs(self):
        return self.model.min_strat_payoffs()

    def to_json(self):
        base = super().to_json()
        base['model'] = self.model.to_json()
        return base

    def __eq__(self, othr):
        return (super().__eq__(othr) and
                self.model == othr.model)

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.model))


class _SampleDeviationGame(_DeviationGame):
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
        utils.check(num_samples > 0, 'num samples must be greater than 0')
        # TODO It might be interesting to play with a sample schedule, i.e.
        # change the number of samples based off of the query number to
        # deviation payoffs (i.e. reduce variance as we get close to
        # convergence)
        self.num_samples = num_samples

    def deviation_payoffs(self, mixture, *, jacobian=False, **_):
        """Compute the deivation payoffs

        The method computes the jacobian as if we were importance sampling the
        results, i.e. the function is really always sample according to mixture
        m', but then importance sample to get the actual result."""
        profs = self.random_role_deviation_profiles(self.num_samples, mixture)
        payoffs = self.model.get_dev_payoffs(profs)
        dev_pays = payoffs.mean(0)
        if not jacobian:
            return dev_pays

        supp = mixture > 0
        weights = np.zeros(profs.shape)
        weights[..., supp] = profs[..., supp] / mixture[supp]
        jac = np.einsum('ij,ijk->jk', payoffs, weights.repeat(
            self.num_role_strats, 1)) / self.num_samples
        return dev_pays, jac

    def restrict(self, restriction):
        return _SampleDeviationGame(
            self.model.restrict(restriction), self.num_samples)

    def _add_constant(self, constant):
        return _SampleDeviationGame(self.model + constant, self.num_samples)

    def _multiply_constant(self, constant):
        return _SampleDeviationGame(self.model * constant, self.num_samples)

    def _add_game(self, othr):
        try:
            assert self.num_samples == othr.num_samples
            return _SampleDeviationGame(
                self.model + othr.model, self.num_samples)
        except (AttributeError, AssertionError):
            return NotImplemented

    def to_json(self):
        base = super().to_json()
        base['samples'] = self.num_samples
        base['type'] = 'sample.1'
        return base

    def __eq__(self, othr):
        return (super().__eq__(othr) and
                self.num_samples == othr.num_samples)

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_samples))


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
    try:
        return _SampleDeviationGame(game.model, num_samples=num_samples)
    except AttributeError:
        return _SampleDeviationGame(game, num_samples=num_samples)


def sample_json(json):
    """Read sample game from json"""
    utils.check(
        json['type'].split('.', 1)[0] == 'sample', 'incorrect type')
    return _SampleDeviationGame(
        gamereader.loadj(json['model']), num_samples=json['samples'])


class _PointDeviationGame(_DeviationGame):
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

    def deviation_payoffs(self, mixture, *, jacobian=False, **_):
        if not jacobian:
            return self.model.get_dev_payoffs(self._dev_players * mixture)

        dev, jac = self.model.get_dev_payoffs(
            self._dev_players * mixture, jacobian=True)
        jac *= self._dev_players.repeat(self.num_role_strats, 0)
        return dev, jac

    def restrict(self, restriction):
        return _PointDeviationGame(self.model.restrict(restriction))

    def _add_constant(self, constant):
        return _PointDeviationGame(self.model + constant)

    def _multiply_constant(self, constant):
        return _PointDeviationGame(self.model * constant)

    def _add_game(self, othr):
        try:
            assert isinstance(othr, _PointDeviationGame)
            return _PointDeviationGame(self.model + othr.model)
        except (AttributeError, AssertionError):
            return NotImplemented

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
    try:
        return _PointDeviationGame(game.model)
    except AttributeError:
        return _PointDeviationGame(game)


def point_json(json):
    """Read point game from json"""
    utils.check(
        json['type'].split('.', 1)[0] == 'point', 'incorrect type')
    return _PointDeviationGame(gamereader.loadj(json['model']))


class _NeighborDeviationGame(_DeviationGame):
    """Create a neighbor game from a model

    This takes a normalized weighted estimate of the deviation payoffs by
    finding all profiles within `num_neighbors` of the maximum probability
    profile for the mixture and weighting them accordingly. This is biased, but
    accurate in the limit as `num_neighbors` approaches `num_players`. It also
    produces discontinuities every time the maximum probability profile
    switches.

    Parameters
    ----------
    game : RsGame
        If this is a payoff model it will be used to take samples, if this is
        an existing deviation game, then this will use it's underlying model.
    num_neighbors : int, optional
        The number of deviations to take.
    """

    def __init__(self, model, num_neighbors=2):
        super().__init__(model)
        utils.check(num_neighbors >= 0, 'num devs must be nonnegative')
        self.num_neighbors = num_neighbors

    def deviation_payoffs(self, mixture, *, jacobian=False, **_):
        # TODO This is not smooth because there are discontinuities when the
        # maximum probability profile jumps at the boundary. If we wanted to
        # make it smooth, one option would be to compute the smoother
        # interpolation between this and lower probability profiles. All we
        # need to ensure smoothness is that the weight at profile
        # discontinuities is 0.
        profiles = self.nearby_profiles(
            self.max_prob_prof(mixture), self.num_neighbors)
        payoffs = self.get_payoffs(profiles)
        game = paygame.game_replace(self, profiles, payoffs)
        return game.deviation_payoffs(mixture, ignore_incomplete=True,
                                      jacobian=jacobian)

    def restrict(self, restriction):
        return _NeighborDeviationGame(
            self.model.restrict(restriction), self.num_neighbors)

    def _add_constant(self, constant):
        return _NeighborDeviationGame(self.model + constant, self.num_neighbors)

    def _multiply_constant(self, constant):
        return _NeighborDeviationGame(self.model * constant, self.num_neighbors)

    def _add_game(self, othr):
        try:
            assert self.num_neighbors == othr.num_neighbors
            return _NeighborDeviationGame(
                self.model + othr.model, self.num_neighbors)
        except (AttributeError, AssertionError):
            return NotImplemented

    def to_json(self):
        base = super().to_json()
        base['neighbors'] = self.num_neighbors
        base['type'] = 'neighbor.2'
        return base

    def __eq__(self, othr):
        return super().__eq__(othr) and self.num_neighbors == othr.num_neighbors

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_neighbors))


def neighbor(game, num_neighbors=2):
    """Create a neighbor game from a model

    Parameters
    ----------
    game : RsGame
        If this is a payoff model it will be used to take samples, if this is
        an existing deviation game, then this will use it's underlying model.
    num_neighbors : int, optional
        The number of deviations to explore out.
    """
    try:
        return _NeighborDeviationGame(game.model, num_neighbors=num_neighbors)
    except AttributeError:
        return _NeighborDeviationGame(game, num_neighbors=num_neighbors)


def neighbor_json(json):
    """Read neighbor game from json"""
    utils.check(
        json['type'].split('.', 1)[0] == 'neighbor', 'incorrect type')
    return _NeighborDeviationGame(
        gamereader.loadj(json['model']),
        num_neighbors=json.get('neighbors', json.get('devs', None)))
