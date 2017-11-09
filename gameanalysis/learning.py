"""Package for learning complete games from data

The API of this individual module is still unstable and may change as
improvements or refinements are made.

There are two general game types in this module: learned games and deviation
games. Learned games vary by the method, but generally expose methods for
computing payoffs and may other features. Deviation games use learned games and
different functions to compute deviation payoffs via various methods.
"""
import functools

import numpy as np
import scipy.special as sps
import sklearn
from sklearn import gaussian_process as gp

from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import subgame


_TINY = np.finfo(float).tiny


class DevRegressionGame(rsgame.CompleteGame):
    """A game regression model that learns deviation payoffs

    This model functions as a game, but doesn't have a default way of computing
    deviation payoffs. It must be wrapped with another game that uses payoff
    data to compute deviation payoffs.

    Subclasses may add additional functionality.
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

    @functools.lru_cache(maxsize=1)
    def profiles(self):
        return self.all_profiles()

    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        return self.get_payoffs(self.profiles())

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

    def get_mean_dev_payoffs(self, profiles):
        """Compute the mean payoff from deviating

        Given, potentially several partial profiles per role, compute the mean
        payoff for deviating to each strategy.

        Parameters
        ----------
        profiles : array-like, shape = (num_samples, num_roles, num_strats)
            A list of partial profiles by role. This is the same structure as
            returned by `random_dev_profiles`.
        """
        profiles = subgame.translate(profiles.reshape(
            (-1, self.num_roles, self.num_strats)), self._sub_mask)
        payoffs = np.empty(self.num_strats)
        for i, (r, reg) in enumerate(zip(
                self.role_indices, self._regressors)):
            payoffs[i] = reg.predict(profiles[:, r]).mean()
        return payoffs * self._scale + self._offset

    def max_strat_payoffs(self):
        return self._max_payoffs.view()

    def min_strat_payoffs(self):
        return self._min_payoffs.view()

    # TODO There's a slightly more efficient way to get dpr payoffs than
    # naively querying all payoffs, since you only need deviations. But that's
    # mainly as a lower bound on the comparison to DPR, and so it probably
    # doesn't matter how efficient that is. We could add a method for "get dpr
    # payoffs" that would return nans for the non deviation payoffs in
    # deviation profiles.

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
                np.all(self._offset == other._offset) and
                np.all(self._scale == other._scale) and
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


class RbfGpGame(DevRegressionGame):
    """A regression game using RBF Gaussian processes

    This regression game has a build in deviation payoff based off of a
    continuous approximation of the multinomial distribution.
    """

    def __init__(self, game, regressors, offset, scale, min_payoffs,
                 max_payoffs, sub_mask, coefs, rbf_scales, train_data, alphas):
        super().__init__(game, regressors, offset, scale, min_payoffs,
                         max_payoffs, sub_mask)
        self._coefs = coefs
        self._coefs.setflags(write=False)
        self._rbf_scales = rbf_scales
        self._rbf_scales.setflags(write=False)
        self._train_data = train_data
        self._alphas = alphas

        self._dev_players = self.num_role_players - np.eye(
            self.num_roles, dtype=int)
        self._dev_players.setflags(write=False)
        self._role_starts = np.argmax(
            self.role_starts[:, None] < sub_mask.cumsum(), 1)
        self._role_starts.setflags(write=False)

    # TODO Add derivatives for other functions

    def deviation_payoffs(self, mix, *, jacobian=False):
        # FIXME Add jacobian
        assert not jacobian, "jacobian not supported yet"
        smix = subgame.translate(mix, self._sub_mask)
        payoffs = np.empty(self.num_strats)
        for i, (players, scale, x, kinvy) in enumerate(zip(
                self._dev_players.repeat(self.num_role_strats, 0),
                self._rbf_scales, self._train_data, self._alphas)):
            avg_prof = subgame.translate(
                players.repeat(self.num_role_strats) * mix,
                self._sub_mask)
            diag = scale ** 2 + avg_prof
            diff = x - avg_prof
            det = 1 - players * np.add.reduceat(
                smix ** 2 / diag, self._role_starts)
            cov_diag = np.add.reduceat(diff ** 2 / diag, self._role_starts, 1)
            cov_outer = np.add.reduceat(
                smix / diag * diff, self._role_starts, 1)
            exp = np.exp(-.5 * np.sum(
                cov_diag + players / det * cov_outer ** 2, 1))
            coef = np.exp(np.log(scale).sum() - .5 * np.log(diag).sum() -
                          .5 * np.log(det).sum())
            payoffs[i] = coef * kinvy.dot(exp)
        return payoffs * self._coefs * self._scale + self._offset

    # TODO Add function that creates sample game which draws payoffs from the
    # gp distribution

    def subgame(self, sub_mask):
        base = super().subgame(sub_mask)
        new_data = tuple(d for d, m in zip(self._train_data, sub_mask) if m)
        new_alphas = tuple(a for a, m in zip(self._alphas, sub_mask) if m)
        return RbfGpGame(
            base, base._regressors, base._offset, base._scale,
            base._min_payoffs, base._max_payoffs, base._sub_mask,
            self._coefs[sub_mask], self._rbf_scales[sub_mask], new_data,
            new_alphas)

    def normalize(self):
        base = super().normalize()
        return RbfGpGame(
            base, base._regressors, base._offset, base._scale,
            base._min_payoffs, base._max_payoffs, base._sub_mask, self._coefs,
            self._rbf_scales, self._train_data, self._alphas)


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
    # TODO As a boon to regularization, we may want our own RBF kernel that has
    # a length for each role, but not each strategy.
    regs = []
    means = np.empty(game.num_strats)
    coefs = np.empty(game.num_strats)
    rbf_scales = np.empty((game.num_strats, game.num_strats))
    train_data = []
    alphas = []
    mins = np.empty(game.num_strats)
    maxs = np.empty(game.num_strats)
    for (i, profs, pays), bound, dplay in zip(
            _dev_profpay(game), bounds, dev_players):
        pay_mean = pays.mean()
        pays -= pay_mean
        reg = gp.GaussianProcessRegressor(
            1.0 * gp.kernels.RBF(bound.mean(1), bound) +
            gp.kernels.WhiteKernel(1), n_restarts_optimizer=num_restarts,
            copy_X_train=False)
        reg.fit(profs, pays)

        means[i] = pay_mean
        coefs[i] = reg.kernel_.k1.k1.constant_value
        rbf_scales[i] = reg.kernel_.k1.k2.length_scale
        train_data.append(profs)
        alphas.append(reg.alpha_)
        regs.append(reg)

        minw = np.exp(-.5 * np.sum(dplay ** 2 / rbf_scales[i]))
        pos = reg.alpha_[reg.alpha_ > 0].sum()
        neg = reg.alpha_[reg.alpha_ < 0].sum()
        maxs[i] = coefs[i] * (pos + minw * neg) + pay_mean
        mins[i] = coefs[i] * (minw * pos + neg) + pay_mean

    return RbfGpGame(
        game, tuple(regs), means, np.ones(game.num_strats),
        mins, maxs, np.ones(game.num_strats, bool), coefs, rbf_scales,
        tuple(train_data), tuple(alphas))


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
        # TODO IT might be interesting to play with a sampel schedule, i.e.
        # change the number of samples based off of the query number to
        # deviation payoffs (i.e. reduce variance as we get close to
        # convergence)
        self._num_samples = num_samples

    def deviation_payoffs(self, mix, *, jacobian=False):
        # TODO We can compute deviations by assuming the samples we took are
        # fixed, but now we're returning an importance sampled version of them
        # from the mixture as a variable, and take the gradient with respect to
        # that
        # TODO It could be possible to set the random seed each time we get
        # samples in a way that makes this somewhat smooth.
        assert not jacobian, "SampleEVs doesn't support jacobian"
        profs = self.random_dev_profiles(mix, self._num_samples)
        return self._model.get_mean_dev_payoffs(profs)

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
        # TODO Computing this gradient should be trivial
        assert not jacobian, "PointEVs doesn't support jacobian"
        return self._model.get_mean_dev_payoffs(self._dev_players * mix)

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
        assert not jacobian, "NeighborEVs doesn't support jacobian"
        # TODO This needs to be it's own math because we re-normalize, although
        # hopefully there's some common aspect of payoff games we can exploit?
        # Maybe a `compute_dev_payoffs` which we then numerically re-normalize
        # since we know we have a complete game
        profiles = self.nearby_profiles(
            self.max_prob_prof(mix), self._num_devs)
        payoffs = self.get_payoffs(profiles)

        player_factorial = np.sum(sps.gammaln(profiles + 1), 1)[:, None]
        tot_factorial = np.sum(sps.gammaln(self.num_role_players + 1))
        log_mix = np.log(mix + _TINY)
        prof_prob = np.sum(profiles * log_mix, 1, keepdims=True)
        profile_probs = tot_factorial - player_factorial + prof_prob
        denom = log_mix + np.log(self.num_role_players).repeat(
            self.num_role_strats)
        with np.errstate(divide='ignore'):
            log_profs = np.log(profiles)
        probs = np.exp(log_profs + profile_probs - denom)
        return np.sum(payoffs * probs, 0) / probs.sum(0)

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
