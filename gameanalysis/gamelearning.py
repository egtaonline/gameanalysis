"""Package for learning complete games from data

This package is not stable"""
import itertools

import numpy as np
import scipy.special as sps
from sklearn import gaussian_process as gp

from gameanalysis import paygame
from gameanalysis import rsgame


_TINY = np.finfo(float).tiny


# FIXME I would change this interface to not exist, and to instead just have
# things that implement a function get_payoffs. Each learned game model takes
# one of these, they don't need to use inheritance as they can just rely on the
# interface. All this gives up is mean dev_payoffs which isn't significantly
# more efficient and is rarely used anyways. If it does seem important, then
# implement both methods or something. It makes the interface more clear
# and allows for arbitrary functions that can produce payoffs, particularly
# ones that can potentially account for correlation between payoffs for a
# single strategy profile.
# As a result, the EVs would just be Games that take a profile function instead
# of actual profiles and otherwise compute evs how they want. This would remove
# the need for the RegressionGame as requiring an EV.
# Also, even if this strategy isn't used, the inheritance is a little wonky as
# it depends on future constructors defining state variables instead of just
# requiring them at construction.

# TODO For learning, it might make more sense to normalize profiles by learning
# profile / game.num_players.repeat(...) so that all input values are in [0,
# 1], this also normalize impact between roles.
# TODO There's a slightly more efficient way to get dpr payoffs than naively
# querying all payoffs, since you only need deviations.
class RegressionPayoffs(object):
    def get_payoffs(self, profiles):
        """Get the payoffs for a set of profiles

        Parameters
        ----------
        profiles : ndarray, int
            The profiles to get payoff data for. Must have shape (num_samples,
            num_role_strats).
        """
        payoffs = np.zeros(profiles.shape)
        for i, model in enumerate(self.regressors):
            mask = profiles[:, i] > 0
            profs = profiles[mask]
            profs[:, i] -= 1
            if profs.size:
                payoffs[mask, i] = model.predict(profs).flat
        return payoffs

    def get_mean_dev_payoffs(self, profiles):
        """Get the mean deviation payoff over role partial profiles

        Parameters
        ----------
        profiles : ndarray
        A (num_roles, num_samples, num_role_strats) array, where the first
        dimension corresponding to the deviating role, i.e. the number of
        players in role i of dimension i should num_players[i] - 1.
        """
        payoffs = np.empty(self.game.num_strats)
        for i, (model, r) in enumerate(zip(
                self.regressors, self.game.role_indices)):
            payoffs[i] = model.predict(profiles[r]).mean()
        return payoffs

    # TODO What about regression functions that learn join payoffs, i.e. you
    # could use an nn to learn all payoffs per profile (with appropriate loss
    # function) that then allows dependency between payoffs per strategy.
    def learn_payoffs(self, profiles, payoffs, training_function,
                      training_args):
        """Trains a regression model for each strategys payoffs"""
        self.regressors = []
        valid = (profiles > 0) & ~np.isnan(payoffs)
        for rs, mask in enumerate(valid.T):
            dev_profs = profiles[mask]
            dev_profs[:, rs] -= 1
            dev_pays = payoffs[mask, rs]
            self.regressors.append(training_function(
                dev_profs, dev_pays, **training_args))


class GPPayoffs(RegressionPayoffs):
    def __init__(self, game_to_learn, **gp_args):
        self.game = rsgame.emptygame_copy(game_to_learn)
        self.learn_payoffs(game_to_learn.profiles(), game_to_learn.payoffs(),
                           _train_gp, gp_args)


class NNPayoffs(RegressionPayoffs):
    def __init__(self, game_to_learn, **nn_args):
        self.game = rsgame.emptygame_copy(game_to_learn)
        self.regressors = []
        sgame = paygame.samplegame_copy(game_to_learn)
        profiles = sgame.flat_profiles()
        payoffs = sgame.flat_payoffs()

        # normalize payoffs to [0,1]
        # TODO allow specification of an arbitrary payoff range, is this only
        # important for different final activations?
        self._scale = np.empty(self.game.num_strats)
        self._offset = np.empty(self.game.num_strats)
        for rs, (pays, mask) in enumerate(zip(payoffs.T, profiles.T > 0)):
            rs_pays = pays[mask]
            minimum = np.nanmin(rs_pays)
            maximum = np.nanmax(rs_pays)
            self._offset[rs] = minimum
            self._scale[rs] = (1 if np.isclose(maximum, minimum)
                               else maximum - minimum)
        payoffs -= self._offset
        payoffs /= self._scale
        self.learn_payoffs(profiles, payoffs, _train_nn, nn_args)

    def unnormalize(self, payoffs):
        """Return payoffs to the original scale."""
        return (payoffs * self._scale) + self._offset

    def get_payoffs(self, profiles):
        """Unnormalizes the payoffs returned by RegressionPayoffs."""
        payoffs = self.unnormalize(
            RegressionPayoffs.get_payoffs(self, profiles))
        print(profiles.shape, payoffs.shape)
        payoffs[profiles == 0] = 0
        return payoffs

    def get_mean_dev_payoffs(self, profiles):
        """Unnormalizes the deviation payoffs returned by RegressionPayoffs."""
        return self.unnormalize(RegressionPayoffs.get_mean_dev_payoffs(
            self, profiles))


# FIXME This should be removed in favor of just "copying" the payoffs from the
# game with something like game_copy
class FullGameEVs(rsgame.RsGame):
    def __init__(self, regression_model, **args):
        self.regression_model = regression_model
        self.game = self.regression_model.game
        profiles = self.game.all_profiles()
        payoffs = self.regression_model.get_payoffs(profiles)
        self.full_game = paygame.game_replace(self.game, profiles, payoffs)

    def deviation_payoffs(self, mix, *args, **kwds):
        return self.full_game.deviation_payoffs(mix, *args, **kwds)


class SampleEVs(rsgame.RsGame):
    def __init__(self, regression_model, num_samples=1000):
        self.regression_model = regression_model
        self.game = self.regression_model.game
        assert num_samples > 0
        self._num_samples = num_samples

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        assert not jacobian, "SampleEVs doesn't support jacobian"
        profs = self.game.random_dev_profiles(
            mix, self._num_samples).swapaxes(0, 1)
        return self.regression_model.get_mean_dev_payoffs(profs)


class PointEVs(rsgame.RsGame):
    def __init__(self, regression_model):
        self.regression_model = regression_model
        self.game = self.regression_model.game

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        assert not jacobian, "PointEVs doesn't support jacobian"
        dev_players = self.game.num_role_players - \
            np.eye(self.game.num_roles, dtype=int)
        dev_profs = dev_players.repeat(self.game.num_role_strats, 1) * mix
        return self.regression_model.get_mean_dev_payoffs(dev_profs[:, None])


class NeighborEVs(rsgame.RsGame):
    def __init__(self, regression_model, num_devs=2):
        self.regression_model = regression_model
        self.game = self.regression_model.game
        self._num_devs = num_devs

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        assert not jacobian, "NeighborEVs doesn't support jacobian"

        profiles = self.game.nearby_profiles(
            self.game.max_prob_prof(mix), self._num_devs)
        payoffs = self.regression_model.get_payoffs(profiles)

        player_factorial = np.sum(sps.gammaln(profiles + 1), 1)[:, None]
        tot_factorial = np.sum(sps.gammaln(self.game.num_role_players + 1))
        log_mix = np.log(mix + _TINY)
        prof_prob = np.sum(profiles * log_mix, 1, keepdims=True)
        profile_probs = tot_factorial - player_factorial + prof_prob
        denom = log_mix + \
            np.log(self.game.num_role_players).repeat(
                self.game.num_role_strats)
        with np.errstate(divide='ignore'):
            log_profs = np.log(profiles)
        probs = np.exp(log_profs + profile_probs - denom)
        return np.sum(payoffs * probs, 0) / probs.sum(0)


_REGRESSION_METHODS = {'gp': GPPayoffs, 'nn': NNPayoffs}
_EV_METHODS = {
    'full': FullGameEVs,
    'sample': SampleEVs,
    'point': PointEVs,
    'neighbor': NeighborEVs,
}


# TODO It is more consistent and easier to modify if this constructor just
# takes the data it needs, i.e. a payoffs and an ev, and then there's a
# constructor function that allows creating those with strings.
# FIXME This call also means that even when copying a game, these need to be
# specified. It is probably more clear to have this take just a regression
# method and have two static functiosn for creation. One called reggame_copy
# that requires a regression game and copies the info but potentially changes
# the ev method, while a second called reggame_learn uses the payoff data of
# the game to lean a new game. In this case you could use the payoff daya from
# one regression game to learn a new one.
class RegressionGame(rsgame.CompleteGame):
    def __init__(self, game_to_learn, regression_method, EV_method,
                 regression_args={}, EV_args={}):
        """A Game that uses a regression model to compute payoffs

        If game to learn is a RegressionGame, then the underlying model will be
        copied, otherwise a game will be learned from the data."""
        assert regression_method.lower() in _REGRESSION_METHODS, \
            "invalid regression_method: {}. options: {}".format(
                regression_method, ', '.join(_REGRESSION_METHODS))
        self.regression_method = _REGRESSION_METHODS[regression_method.lower()]
        assert EV_method.lower() in _EV_METHODS, \
            "invalid EV_method: {}. options: {}".format(
                EV_method, ', '.join(_EV_METHODS))
        self.EV_method = _EV_METHODS[EV_method.lower()]
        super().__init__(game_to_learn.role_names, game_to_learn.strat_names,
                         game_to_learn.num_role_players)

        if isinstance(game_to_learn, RegressionGame):
            # shallow copy trained models
            assert self.regression_method == game_to_learn.regression_method, \
                "can't change regression method; use original payoff data"
            self._min_payoffs = game_to_learn._min_payoffs
            self._max_payoffs = game_to_learn._max_payoffs
            self.regression_model = game_to_learn.regression_model
        else:
            # FIXME This is not the correct assertion and it should be handled
            # by each regression method independently. Also, it only existed
            # when the GP was cross validated.
            assert game_to_learn.num_complete_profiles >= 3, \
                "can't learn a game from less than 3 profiles"
            self._min_payoffs = game_to_learn.min_strat_payoffs()
            self._max_payoffs = game_to_learn.max_strat_payoffs()
            self.regression_model = self.regression_method(
                game_to_learn, **regression_args)

        self.EV_model = self.EV_method(self.regression_model, **EV_args)

    def deviation_payoffs(self, mix, *args, **kwds):
        return self.EV_model.deviation_payoffs(mix, *args, **kwds)

    def get_payoffs(self, profiles):
        payoffs = self.regression_model.get_payoffs(
            profiles.reshape((-1, self.num_strats)))
        return payoffs.reshape(profiles.shape)

    def min_strat_payoffs(self):
        return self._min_payoffs.view()

    def max_strat_payoffs(self):
        return self._max_payoffs.view()


# FIXME I feel like these should probably do some cross validation over
# parameters, also shouldn't length_scale be somewhere between 1 and
# game.num_players.max()?

RBF_PARAMS = {'length_scale': 10.0, 'length_scale_bounds': (1.0, 1e+3)}
NOISE_PARAMS = {'noise_level': 1.0, 'noise_level_bounds': (1e-3, 1e+3)}
GP_PARAMS = {'n_restarts_optimizer': 10, 'normalize_y': True}


# FIXME why are funcs, params, and weights passed independently instead of just
# as one kernel? What does this save?
def _train_gp(x, y, kernel_funcs=[gp.kernels.RBF, gp.kernels.WhiteKernel],
              kernel_params=[RBF_PARAMS, NOISE_PARAMS],
              kernel_weights=[1.0, 1.0], gp_params=GP_PARAMS):
    """Train a gaussian process regressor

    Parameters
    ----------
    x : ndarray
        The input vectors. shape (num_samples, dimension).
    y : ndarray
        The output value. shape (num_samples).
    kernel_funcs, kernel_params, kernel_weights : lists
        See documentation for _build_kernel.
    gp_params: dict
        Parameters with which to initialize the GaussianProcessRegressor.
    """
    k = _build_kernel(kernel_funcs, kernel_params, kernel_weights)
    proc = gp.GaussianProcessRegressor(kernel=k, **gp_params)
    # FIXME This is throwing a divide by zero error somewhere and it probably
    # shouldn't just be ignored.
    with np.errstate(divide='ignore'):
        proc.fit(x, y)
    return proc


def _build_kernel(kernels, params, weights=itertools.repeat(1)):
    """Create a weighted linear combination of several kernels

    Parameters
    ----------
    kernels : [kernel]
        Generally imported from gaussian_process.kernels.
    params : [{params}]
        For each kernel, a dictionary of the parameters with which it will
        be initialized.
    weights : [floats], optional
        For each kernel, its weight in the linear combination. If unspecified,
        all kernels will have equal weight.
    """
    return sum(w * k(**p)for k, p, w in zip(kernels, params, weights))


HIDDEN_PARAMS = [{'units': 32, 'activation': 'relu'}] * 2
OUTPUT_PARAMS = {'activation': 'sigmoid'}
COMPILE_PARAMS = {'optimizer': 'sgd', 'loss': 'mean_squared_error'}
FIT_PARAMS = {'epochs': 100, 'verbose': 0}

# XXX right now NNPayoffs always normalizes payoffs to the range [0,1], which
# is compatible with sigmoid output activation, but not necessarily other
# activation functions.


def _train_nn(x, y, num_hidden_layers=1, hidden_params=HIDDEN_PARAMS,
              output_params=OUTPUT_PARAMS, compile_params=COMPILE_PARAMS,
              fit_params=FIT_PARAMS, dropout=.2):
    """Train a keras neural network.

    Parameters
    ----------
    x : ndarray
        Input vectors. shape (num_samples, dimension).
    y : ndarray
        Output values, normalized to the output layer's activation function
        (e.g. [0, 1] for sigmoid. shape (num_samples)
    hidden_layers : integer
        The number of hidden layers. Must match len(hidden_params) or
        hidden_params must be a dict.
    *_params
        Keras parameters used to create, compile, and train the network.
    hidden_params : dict or [dict]
        Can be a dict if all layers have the same parameters
        or list of dicts with length equal to num_hidden_layers.
    output_params, compile_params, fit_params : dict
        TODO
    """
    # XXX These are here to prevent long import times unless keras is actually
    # needed
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    # FIXME Why is this necessary, when it's just as easy to do this at input?
    # and the api is now confusing as num_hiddins does nothing if a list is
    # passed in. In general the role of num_hidden_layers seems to work
    # independent which makes these very confusing.
    if isinstance(hidden_params, dict):
        hidden_params = [hidden_params] * num_hidden_layers
    # FIXME This seems like all of these parameters are just a veiled way to
    # define an NN. Wouldn't it make more sense to have a helper function to
    # create one a separate thing, and then potentially clone it several times
    # for each time you want to train it?
    nn = Sequential()
    nn.add(Dense(input_shape=[x.shape[1]], **hidden_params[0]))
    for l in range(1, num_hidden_layers):
        nn.add(Dense(**hidden_params[l]))
        if dropout > 0:
            nn.add(Dropout(dropout))
    nn.add(Dense(1, **output_params))
    nn.compile(**compile_params)
    nn.fit(x, y, **fit_params)
    return nn
