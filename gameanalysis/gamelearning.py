import numpy as np
import scipy.special as sps
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from keras.models import Sequential
from keras.layers import Dense

from gameanalysis import rsgame
from gameanalysis import reduction
from gameanalysis import subgame
from gameanalysis import utils


_TINY = np.finfo(float).tiny


class RegressionPayoffs(object):
    def get_payoffs(self, profiles):
        """Get the payoffs for a set of profiles"""
        payoffs = np.zeros(profiles.shape)
        for i, model in enumerate(self.regressors):
            mask = profiles[:, i] > 0
            profs = profiles[mask]
            profs[:, i] -= 1
            if profs.shape[0]:
                payoffs[mask, i] = model.predict(profs)
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
        payoffs = np.empty(self.game.num_role_strats)
        for i, (model, r) in enumerate(zip(self.regressors, self.game.role_indices)):
            payoffs[i] = model.predict(profiles[r]).mean()
        return payoffs

    def learn_payoffs(self, profiles, payoffs, num_role_strats, training_function, training_args):
        """Trains a regression model for each strategy on the appropriate payoff data."""
        self.regressors = []
        all_payoffs = ~np.isnan(payoffs).any(1)
        for rs in range(num_role_strats):
            prof_mask = (profiles[:, rs] > 0) & all_payoffs
            rs_profs = profiles[prof_mask]
            rs_profs[:, rs] -= 1
            rs_pays = payoffs[prof_mask, rs]
            self.regressors.append(training_function(rs_profs, rs_pays, **training_args))


class GPPayoffs(RegressionPayoffs):
    def __init__(self, game_to_learn, **gp_args):
        self.game = rsgame.BaseGame(game_to_learn.num_players, game_to_learn.num_strategies)
        profiles = game_to_learn.profiles
        payoffs = game_to_learn.payoffs
        self.learn_payoffs(profiles, payoffs, self.game.num_role_strats, _train_gp, gp_args)


class NNPayoffs(RegressionPayoffs):
    def __init__(self, game_to_learn, **nn_args):
        self.game = rsgame.BaseGame(game_to_learn.num_players, game_to_learn.num_strategies)
        self.regressors = []
        if isinstance(game_to_learn, rsgame.SampleGame):
            profiles = []
            payoffs = []
            for prof in game_to_learn.profiles:
                for pay in game_to_learn.get_sample_payoffs(prof):
                    profiles.append(prof)
                    payoffs.append(pay)
            profiles = np.array(profiles)
            payoffs = np.array(payoffs)
        else:
            profiles = game_to_learn.profiles
            payoffs = np.array(game_to_learn.payoffs)
        all_payoffs = ~np.isnan(payoffs).any(1)

        # normalize payoffs to [0,1]
        # TODO: allow specification of an arbitrary payoff range
        self._norm_factors = []
        for rs in range(self.game.num_role_strats):
            prof_mask = (profiles[:, rs] > 0) & all_payoffs
            rs_pays = payoffs[prof_mask, rs]
            minimum = rs_pays.min()
            maximum = rs_pays.max()
            if maximum == minimum: # all payoffs 0; don't divide by 0
                maximum += 1
            self._norm_factors.append([minimum, maximum - minimum])
        self._norm_factors = np.array(self._norm_factors)
        payoffs -= self._norm_factors[:,0]
        payoffs /= self._norm_factors[:,1]
        self.learn_payoffs(profiles, payoffs, self.game.num_role_strats, _train_nn, nn_args)

    def unnormalize(self, payoffs):
        """Return payoffs to the original scale."""
        return (payoffs * self._norm_factors[:,1]) + self._norm_factors[:,0]

    def get_payoffs(self, profiles):
        """Unnormalizes the payoffs returned by RegressionPayoffs."""
        return self.unnormalize(RegressionPayoffs.get_payoffs(self, profiles))

    def get_mean_dev_payoffs(self, profiles):
        """Unnormalizes the deviation payoffs returned by RegressionPayoffs."""
        return self.unnormalize(RegressionPayoffs.get_mean_dev_payoffs(self, profiles))


class FullGameEVs(rsgame.BaseGame):
    def __init__(self, regression_model, **args):
        self.regression_model = regression_model
        self.game = self.regression_model.game
        profiles = self.game.all_profiles()
        payoffs = self.regression_model.get_payoffs(profiles)
        self.full_game = rsgame.game_copy(self.game, profiles, payoffs)

    def deviation_payoffs(self, mix, *args, **kwds):
        return self.full_game.deviation_payoffs(mix, *args, **kwds)


class SampleEVs(rsgame.BaseGame):
    def __init__(self, regression_model, num_samples=1000):
        self.regression_model = regression_model
        self.game = self.regression_model.game
        assert num_samples > 0
        self._num_samples = num_samples

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        assert not jacobian, "SampleEVs doesn't support jacobian"
        profs = self.game.random_dev_profiles(mix, self._num_samples).swapaxes(0, 1)
        return self.regression_model.get_mean_dev_payoffs(profs)


class PointEVs(rsgame.BaseGame):
    def __init__(self, regression_model):
        self.regression_model = regression_model
        self.game = self.regression_model.game

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        assert not jacobian, "PointEVs doesn't support jacobian"
        dev_players = self.game.num_players - np.eye(self.game.num_roles, dtype=int)
        dev_profs = self.game.role_repeat(dev_players) * mix
        return self.regression_model.get_mean_dev_payoffs(dev_profs[:, None])


class NeighborEVs(rsgame.BaseGame):
    def __init__(self, regression_model, num_devs=2):
        self.regression_model = regression_model
        self.game = self.regression_model.game
        self._num_devs = num_devs

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        assert not jacobian, "NeighborEVs doesn't support jacobian"

        profiles = self.game.nearby_profs(self.game.max_prob_prof(mix), self._num_devs)
        payoffs = self.regression_model.get_payoffs(profiles)

        player_factorial = np.sum(sps.gammaln(profiles + 1), 1)[:, None]
        tot_factorial = np.sum(sps.gammaln(self.num_players + 1))
        log_mix = np.log(mix + _TINY)
        prof_prob = np.sum(profiles * log_mix, 1, keepdims=True)
        profile_probs = tot_factorial - player_factorial + prof_prob
        denom = log_mix + self.game.role_repeat(np.log(self.game.num_players))
        with np.errstate(divide='ignore'):
            log_profs = np.log(profiles)
        probs = np.exp(log_profs + profile_probs - denom)
        return np.sum(payoffs * probs, 0) / probs.sum(0)

    def nearby_profs(self, prof, num_devs):
        """Returns profiles reachable by at most num_devs deviations"""
        # XXX this is the bottleneck for gpgame.neighbor_EVs. It seems like
        # there should be some clever way to speed it up.
        assert num_devs >= 0
        dev_players = utils.acomb(self.game.num_roles, num_devs, True)
        mask = np.all(dev_players <= self.game.num_players, 1)
        dev_players = dev_players[mask]
        supp = prof > 0
        sub = subgame.subgame(rsgame.basegame_copy(self.game), supp)

        profs = [prof[None]]
        for players in dev_players:
            to_dev_profs = rsgame.basegame(
                players, self.game.num_strategies).all_profiles()
            from_dev_profs = subgame.translate(
                rsgame.basegame(players, sub.num_strategies).all_profiles(), supp)
            before_devs = prof - from_dev_profs
            before_devs = before_devs[np.all(before_devs >= 0, 1)]
            before_devs = utils.unique_axis(before_devs)
            nearby = before_devs[:, None] + to_dev_profs
            nearby.shape = (-1, self.game.num_role_strats)
            profs.append(utils.unique_axis(nearby))
        profs = np.concatenate(profs)
        return utils.unique_axis(profs)


class DPREVs(rsgame.BaseGame):
    def __init__(self, regression_model, dpr_players=None):
        self.regression_model = regression_model
        self.game = self.regression_model.game
        if dpr_players is None:
            dpr_players = np.maximum(self.game.num_players, 2)
        else:
            dpr_players = np.asarray(dpr_players, int)
        red_game = rsgame.basegame(dpr_players, self.num_strategies)
        red = reduction.DeviationPreserving(self.game.num_strategies,
                                            self.game.num_players, dpr_players)
        red_profiles = red_game.all_profiles()
        full_profiles, contributions = red.expand_profiles(red_profiles, True)
        full_payoffs = np.zeros(full_profiles.shape, float)

        for i, (gp, cont_mask) in enumerate(zip(self._gps, contributions.T)):
            mask = cont_mask & full_profiles[:, i] > 0
            profs = full_profiles[mask]
            profs[:, i] -= 1
            full_payoffs[mask, i] = gp.predict(profs)

        self.dpr_game = red.reduce_game(rsgame.game_copy(game, full_profiles,
                                                         full_payoffs))

    def deviation_payoffs(self, mix, *args, **kwds):
        return self.dpr_game.deviation_payoffs(mix, *args, **kwds)


_REGRESSION_METHODS = {"gp":GPPayoffs, "nn":NNPayoffs}
_EV_METHODS = {"full":FullGameEVs, "sample":SampleEVs, "point":PointEVs,
               "neighbor":NeighborEVs, "dpr":DPREVs}

class RegressionGame(rsgame.BaseGame):
    def __init__(self, game_to_learn, regression_method, EV_method,
                 regression_args={}, EV_args={}):
        """
        game_to_learn can take any of the following forms:
        - RegressionGame ... learned regression_model is preserved;
                             allows changing of EV_model
        - Game/SampleGame ... game data is passed to the regression_model
        """
        assert regression_method.lower() in _REGRESSION_METHODS, \
               "invalid regression_method: " + str(regression_method) + \
               ". options: " + ", ".join(_REGRESSION_METHODS)
        self.regression_method = _REGRESSION_METHODS[regression_method.lower()]
        assert EV_method.lower() in _EV_METHODS, \
               "invalid EV_method: " + str(EV_method) + \
               ". options: " + ", ".join(_EV_METHODS)
        self.EV_method = _EV_METHODS[EV_method.lower()]
        super().__init__(game_to_learn.num_players, game_to_learn.num_strategies)

        if isinstance(game_to_learn, RegressionGame):
            # shallow copy trained models
            assert self.regression_method == game_to_learn.regression_method, \
                "can't change regression method; use original payoff data"
            self._min_payoffs = game_to_learn._min_payoffs
            self._max_payoffs = game_to_learn._max_payoffs
            self.regression_model = game_to_learn.regression_model
        else:
            assert game_to_learn.num_complete_profiles >= 3, \
                "can't learn a game from less than 3 profiles"
            self._min_payoffs = game_to_learn.min_payoffs()
            self._max_payoffs = game_to_learn.max_payoffs()
            self.regression_model = self.regression_method(game_to_learn, **regression_args)

        self.EV_model = self.EV_method(self.regression_model, **EV_args)

    def deviation_payoffs(self, mix, *args, **kwds):
        return self.EV_model.deviation_payoffs(mix, *args, **kwds)

    def get_payoffs(self, profiles):
        return self.regression_model.get_payoffs(profiles)

    def is_complete(self):
        # RegressionGames are always complete
        return True

    def min_payoffs(self):
        return self._min_payoffs.view()

    def max_payoffs(self):
        return self._max_payoffs.view()


RBF_PARAMS = {"length_scale":10.0, "length_scale_bounds":(1.0, 1e+3)}
NOISE_PARAMS = {"noise_level":1.0, "noise_level_bounds":(1e-3, 1e+3)}
GP_PARAMS = {"n_restarts_optimizer":10, "normalize_y":True}

def _train_gp(x, y, kernel_funcs=[kernels.RBF, kernels.WhiteKernel],
          kernel_params=[RBF_PARAMS, NOISE_PARAMS],
          kernel_weights=[1.0, 1.0], gp_params=GP_PARAMS):
    """Train a gaussian process regressor.

    Parameters
    ----------
    x, y : numpy arrays
        training data
    kernel_funcs, kernel_params, kernel_weights : lists
        see documentation for _build_kernel
    gp_params: dictionary
        parameters with which to initialize the GaussianProcessRegressor
    """
    k = _build_kernel(kernel_funcs, kernel_params, kernel_weights)
    gp = GaussianProcessRegressor(kernel=k, **gp_params)
    gp.fit(x, y)
    return gp


def _build_kernel(kernels, params, weights=None):
    """Create a linear combination of several kernels.

    Parameters
    ----------
    kernels : list of kernel functions
        (generally imported from gaussian_process.kernels)
    params : list of parameter dictionaries
        For each kernel, a dictionary of the parameters with which it will
        be initialized.
    weights : list of floats, optional
        For each kernel, its weight in the linear combination. If unspecified,
        all kernels will have equal weight.
    """
    if weights is None:
        weights = [1.0] * len(kernel_functions)
    kernel = 0.0
    for k, p, w in zip(kernels, params, weights):
        kernel += w * k(**p)
    return kernel


HIDDEN_PARAMS = [{"units":32, "activation":"relu"}]*2
OUTPUT_PARAMS = {"activation":"sigmoid"}
COMPILE_PARAMS = {"optimizer":"sgd", "loss":"mean_squared_error"}
FIT_PARAMS = {"epochs":100, "verbose":0}

# NOTE: right now NNPayoffs always normalizes payoffs to the range [0,1],
# which is compatible with sigmoid output activation, but not necessarily
# other activation functions.

def _train_nn(x, y, num_hidden_layers=1, hidden_params=HIDDEN_PARAMS,
              output_params=OUTPUT_PARAMS, compile_params=COMPILE_PARAMS,
              fit_params=FIT_PARAMS, dropout=.2):
    """Train a keras neural network.

    Parameters
    ----------
    x, y : numpy arrays
        training data
        y should be one-dimensional and normalized to the range of the output
        layer's activation function (e.g. [0,1] for sigmoid)

    hidden_layers : integer
    *_params
        keras parameters used to create, compile, and train the network
    hidden_params : dict or list of dicts
        can be a dict if all layers have the same parameters
        or list of dicts with length equal to num_hidden_layers
    output_params, compile_params, fit_params : dicts
    """
    if isinstance(hidden_params, dict):
        hidden_params = [hidden_params] * num_hidden_layers
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
