import multiprocessing

import numpy as np
import scipy.special as sps
from scipy import stats
from sklearn import gaussian_process
from sklearn import model_selection

from gameanalysis import rsgame
from gameanalysis import reduction
from gameanalysis import subgame
from gameanalysis import utils


# TODO Make constructors follow format in rsgame

_TINY = np.finfo(float).tiny


class BaseGPGame(rsgame.BaseGame):
    """A game that regresses payoffs with a Gaussian process

    cv_jobs and cv_iters are passed to train_gp and subsequently."""
    # XXX running the gps can be expensive. It might useful to wrap the gps in
    # something like lru-dict so that recent profiles are cached.

    def __init__(self, game, cv_jobs=0, cv_iters=16):
        super().__init__(game.num_players, game.num_strategies)

        if isinstance(game, BaseGPGame):
            # Copy trained models
            self._min_payoffs = game._min_payoffs
            self._max_payoffs = game._max_payoffs
            self._gps = game._gps

        else:
            # copy game's attributes
            self._min_payoffs = game.min_payoffs()
            self._max_payoffs = game.max_payoffs()

            # train GPs for each role/strategy
            self._gps = []
            for rs in range(self.num_role_strats):
                prof_mask = game.profiles[:, rs] > 0
                gp_profs = game.profiles[prof_mask]
                gp_profs[:, rs] -= 1
                gp_pays = game.payoffs[prof_mask, rs]
                self._gps.append(_train_gp(gp_profs, gp_pays, n_jobs=cv_jobs,
                                           n_iter=cv_iters))

    def is_complete(self):
        # GP Games are always complete
        return True

    def get_payoffs(self, profiles):
        """Get the payoffs for a set of profiles"""
        payoffs = np.zeros(profiles.shape)
        for i, gp in enumerate(self._gps):
            mask = profiles[:, i] > 0
            profs = profiles[mask]
            profs[:, i] -= 1
            if profs.shape[0]:
                payoffs[mask, i] = gp.predict(profs)
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
        payoffs = np.empty(self.num_role_strats)
        for i, (gp, r) in enumerate(zip(self._gps, self.role_indices)):
            payoffs[i] = gp.predict(profiles[r]).mean()
        return payoffs

    def min_payoffs(self):
        return self._min_payoffs.view()

    def max_payoffs(self):
        return self._max_payoffs.view()


_CV_PARAMS = {'alpha': stats.powerlaw(.2, loc=1e-3, scale=50)}


# XXX This changed in a scipy update and should be verified that its doing what
# we want
def _train_gp(x, y, **search_kwds):
    if 'n_jobs' in search_kwds and search_kwds['n_jobs'] < 1:
        # one job per cpu core
        search_kwds['n_jobs'] = multiprocessing.cpu_count()
    cv = model_selection.RandomizedSearchCV(
        gaussian_process.GaussianProcessRegressor(),
        _CV_PARAMS, error_score=-np.inf, **search_kwds)
    cv.fit(x, y)
    return cv.best_estimator_


class PointGPGame(BaseGPGame):
    """Evaluates GPs at the 'profile' corresponding to mixture fractions.

    This is similar to neighbor_devs with devs=0, but without rounding to
    integer numbers of players."""

    def __init__(self, game, **base_args):
        super().__init__(game, **base_args)

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        # TODO To add jacobian support, we'd need the derivative of the gp
        # function, which is likely possible, but may not be easy to access in
        # a robust way.
        assert not jacobian, "PointGPGame doesn't support jacobian"
        dev_players = self.num_players - np.eye(self.num_roles, dtype=int)
        dev_profs = self.role_repeat(dev_players) * mix
        return self.get_mean_dev_payoffs(dev_profs[:, None])


class SampleGPGame(BaseGPGame):
    """Averages GP payoff estimates over profiles sampled from mix.

    `samples` random profiles are drawn, distributed according to mix.  The
    learned GP for each strategy is queried at each random profile.  The values
    returned are averages over payoff estimates at the sampled profiles."""

    def __init__(self, game, num_samples=1000, **base_args):
        super().__init__(game, **base_args)
        assert num_samples > 0
        self._num_samples = num_samples

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        assert not jacobian, "SampleGPGame doesn't support jacobian"
        profs = self.random_dev_profiles(mix, self._num_samples).swapaxes(0, 1)
        return self.get_mean_dev_payoffs(profs)


class NeighborGPGame(BaseGPGame):
    """Evaluates GPs at profiles with the highest probability under mix.

    Computes the weighted sum for an exact deviation_payoffs calculation,
    but on a subset of the profiles. Evaluates the GPs at the EV_samples
    profiles closest to mix. Weights are normalized by the sum of
    probabilities of evaluated profiles."""

    def __init__(self, game, num_devs=4, **base_args):
        super().__init__(game, **base_args)
        self._num_devs = num_devs

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        # TODO Add jacobian - difficult because of the division by probs
        # TODO this should probably do some caching to speed up Nash
        # computation Could easily use dynamic array to add a bunch of payoffs
        # and profiles only when necessary
        assert not jacobian, "NeighborGPGame doesn't support jacobian"

        profiles = self.nearby_profs(self.max_prob_prof(mix), self._num_devs)
        payoffs = self.get_payoffs(profiles)

        player_factorial = np.sum(sps.gammaln(profiles + 1), 1)[:, None]
        tot_factorial = np.sum(sps.gammaln(self.num_players + 1))
        log_mix = np.log(mix + _TINY)
        prof_prob = np.sum(profiles * log_mix, 1, keepdims=True)
        profile_probs = tot_factorial - player_factorial + prof_prob
        denom = log_mix + self.role_repeat(np.log(self.num_players))
        with np.errstate(divide='ignore'):
            log_profs = np.log(profiles)
        probs = np.exp(log_profs + profile_probs - denom)
        return np.sum(payoffs * probs, 0) / probs.sum(0)

    def nearby_profs(self, prof, num_devs):
        """Returns profiles reachable by at most num_devs deviations"""
        # XXX this is the bottleneck for gpgame.neighbor_EVs. It seems like
        # there should be some clever way to speed it up.
        assert num_devs >= 0
        dev_players = utils.acomb(self.num_roles, num_devs, True)
        mask = np.all(dev_players <= self.num_players, 1)
        dev_players = dev_players[mask]
        supp = prof > 0
        sub = subgame.subgame(rsgame.basegame_copy(self), supp)

        profs = [prof[None]]
        for players in dev_players:
            to_dev_profs = rsgame.basegame(
                players, self.num_strategies).all_profiles()
            from_dev_profs = subgame.translate(
                rsgame.basegame(players, sub.num_strategies).all_profiles(),
                supp)
            before_devs = prof - from_dev_profs
            before_devs = before_devs[np.all(before_devs >= 0, 1)]
            before_devs = utils.unique_axis(before_devs)
            nearby = before_devs[:, None] + to_dev_profs
            nearby.shape = (-1, self.num_role_strats)
            profs.append(utils.unique_axis(nearby))
        profs = np.concatenate(profs)
        return utils.unique_axis(profs)


class DprGPGame(BaseGPGame):
    """Constructs a DPR game from GPs to estimate payoffs.

    Uses self.DPR_players to determine number of reduced-game players for
    each role."""

    def __init__(self, game, dpr_players=None, **base_args):
        super().__init__(game, **base_args)
        dpr_players = (np.maximum(game.num_players, 2) if dpr_players is None
                       else np.asarray(dpr_players, int))
        red_game = rsgame.basegame(dpr_players, self.num_strategies)
        red = reduction.DeviationPreserving(self.num_strategies,
                                            self.num_players, dpr_players)
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

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        return self.dpr_game.deviation_payoffs(mix, jacobian=jacobian)


class FullGPGame(BaseGPGame):
    """Fills in every profile in the game to estimate payoffs"""

    def __init__(self, game, **base_args):
        super().__init__(game, **base_args)
        profiles = self.all_profiles()
        payoffs = self.get_payoffs(profiles)
        self.full_game = rsgame.game_copy(self, profiles, payoffs)

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        return self.full_game.deviation_payoffs(mix, assume_complete=True,
                                                jacobian=jacobian)
