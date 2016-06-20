from gameanalysis import rsgame
from gameanalysis import reduction

import numpy as np

from multiprocessing import cpu_count
from scipy.stats import powerlaw
from scipy.special import gammaln
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import RandomizedSearchCV

_TINY = np.finfo(float).tiny

class GPGame(rsgame.EmptyGame):
    """
    parameters:
    EV_mode should be one of ['point','sample','neighbor','DPR','full_game']
    EV_samples controls the number of queries to the GP for each strategy made
                by each call to sample_EVs.
    neighbor_devs controls how many profiles are used to compute neighbor_EVs.
                It gives the number of deviations away from the max-probability
                profile that should be generated.
    DPR_players should be either an integer or a list of integers with length
                equal to the number of roles. This specifies the number of
                players in each reduced game role when calculating DPR_EVs.
    CV_jobs and CV_iters are passed to train_gp and subsequently
                RandomizedSearchCV
    """
    def __init__(self, game, EV_mode="point", EV_samples=1000, neighbor_devs=4,
                 DPR_players=2, CV_jobs=0, CV_iters=16):

        assert hasattr(self, EV_mode+"_EVs"), "invalid EV_mode: " + EV_mode
        self.EV_func = getattr(self, EV_mode+"_EVs")
        self.EV_samples = EV_samples
        self.DPR_players = DPR_players
        self.neighbor_devs = neighbor_devs

        # copy game's attributes
        self.strategies = game.strategies
        self.players = game.players
        self.aplayers = game.aplayers
        self.aplayers.setflags(write=False)        
        self.astrategies = game.astrategies
        self.astrategies.setflags(write=False)        
        self.num_roles = game.num_roles
        self.num_role_strats = game.num_role_strats
        self._at_indices = game._at_indices
        self._at_indices.setflags(write=False)        
        self.size = game.size
        self._role_index = game._role_index
        self._role_strat_index = game._role_strat_index
        self._min_payoffs = game.min_payoffs(True)

        # train GPs for each role/strategy
        self.gps = []
        for rs in range(self.num_role_strats):
            x = game._aprofiles[game._aprofiles[:,rs] > 0]
            x[:,rs] -= 1
            y = game._apayoffs[game._aprofiles[:,rs] > 0][:,rs]
            self.gps.append(train_gp(x, y, n_jobs=CV_jobs, n_iter=CV_iters))

    def deviation_payoffs(self, mix, *args, **kwds):
        return self.EV_func(mix)

    def point_EVs(self, mix):
        """Evaluates GPs at the 'profile' corresponding to mixture fractions.

        This is similar to neighbor_EVs with neighbor_devs=0, but without
        rounding to integer numbers of players.
        """
        prof = mix * self.aplayers.repeat(self.astrategies)
        return np.array([gp.predict([prof])[0] for gp in self.gps])

    def sample_EVs(self, mix):
        """Averages GP payoff estimates over profiles sampled from mix.

        EV_samples random profiles are drawn, distributed according to mix.
        The learned GP for each strategy is queried at each random profile.
        The values returned are averages over payoff estimates at the sampled
        profiles.
        """
        profs = self.random_profiles(mix, self.EV_samples, as_array=True)
        return np.array([gp.predict(profs).mean() for gp in self.gps])

    def neighbor_EVs(self, mix):
        """Evaluates GPs at profiles with the highest probability under mix.

        Computes the weighted sum for an exact deviation_payoffs calculation,
        but on a subset of the profiles. Evaluates the GPs at the EV_samples
        search_kwds["n_jobs"] = cpu_count() # one job per cpu core
        profiles closest to mix. Weights are normalized by the sum of
        probabilities of evaluated profiles.


        NOTE: this should probably do some caching to speed up Nash computation
        """
        aprofiles = np.array(self.nearby_profs(self.max_prob_prof(mix, None), 
                                               self.neighbor_devs, True))
        apayoffs = np.array([gp.predict(aprofiles) for gp in self.gps]).T
        player_factorial = self.role_reduce(gammaln(aprofiles + 1), axis=1)
        totals = np.exp(np.sum(gammaln(self.aplayers+1) - player_factorial, 1))
        dev_reps = (totals[:, None] * aprofiles /
                    self.aplayers.repeat(self.astrategies))
        prod = np.prod((mix + _TINY) ** aprofiles, 1, keepdims=True)
        weights = (prod * dev_reps / (mix + _TINY))
        return np.sum(apayoffs * weights, 0) / weights.sum(0)

    def DPR_EVs(self, mix):
        """Constructs a DPR game from GPs to estimate payoffs.

        Uses self.DPR_players to determine number of reduced-game players for
        each role.
        """
        if hasattr(self, "_DPR"):
            return self._DPR.deviation_payoffs(mix, as_array=True)
        if isinstance(self.DPR_players, int):
            self.DPR_players = {r: self.DPR_players for r in self.players}
        self._DPR = rsgame.Game(self.DPR_players, self.strategies,
                        np.empty((0, self.num_role_strats), dtype=int),
                        np.empty((0, self.num_role_strats)))
        self._DPR._aprofiles = self._DPR.all_profiles(as_array=True)
        self._DPR._apayoffs = np.zeros_like(self._DPR._aprofiles, dtype=float)
        DP = reduction.DeviationPreserving(self.players, self.DPR_players)
        for i, prof in enumerate(self._DPR.profiles(as_array=False)):
            for (role, strat), j in self._role_strat_index.items():
                if strat in prof[role]:
                    full_prof = self.as_array(DP.full_prof(prof, role, strat),
                                              dtype=int)
                    self._DPR._apayoffs[i,j] = self.gps[j].predict([full_prof])
        return self._DPR.deviation_payoffs(mix, as_array=True)

    def full_game_EVs(self, mix):
        """Fills in every profile in the game to estimate payoffs."""
        if hasattr(self, "_full_game"):
            return self._full_game.deviation_payoffs(mix, as_array=True)
        self._full_game = rsgame.Game.from_game(self)
        self._full_game._aprofiles = self.all_profiles(as_array=True)
        self._full_game._apayoffs = np.array([gp.predict(
                self._full_game._aprofiles) for gp in self.gps]).T
        return self._full_game.deviation_payoffs(mix, as_array=True)
        
    def min_payoffs(self, as_array=False):
        if as_array or as_array is None:
            return self._min_payoffs.view()
        else:
            return {r: float(m) for r, m
                    in zip(self.strategies.keys(), self._min_payoffs)}


CV_params = {"nugget":powerlaw(.05, loc=1e-12, scale=10),
             "theta0":powerlaw(.2, loc=1e-3, scale=50)}
fixed_params = {"storage_mode":"light"}


def train_gp(x, y, **search_kwds):
    if "n_jobs" in search_kwds and search_kwds["n_jobs"] < 1:
        search_kwds["n_jobs"] = cpu_count() # one job per cpu core
    CV = RandomizedSearchCV(GaussianProcess(**fixed_params), CV_params,
                            error_score=-float("inf"), **search_kwds)
    CV.fit(x, y)
    return CV.best_estimator_
