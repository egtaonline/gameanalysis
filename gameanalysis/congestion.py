import itertools

import numpy as np
import numpy.random as rand
import scipy.misc as spm

from gameanalysis import gameio
from gameanalysis import rsgame


class CongestionGame(rsgame.BaseGame):
    """Congestion Game"""
    # TODO This could be extended to any-order polynomials. The dth moment of
    # the binomial distribution is \sum_{k = 1}^d n!/(n - k)! p^k {d k}, where
    # {d k} is a sterling number of the second kind.
    def __init__(self, num_players, num_facilities, num_required):
        assert num_required > 0
        assert num_facilities >= num_required
        num_strats = spm.comb(num_facilities, num_required, exact=True)
        super().__init__(num_players, num_strats)
        self.num_facilities = num_facilities
        self.num_required = num_required

        self._strats = np.zeros((num_strats, num_facilities), bool)
        for i, inds in enumerate(itertools.combinations(
                range(num_facilities), num_required)):
            self._strats[i, inds] = True

        # Generate value for congestions
        # [Constant, Linear, Quadratic]
        ranges = np.array([num_facilities, -num_required, -1])
        self._congest = rand.random((num_facilities, 3)) * ranges

        # Compute extreme payoffs
        self._min_payoffs = np.array(np.partition(
            self._congest.dot(num_players ** np.arange(3)),
            num_required - 1)[:num_required].sum())
        self._min_payoffs.shape = (1,)
        self._min_payoffs.setflags(write=False)
        if num_facilities >= 2 * num_required:
            self._max_payoffs = np.array(np.partition(
                self._congest.sum(-1),
                -self.num_required)[-self.num_required:].sum())
        else:
            # XXX Because this is an integer problem, I'm not convinced there's
            # a great way to calculate this that's not roughly on the order of
            # enumerating all of the profiles so it's not calculated.
            self._max_payoffs = self._min_payoffs
        self._max_payoffs.shape = (1,)
        self._max_payoffs.setflags(write=False)

    def min_payoffs(self):
        return self._min_payoffs

    def max_payoffs(self):
        return self._max_payoffs

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        mix = np.asarray(mix, float)
        n = self.num_players[0] - 1
        fac_probs = np.sum(mix[..., None] * self._strats, -2)
        ex = fac_probs * n
        fac_payoffs = (self._congest[..., 0] +
                       self._congest[..., 1] * (ex + 1) +
                       self._congest[..., 2] * (ex * fac_probs * (n - 1) +
                                                3 * ex + 1))
        payoffs = fac_payoffs.dot(self._strats.T)

        if not jacobian:
            return payoffs

        dfac_payoffs = (self._congest[..., 1] * n +
                        self._congest[..., 2] * (2 * n * (n - 1) * fac_probs +
                                                 3 * n))
        jac = np.dot(self._strats * dfac_payoffs[..., None, :], self._strats.T)
        return payoffs, jac

    def get_payoffs(self, profiles):
        usage = np.asarray(profiles, int).dot(self._strats)
        fac_payoffs = np.sum(self._congest * usage[..., None] ** np.arange(3),
                             -1)
        return fac_payoffs.dot(self._strats.T) * (profiles > 0)

    def to_game(self):
        profiles = self.all_profiles()
        payoffs = self.get_payoffs(profiles)
        return rsgame.Game(self, profiles, payoffs)

    def gen_serializer(self):
        strats = ['_'.join(map(str, np.nonzero(x)[0])) for x in self._strats]
        return gameio.GameSerializer(['all'], [strats])

    def __repr__(self):
        return '{}({}, {}, {})'.format(
            self.__class__.__name__,
            self.num_players[0],
            self.num_facilities,
            self.num_required)
