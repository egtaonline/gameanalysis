import itertools
import math
import warnings

import numpy as np
import numpy.linalg as nla
import scipy.misc as spm
import scipy.optimize as opt

from gameanalysis import gameio
from gameanalysis import rsgame


class CongestionGame(rsgame.BaseGame):
    """Congestion Game

    Parameters
    ----------
    num_players : int
        The number of players in the symmetric congestion game.
    num_required : int
        The number of required facilities in a strategy.
    facility_coefs : ndarray, (num_facilities, 3)
        The polynomial coefficients for the congestion function. The first
        column is constant, then linear, then quadratic.
    """
    # TODO This could be extended to any-order polynomials. The dth moment of
    # the binomial distribution is \sum_{k = 1}^d n!/(n - k)! p^k {d k}, where
    # {d k} is a sterling number of the second kind.
    def __init__(self, num_players, num_required, facility_coefs):
        self.num_facilities = facility_coefs.shape[0]
        assert facility_coefs.shape[1] == 3, \
            "Congestion games only support quadratic congestion"
        self.facility_coefs = facility_coefs.view()
        self.facility_coefs.setflags(write=False)

        assert num_required > 0
        assert self.num_facilities >= num_required
        num_strats = spm.comb(self.num_facilities, num_required, exact=True)
        super().__init__(num_players, num_strats)
        self.num_required = num_required

        self._strats = np.zeros((num_strats, self.num_facilities), bool)
        for i, inds in enumerate(itertools.combinations(
                range(self.num_facilities), num_required)):
            self._strats[i, inds] = True
        self._strats.setflags(write=False)

        # Set placeholder values
        self._min_payoffs = None
        self._max_payoffs = None

    def is_complete(self):
        """Congestion games are always complete"""
        return True

    def min_payoffs(self):
        # Computes the min payoff by finding the required facilities that have
        # the lowest payoff when everyone uses them. This will fail if the
        # highest order term isn't negative.
        if self._min_payoffs is None:
            self._min_payoffs = self.num_players.astype(float)
            self._min_payoffs *= np.partition(
                self.facility_coefs.dot(self.num_players ** np.arange(3)),
                self.num_required - 1)[:self.num_required].sum()
            self._min_payoffs.setflags(write=False)
        return self._min_payoffs

    def max_payoffs(self):
        """Computes the max payoff per role

        For computational efficiency, this computes an upper bound on max
        payoff by relaxing integer assignment to facilities. In practice it
        seems very close.
        """
        # This is structured to solve the facility assignment optimization,
        # e.g. assign num_required * num_players to num_facilities to maximize
        # total payoff. This is hard, so we solve the relaxation where
        # potentially everyone could play the same facility, and we allow
        # fractional assignment. We solve it by solving the lagrange multiplier
        # equation instead of doing gradient descent. We also ignore the x >= 0
        # constraint. Most valid congestion games should have an optimum that
        # already satisfies that constraint, and if not, the answer will still
        # be a valid upper bound.
        if self._max_payoffs is None:
            total = self.num_required * self.num_players[0]

            def eqas(x):
                vals = np.empty(x.size)
                vals[:-1] = np.sum(x[:-1, None] ** np.arange(3) *
                                   np.arange(1, 4) * self.facility_coefs, 1)
                vals[:-1] -= x[-1]
                vals[-1] = x[:-1].sum() - total
                return vals

            def eqajac(x):
                jac = np.zeros((self.num_facilities + 1,) * 2)
                diag = (2 * self.facility_coefs[:, 1] +
                        6 * self.facility_coefs[:, 2] * x[:-1])
                np.fill_diagonal(jac[:-1, :-1], diag)
                jac[-1, :-1] = 1
                jac[:-1, -1] = -1
                return jac

            x0 = np.ones(self.num_facilities + 1) / total
            res = opt.fsolve(eqas, x0, fprime=eqajac)
            self._max_payoffs = np.empty(1)
            self._max_payoffs[0] = np.sum(res[:-1, None] ** np.arange(1, 4) *
                                          self.facility_coefs)
            self._max_payoffs.setflags(write=False)
        return self._max_payoffs

    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        mix = np.asarray(mix, float)
        n = self.num_players[0] - 1
        fac_probs = np.sum(mix[..., None] * self._strats, -2)
        ex = fac_probs * n
        fac_payoffs = (self.facility_coefs[..., 0] +
                       self.facility_coefs[..., 1] * (ex + 1) +
                       self.facility_coefs[..., 2] *
                       (ex * fac_probs * (n - 1) + 3 * ex + 1))
        payoffs = fac_payoffs.dot(self._strats.T)

        if not jacobian:
            return payoffs

        dfac_payoffs = (self.facility_coefs[..., 1] * n +
                        self.facility_coefs[..., 2] * (2 * n * (n - 1) *
                                                       fac_probs + 3 * n))
        jac = np.dot(self._strats * dfac_payoffs[..., None, :], self._strats.T)
        return payoffs, jac

    def get_payoffs(self, profiles):
        usage = np.asarray(profiles, int).dot(self._strats)
        fac_payoffs = np.sum(self.facility_coefs * usage[..., None] **
                             np.arange(3), -1)
        payoffs = fac_payoffs.dot(self._strats.T) * (profiles > 0)
        return payoffs

    def to_game(self):
        profiles = self.all_profiles()
        payoffs = self.get_payoffs(profiles)
        return rsgame.Game(self, profiles, payoffs)

    def gen_serializer(self):
        digits = math.ceil(math.log10(self.num_facilities))
        strats = ['_'.join('{:0{}d}'.format(f, digits)
                           for f in np.nonzero(x)[0]) for x in self._strats]
        return gameio.GameSerializer(['all'], [strats])

    def __repr__(self):
        return '{}({}, {}, {})'.format(
            self.__class__.__name__,
            self.num_players[0],
            self.num_required,
            self.num_facilities)

    def to_json(self, serial=None):
        """Convert game to json

        Parameters
        ----------
        serial : GameSerializer
            If unspecified, one will be generated using :func:`gen_serializer`
        """
        if serial is None:
            serial = self.gen_serializer()
        facilities = sorted(set(itertools.chain.from_iterable(
            s.split('_') for s in serial.strat_names[0])))
        if len(facilities) != self.num_facilities:
            warnings.warn('Splitting strategies with "_" did not result '
                          'in the right number of facilities. `to_json` '
                          'will not produce accurate results.')
        json_ = super().to_json(serial)
        json_['num_required_facilities'] = self.num_required
        json_['facilities'] = {f: coefs.tolist() for f, coefs
                               in zip(facilities, self.facility_coefs)}
        return json_

    def to_str(self, serial=None):
        """Convert game to a human string

        Parameters
        ----------
        serial : GameSerializer
            If unspecified, one will be generated on the fly.
        """
        if serial is None:
            digits = math.ceil(math.log10(self.num_facilities))
            facilities = ('{:0{}d}'.format(f, digits)
                          for f in range(self.num_facilities))
        else:
            facilities = sorted(set(itertools.chain.from_iterable(
                s.split('_') for s in serial.strat_names[0])))
            if len(facilities) != self.num_facilities:
                warnings.warn('Splitting strategies with "_" did not result '
                              'in the right number of facilities. `to_json` '
                              'will not produce accurate results.')
        return ('{}:\n\tPlayers: {:d}\n\tRequired Facilities: {:d}'
                '\n\tFacilities: {}\n'
                .format(
                    self.__class__.__name__,
                    self.num_players[0],
                    self.num_required,
                    ', '.join(facilities)
                )).expandtabs(4)


def read_congestion_game(json_):
    num_players = next(iter(json_['players'].values()))
    num_required = json_['num_required_facilities']
    ordered = sorted(json_['facilities'].items())
    congest_matrix = np.array(list(x[1] for x in ordered), float)
    facilities = list(x[0] for x in ordered)
    strats = list('_'.join(facs) for facs
                  in itertools.combinations(facilities, num_required))
    strategies = json_['strategies']
    assert strats == next(iter(strategies.values())), (
        "strategies recovered from facilities didn't equal provided "
        "strategies. This likely means there was an error with serializing "
        "the congestion game, potentially because the facility names "
        "contained an invalid character.")
    cgame = CongestionGame(num_players, num_required, congest_matrix)
    serial = gameio.GameSerializer(strategies)
    return cgame, serial
