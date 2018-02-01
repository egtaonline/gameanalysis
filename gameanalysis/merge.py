import functools

import numpy as np
from scipy import integrate

from gameanalysis import gamereader
from gameanalysis import rsgame
from gameanalysis import utils


def trace_equilibria(game1, game2, t, eqm, *, regret_thresh=1e-4,
                     singular=1e-7):
    """Trace an equilibrium between games

    Takes two games, a fraction that they're merged, and an equilibrium of the
    merged game, and traces the equilibrium out to nearby merged games.

    Parameters
    ----------
    game1 : RsGame
        The first game that's merged. Represents the payoffs when `t` is 0.
    game1 : RsGame
        The second game that's merged. Represents the payoffs when `t` is 1.
    t : float
        The amount that the two games are merged such that `eqm` is an
        equilibrium. Must be in [0, 1].
    eqm : ndarray
        An equilibrium when `game1` and `game2` are merged a `t` fraction.
    regret_thresh : float, optional
        The amount of gain from deviating to a strategy outside support can
        have before it's considered a beneficial deviation and the tracing
        stops. This should be larger than zero as most equilibria are
        approximate due to floating point precision.
    singular : float, optional
        An absolute determinant below this value is considered singular.
        Occasionally the derivative doesn't exist, and this is one way in which
        that manifests. This values regulate when ODE solving terminates due to
        a singular matrix.
    """
    egame = rsgame.emptygame_copy(game1)
    eqm = np.asarray(eqm, float)
    assert egame.is_mixture(eqm), "equilibrium wasn't a valid mixture"

    @functools.lru_cache(maxsize=2)
    def cache_comp(hash_m):
        mix = egame.trim_mixture_support(hash_m.array, thresh=0)
        supp = mix > 0
        rgame = egame.restrict(supp)

        d1, j1 = game1.deviation_payoffs(mix, jacobian=True)
        d2, j2 = game2.deviation_payoffs(mix, jacobian=True)

        gs = (d2 - d1)[supp]
        fs = ((1 - t) * j1 + t * j2)[supp][:, supp]

        g = np.concatenate([
            np.delete(np.diff(gs), rgame.role_starts[1:] - 1),
            np.zeros(egame.num_roles)])
        f = np.concatenate([
            np.delete(np.diff(fs, 1, 0), rgame.role_starts[1:] - 1, 0),
            np.eye(egame.num_roles).repeat(rgame.num_role_strats, 1)])
        det_f = np.abs(np.linalg.det(f))
        return supp, mix, d1, d2, g, f, det_f

    # It may be handy to have the derivative of this so that the ode solver can
    # be more efficient, except that computing the derivative w.r.t. requires
    # the hessian of the deviation payoffs, which would be complicated and so
    # far has no use anywhere else.
    def ode(t, mix):
        div = np.zeros(egame.num_strats)
        supp, *_, g, f, det_f = cache_comp(utils.hash_array(mix))
        if det_f > singular:
            div[supp] = np.linalg.solve(f, -g)
        return div

    def beneficial_deviation(t, m):
        supp, mix, d1, d2, *_ = cache_comp(utils.hash_array(m))
        if supp.all():
            return -np.inf
        devs = ((1 - t) * d1 + t * d2)
        exp = np.add.reduceat(devs * mix, egame.role_starts)
        regret = np.max((devs - exp.repeat(egame.num_role_strats))[~supp])
        return regret - regret_thresh

    beneficial_deviation.terminal = True
    beneficial_deviation.direction = 1

    def singular_jacobian(t, mix):
        *_, det_f = cache_comp(utils.hash_array(mix))
        return det_f - singular

    singular_jacobian.terminal = True
    singular_jacobian.direction = -1

    events = [beneficial_deviation, singular_jacobian]

    # This is to scope the index
    def create_support_loss(ind):
        def support_loss(t, mix):
            return mix[ind]

        support_loss.direction = -1
        return support_loss

    for i in range(egame.num_strats):
        events.append(create_support_loss(i))

    with np.errstate(divide='ignore'):
        # Known warning for when gradient equals zero
        res_backward = integrate.solve_ivp(ode, [t, 0], eqm, events=events)
        res_forward = integrate.solve_ivp(ode, [t, 1], eqm, events=events)

    ts = np.concatenate([res_backward.t[::-1], res_forward.t[1:]])
    mixes = np.concatenate([res_backward.y.T[::-1], res_forward.y.T[1:]])
    return ts, egame.trim_mixture_support(mixes, thresh=0)


class MergeGame(rsgame.RsGame):
    """A Game representing the `t` merger between two other games

    Payoffs in this game are simply the weighted fraction of payoffs from game1
    and game2 such that the interpolation is smooth."""

    def __init__(self, game1, game2, t):
        super().__init__(
            game1.role_names, game1.strat_names, game1.num_role_players)
        self._game1 = game1
        self._game2 = game2
        self.t = t

    @property
    @utils.memoize
    def num_complete_profiles(self):
        if self._game1.is_complete() and self._game2.is_complete():
            return self.num_all_profiles
        else:
            profs1 = frozenset(
                utils.hash_array(prof) for prof, pay
                in zip(self._game1.profiles(), self._game1.payoffs())
                if not np.isnan(pay).any())
            profs2 = frozenset(
                utils.hash_array(prof) for prof, pay
                in zip(self._game2.profiles(), self._game2.payoffs())
                if not np.isnan(pay).any())
            return len(profs1.intersection(profs2))

    @property
    @utils.memoize
    def num_profiles(self):
        if self._game1.is_complete() and self._game2.is_complete():
            return self.num_all_profiles
        else:
            return self.profiles().shape[0]

    @functools.lru_cache(maxsize=1)
    def profiles(self):
        if self._game1.is_complete() and self._game2.is_complete():
            return self.all_profiles()
        else:
            profs1 = utils.axis_to_elem(self._game1.profiles())
            profs2 = utils.axis_to_elem(self._game2.profiles())
            return utils.elem_to_axis(np.intersect1d(profs1, profs2), int)

    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        return self.get_payoffs(self.profiles())

    def deviation_payoffs(self, mix, *, jacobian=False):
        if jacobian:
            d1, j1 = self._game1.deviation_payoffs(mix, jacobian=True)
            d2, j2 = self._game2.deviation_payoffs(mix, jacobian=True)
            return ((1 - self.t) * d1 + self.t * d2,
                    (1 - self.t) * j1 + self.t * j2)
        else:
            return ((1 - self.t) * self._game1.deviation_payoffs(mix) +
                    self.t * self._game2.deviation_payoffs(mix))

    def get_payoffs(self, profile):
        return ((1 - self.t) * self._game1.get_payoffs(profile) +
                self.t * self._game2.get_payoffs(profile))

    @utils.memoize
    def max_strat_payoffs(self):
        return ((1 - self.t) * self._game1.max_strat_payoffs() +
                self.t * self._game2.max_strat_payoffs())

    @utils.memoize
    def min_strat_payoffs(self):
        return ((1 - self.t) * self._game1.min_strat_payoffs() +
                self.t * self._game2.min_strat_payoffs())

    def normalize(self):
        return MergeGame(
            self._game1.normalize(), self._game2.normalize(), self.t)

    def restrict(self, rest):
        return MergeGame(
            self._game1.restrict(rest), self._game2.restrict(rest), self.t)

    def to_json(self):
        base = super().to_json()
        base['game_1'] = self._game1.to_json()
        base['game_2'] = self._game2.to_json()
        base['t'] = self.t
        base['type'] = 'merge.1'
        return base

    def __contains__(self, profile):
        return profile in self._game1 and profile in self._game2

    def __eq__(self, other):
        return (super().__eq__(other) and
                ((self._game1 == other._game1 and
                  self._game2 == other._game2 and
                  np.isclose(self.t, other.t)) or
                 (self._game1 == other._game2 and
                  self._game2 == other._game1 and
                  np.isclose(self.t, 1 - other.t))))

    def __hash__(self):
        return hash(frozenset((self._game1, self._game2)))

    def __repr__(self):
        return '{}, {:d} / {:d})'.format(
            super().__repr__()[:-1], self.num_profiles, self.num_all_profiles)


# TODO This creates a lazy merge. Once games implement + and *, this can be
# made lazy_merge and a full merge can be implemented.
def merge(game1, game2, t):
    """Merge two games by a `t` fraction

    Parameters
    ----------
    game1 : RsGame
        The first game to merge.
    game2 : RsGame
        The second game to merge.
    t : float
        The fraction to merge the games. 0 corresponds to a copy of `game1`, 1
        corresponds to `game2`, and somewhere between corresponds to the linear
        interpolation between them.
    """
    assert 0 <= t <= 1, "t must be in [0, 1] but was {:f}".format(t)
    assert rsgame.emptygame_copy(game1) == rsgame.emptygame_copy(game2), \
        "games must have identical structure"
    return MergeGame(game1, game2, t)


def merge_json(jgame):
    """Read a merged game from json"""
    base = rsgame.emptygame_json(jgame)
    game1 = gamereader.loadj(jgame['game_1'])
    game2 = gamereader.loadj(jgame['game_2'])
    assert base == rsgame.emptygame_copy(game1), \
        "game structure didn't match each merged game"
    return merge(game1, game2, jgame['t'])
