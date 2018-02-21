import functools
import threading

import numpy as np
from scipy import integrate

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

    with _trace_lock:
        with np.errstate(divide='ignore'):
            # Known warning for when gradient equals zero
            res_backward = integrate.solve_ivp(ode, [t, 0], eqm, events=events)
            res_forward = integrate.solve_ivp(ode, [t, 1], eqm, events=events)

    ts = np.concatenate([res_backward.t[::-1], res_forward.t[1:]])
    mixes = np.concatenate([res_backward.y.T[::-1], res_forward.y.T[1:]])
    return ts, egame.trim_mixture_support(mixes, thresh=0)


_trace_lock = threading.Lock()
