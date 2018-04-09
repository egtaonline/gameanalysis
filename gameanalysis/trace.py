import numpy as np
from scipy import integrate

from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import utils


def _ode(game0, game1, t_eq, eqm, t_dest, *, regret_thresh=1e-3, max_step=0.1,
         singular=1e-7, **ivp_args):
    '''Trace an equilibrium out to target

    See trace_equilibrium for full info
    '''
    egame = rsgame.emptygame_copy(game0)
    eqm = np.asarray(eqm, float)
    utils.check(
        egame.is_mixture(eqm), 'equilibrium wasn\'t a valid mixture')
    utils.check(
        regret.mixture_regret(
            rsgame.mix(game0, game1, t_eq), eqm) <= regret_thresh + 1e-7,
        'equilibrium didn\'t have regret below threshold')
    ivp_args.update(max_step=max_step)

    # It may be handy to have the derivative of this so that the ode solver can
    # be more efficient, except that computing the derivative w.r.t. t requires
    # the hessian of the deviation payoffs, which would be complicated and so
    # far has no use anywhere else.
    def ode(t, mix_neg):
        div = np.zeros(egame.num_strats)
        mix = egame.trim_mixture_support(mix_neg, thresh=0)
        supp = mix > 0
        rgame = egame.restrict(supp)

        d1, j1 = game0.deviation_payoffs(mix, jacobian=True)
        d2, j2 = game1.deviation_payoffs(mix, jacobian=True)

        gs = (d1 - d2)[supp]
        fs = ((1 - t) * j1 + t * j2)[supp][:, supp]

        g = np.concatenate([
            np.delete(np.diff(gs), rgame.role_starts[1:] - 1),
            np.zeros(egame.num_roles)])
        f = np.concatenate([
            np.delete(np.diff(fs, 1, 0), rgame.role_starts[1:] - 1, 0),
            np.eye(egame.num_roles).repeat(rgame.num_role_strats, 1)])
        if singular < np.abs(np.linalg.det(f)):
            div[supp] = np.linalg.solve(f, g)
        return div

    def below_regret_thresh(t, mix_neg):
        mix = egame.trim_mixture_support(mix_neg, thresh=0)
        reg = regret.mixture_regret(rsgame.mix(game0, game1, t), mix)
        return reg - regret_thresh

    below_regret_thresh.terminal = True
    below_regret_thresh.direction = 1

    def singular_jacobian(t, mix_neg):
        mix = egame.trim_mixture_support(mix_neg, thresh=0)
        supp = mix > 0
        rgame = egame.restrict(supp)
        _, j1 = game0.deviation_payoffs(mix, jacobian=True)
        _, j2 = game1.deviation_payoffs(mix, jacobian=True)
        fs = ((1 - t) * j1 + t * j2)[supp][:, supp]
        f = np.concatenate([
            np.delete(np.diff(fs, 1, 0), rgame.role_starts[1:] - 1, 0),
            np.eye(egame.num_roles).repeat(rgame.num_role_strats, 1)])
        return np.abs(np.linalg.det(f)) - singular

    singular_jacobian.terminal = True
    singular_jacobian.direction = -1

    events = [below_regret_thresh, singular_jacobian]

    # This is to scope the index
    def create_support_loss(ind):
        def support_loss(t, mix):
            return mix[ind]

        support_loss.direction = -1
        return support_loss

    for i in range(egame.num_strats):
        events.append(create_support_loss(i))

    with np.errstate(divide='ignore'):
        res = integrate.solve_ivp(
            ode, [t_eq, t_dest], eqm, events=events, **ivp_args)
    return res.t, egame.trim_mixture_support(res.y.T, thresh=0)


def trace_equilibria(game0, game1, t, eqm, *, regret_thresh=1e-3, max_step=0.1,
                     singular=1e-7, **ivp_args):
    '''Trace an equilibrium between games

    Takes two games, a fraction that they're merged, and an equilibrium of the
    merged game, and traces the equilibrium out to nearby merged games, as far
    as possible.

    Parameters
    ----------
    game0 : RsGame
        The first game that's merged. Represents the payoffs when `t` is 0.
    game1 : RsGame
        The second game that's merged. Represents the payoffs when `t` is 1.
    t : float
        The amount that the two games are merged such that `eqm` is an
        equilibrium. Must be in [0, 1].
    eqm : ndarray
        An equilibrium when `game0` and `game1` are merged a `t` fraction.
    regret_thresh : float, optional
        The amount of gain from deviating to a strategy outside support can
        have before it's considered a beneficial deviation and the tracing
        stops. This should be larger than zero as most equilibria are
        approximate due to floating point precision.
    max_step : float, optional
        The maximum step to take in t when evaluating.
    singular : float, optional
        An absolute determinant below this value is considered singular.
        Occasionally the derivative doesn't exist, and this is one way in which
        that manifests. This values regulate when ODE solving terminates due to
        a singular matrix.
    ivp_args
        Any remaining keyword arguments are passed to the ivp solver.
    '''
    tsb, eqab = _ode(
        game0, game1, t, eqm, 0, regret_thresh=regret_thresh,
        max_step=max_step, singular=singular, **ivp_args)
    tsf, eqaf = _ode(
        game0, game1, t, eqm, 1, regret_thresh=regret_thresh,
        max_step=max_step, singular=singular, **ivp_args)
    ts = np.concatenate([tsb[::-1], tsf[1:]])
    mixes = np.concatenate([eqab[::-1], eqaf[1:]])
    return ts, mixes


def trace_interpolate(game0, game1, ts, eqa, t, **kwargs):
    '''Get an equilibrium at a specific time

    Parameters
    ----------
    game0 : RsGame
        The game to get data from when t is 0.
    game1 : RsGame
        The game to get data from when t is 1.
    ts : [float]
        A parallel list of times for each equilibria in a continuous trace.
    eqa : [eqm]
        A parallel list of equilibria for each time representing continuous
        equilibria for t mixture games.
    t : float
        The time to compute an equilibrium at.
    kwargs : options
        The same options as `trace_equilibria`.
    '''
    ts = np.asarray(ts, float)
    eqa = np.asarray(eqa, float)
    utils.check(ts[0] <= t <= ts[-1], 't must be in trace')
    ind = ts.searchsorted(t)
    if ts[ind] == t:
        return eqa[ind]
    # select nearby equilibrium with maximum support if tied, take lowest reg
    ind = max(ind - 1, ind, key=lambda i: (
        np.sum(eqa[i] > 0),
        regret.mixture_regret(rsgame.mix(game0, game1, ts[i]), eqa[i])))
    (*_, t_res), (*_, eqm_res) = _ode(
        game0, game1, ts[ind], eqa[ind], t, **kwargs)
    utils.check(np.isclose(t_res, t), 'ode solving failed to reach t')
    return eqm_res
