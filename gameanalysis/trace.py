"""Module for tracing equilibria in mixture games"""
import numpy as np
from scipy import integrate

from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import utils


def _ode( # pylint: disable=too-many-locals
        game0, game1, p_eq, eqm, p_dest, *, regret_thresh=1e-3, max_step=0.1,
        singular=1e-7, **ivp_args):
    """Trace an equilibrium out to target

    See trace_equilibrium for full info
    """
    egame = rsgame.empty_copy(game0)
    eqm = np.asarray(eqm, float)
    utils.check(
        egame.is_mixture(eqm), "equilibrium wasn't a valid mixture")
    utils.check(
        regret.mixture_regret(
            rsgame.mix(game0, game1, p_eq), eqm) <= regret_thresh + 1e-7,
        "equilibrium didn't have regret below threshold")
    ivp_args.update(max_step=max_step)

    # It may be handy to have the derivative of this so that the ode solver can
    # be more efficient, except that computing the derivative w.r.t. t requires
    # the hessian of the deviation payoffs, which would be complicated and so
    # far has no use anywhere else.
    def ode(prob, mix_neg):
        """ODE function for solve_ivp"""
        div = np.zeros(egame.num_strats)
        mix = egame.trim_mixture_support(mix_neg, thresh=0)
        supp = mix > 0
        rgame = egame.restrict(supp)

        dev1, jac1 = game0.deviation_payoffs(mix, jacobian=True)
        dev2, jac2 = game1.deviation_payoffs(mix, jacobian=True)

        gvals = (dev1 - dev2)[supp]
        fvecs = ((1 - prob) * jac1 + prob * jac2)[supp][:, supp]

        gvec = np.concatenate([
            np.delete(np.diff(gvals), rgame.role_starts[1:] - 1),
            np.zeros(egame.num_roles)])
        fmat = np.concatenate([
            np.delete(np.diff(fvecs, 1, 0), rgame.role_starts[1:] - 1, 0),
            np.eye(egame.num_roles).repeat(rgame.num_role_strats, 1)])
        if singular < np.abs(np.linalg.det(fmat)):
            div[supp] = np.linalg.solve(fmat, gvec)
        return div

    def below_regret_thresh(prob, mix_neg):
        """Event for regret going above threshold"""
        mix = egame.trim_mixture_support(mix_neg, thresh=0)
        reg = regret.mixture_regret(rsgame.mix(game0, game1, prob), mix)
        return reg - regret_thresh

    below_regret_thresh.terminal = True
    below_regret_thresh.direction = 1

    def singular_jacobian(prob, mix_neg):
        """Event for when jacobian is singular"""
        mix = egame.trim_mixture_support(mix_neg, thresh=0)
        supp = mix > 0
        rgame = egame.restrict(supp)
        _, jac1 = game0.deviation_payoffs(mix, jacobian=True)
        _, jac2 = game1.deviation_payoffs(mix, jacobian=True)
        fvecs = ((1 - prob) * jac1 + prob * jac2)[supp][:, supp]
        fmat = np.concatenate([
            np.delete(np.diff(fvecs, 1, 0), rgame.role_starts[1:] - 1, 0),
            np.eye(egame.num_roles).repeat(rgame.num_role_strats, 1)])
        return np.abs(np.linalg.det(fmat)) - singular

    singular_jacobian.terminal = True
    singular_jacobian.direction = -1

    events = [below_regret_thresh, singular_jacobian]

    # This is to scope the index
    def create_support_loss(ind):
        """Create support loss for every ind"""
        def support_loss(_, mix):
            """Support loss event"""
            return mix[ind]

        support_loss.direction = -1
        return support_loss

    for strat in range(egame.num_strats):
        events.append(create_support_loss(strat))

    with np.errstate(divide='ignore'):
        res = integrate.solve_ivp(
            ode, [p_eq, p_dest], eqm, events=events, **ivp_args)
    return res.t, egame.trim_mixture_support(res.y.T, thresh=0)


def trace_equilibria(
        game0, game1, prob, eqm, *, regret_thresh=1e-3, max_step=0.1,
        singular=1e-7, **ivp_args):
    """Trace an equilibrium between games

    Takes two games, a fraction that they're merged, and an equilibrium of the
    merged game, and traces the equilibrium out to nearby merged games, as far
    as possible.

    Parameters
    ----------
    game0 : RsGame
        The first game that's merged. Represents the payoffs when `prob` is 0.
    game1 : RsGame
        The second game that's merged. Represents the payoffs when `prob` is 1.
    prob : float
        The amount that the two games are merged such that `eqm` is an
        equilibrium. Must be in [0, 1].
    eqm : ndarray
        An equilibrium when `game0` and `game1` are merged a `prob` fraction.
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
    """
    psb, eqab = _ode(
        game0, game1, prob, eqm, 0, regret_thresh=regret_thresh,
        max_step=max_step, singular=singular, **ivp_args)
    psf, eqaf = _ode(
        game0, game1, prob, eqm, 1, regret_thresh=regret_thresh,
        max_step=max_step, singular=singular, **ivp_args)
    probs = np.concatenate([psb[::-1], psf[1:]])
    mixes = np.concatenate([eqab[::-1], eqaf[1:]])
    return probs, mixes


def trace_interpolate(game0, game1, probs, eqa, prob, **kwargs):
    """Get an equilibrium at a specific time

    Parameters
    ----------
    game0 : RsGame
        The game to get data from when prob is 0.
    game1 : RsGame
        The game to get data from when prob is 1.
    probs : [float]
        A parallel list of probs for each equilibria in a continuous trace.
    eqa : [eqm]
        A parallel list of equilibria for each prob representing continuous
        equilibria for prob mixture games.
    prob : float
        The probability to compute an equilibrium at.
    kwargs : options
        The same options as `trace_equilibria`.
    """
    probs = np.asarray(probs, float)
    eqa = np.asarray(eqa, float)
    utils.check(probs[0] <= prob <= probs[-1], 't must be in trace')
    ind = probs.searchsorted(prob)
    if probs[ind] == prob:
        return eqa[ind]
    # select nearby equilibrium with maximum support if tied, take lowest reg
    ind = max(ind - 1, ind, key=lambda i: (
        np.sum(eqa[i] > 0),
        regret.mixture_regret(rsgame.mix(game0, game1, probs[i]), eqa[i])))
    (*_, p_res), (*_, eqm_res) = _ode( # pylint: disable=too-many-star-expressions
        game0, game1, probs[ind], eqa[ind], prob, **kwargs)
    utils.check(np.isclose(p_res, prob), 'ode solving failed to reach prob')
    return eqm_res
