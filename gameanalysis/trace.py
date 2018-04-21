"""Module for tracing equilibria in mixture games"""
import numpy as np
from scipy import integrate

from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import utils


# FIXME Doesn't matter if F is singular, it matters if any solution exists. If
# F is nonsingular, then a solution definitely exists, otherwise, it might, and
# we can use np.linalg.lstsq to find it. We need to text that we've found a
# solution afterwards. This should be done with np.linalg.norm <
# np.finfo(dtype).eps * num_strats
def trace_equilibrium( # pylint: disable=too-many-locals
        game0, game1, peq, eqm, target, *, regret_thresh=1e-3, max_step=0.1,
        singular=1e-7, **ivp_args):
    """Try to trace an equilibrium out to target

    Takes two games, a fraction that they're mixed (`peq`), and an equilibrium
    of the mixed game (`eqm`). It then attempts to find the equilibrium at the
    `target` mixture. It may not reach target, but will return as far as it
    got. The return value is two parallel arrays for the probabilities with
    known equilibria and the equilibria.

    Parameters
    ----------
    game0 : RsGame
        The first game that's merged. Represents the payoffs when `peq` is 0.
    game1 : RsGame
        The second game that's merged. Represents the payoffs when `peq` is 1.
    peq : float
        The amount that the two games are merged such that `eqm` is an
        equilibrium. Must be in [0, 1].
    eqm : ndarray
        An equilibrium when `game0` and `game1` are merged a `peq` fraction.
    target : float
        The desired mixture probability to have an equilibrium at.
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
    egame = rsgame.empty_copy(game0)
    eqm = np.asarray(eqm, float)
    utils.check(
        egame.is_mixture(eqm), "equilibrium wasn't a valid mixture")
    utils.check(
        regret.mixture_regret(
            rsgame.mix(game0, game1, peq), eqm) <= regret_thresh + 1e-7,
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
            ode, [peq, target], eqm, events=events, **ivp_args)
    return res.t, egame.trim_mixture_support(res.y.T, thresh=0)


def trace_interpolate(game0, game1, peqs, eqa, targets, **kwargs): # pylint: disable=too-many-locals
    """Get an equilibrium at a specific time

    Parameters
    ----------
    game0 : RsGame
        The game to get data from when the mixture probability is 0.
    game1 : RsGame
        The game to get data from when the mixture probability is 1.
    peqs : [float]
        A parallel list of probabilities for each equilibria in a continuous
        trace.
    eqa : [eqm]
        A parallel list of equilibria for each probability representing
        continuous equilibria for prob mixture games.
    targets : [float]
        The probabilities to compute an equilibria at.
    kwargs : options
        The same options as `trace_equilibrium`.
    """
    peqs = np.asarray(peqs, float)
    eqa = np.asarray(eqa, float)
    targets = np.asarray(targets, float)

    # Make everything sorted
    if np.all(np.diff(peqs) <= 0):
        peqs = peqs[::-1]
        eqa = eqa[::-1]
    order = np.argsort(targets)
    targets = targets[order]

    utils.check(
        np.all(np.diff(peqs) >= 0), 'trace probabilities must be sorted')
    utils.check(
        peqs[0] <= targets[0] and targets[-1] <= peqs[-1],
        'targets must be internal to trace')

    result = np.empty((targets.size, game0.num_strats))
    scan = zip(utils.subsequences(peqs), utils.subsequences(eqa))
    (pi1, pi2), (eqm1, eqm2) = next(scan)
    for target, i in zip(targets, order):
        while target > pi2:
            (pi1, pi2), (eqm1, eqm2) = next(scan)
        (*_, pt1), (*_, eqt1) = trace_equilibrium( # pylint: disable=too-many-star-expressions
            game0, game1, pi1, eqm1, target, **kwargs)
        (*_, pt2), (*_, eqt2) = trace_equilibrium( # pylint: disable=too-many-star-expressions
            game0, game1, pi2, eqm2, target, **kwargs)
        if np.isclose(pt1, target) and np.isclose(pt2, target):
            mixgame = rsgame.mix(game0, game1, target)
            _, _, result[i] = min(
                (regret.mixture_regret(mixgame, eqt1), 0, eqt1),
                (regret.mixture_regret(mixgame, eqt2), 1, eqt2))
        elif np.isclose(pt1, target):
            result[i] = eqt1
        elif np.isclose(pt2, target):
            result[i] = eqt2
        else: # pragma: no cover
            raise ValueError('ode solving failed to reach prob')
    return result
