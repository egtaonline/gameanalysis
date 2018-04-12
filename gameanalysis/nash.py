"""Module for computing nash equilibria"""
import functools
import itertools
import multiprocessing
import warnings

import numpy as np
from numpy import linalg
from scipy import optimize

from gameanalysis import collect
from gameanalysis import fixedpoint
from gameanalysis import regret
from gameanalysis import utils


def pure_nash(game, *, epsilon=0):
    """Returns an array of all pure nash profiles"""
    eqa = [prof[None] for prof in game.profiles()
           if regret.pure_strategy_regret(game, prof) <= epsilon]
    if eqa: # pylint: disable=no-else-return
        return np.concatenate(eqa)
    else:
        return np.empty((0, game.num_strats))


def min_regret_profile(game):
    """Finds the profile with the confirmed lowest regret

    An error will be raised if there are no profiles with a defined regret.
    """
    regs = np.fromiter(  # pragma: no branch
        (regret.pure_strategy_regret(game, prof)
         for prof in game.profiles()), float, game.num_profiles)
    return game.profiles()[np.nanargmin(regs)]


def min_regret_grid_mixture(game, points):
    """Finds the mixed profile with the confirmed lowest regret

    The search is done over a grid with `points` per dimensions.

    Arguments
    ---------
    points : int > 1
        Number of points per dimension to search.
    """
    mixes = game.grid_mixtures(points)
    regs = np.fromiter((regret.mixture_regret(game, mix)  # pragma: no branch
                        for mix in mixes), float, mixes.shape[0])
    return mixes[np.nanargmin(regs)]


def min_regret_rand_mixture(game, mixtures):
    """Finds the mixed profile with the confirmed lowest regret

    The search is done over a random sampling of `mixtures` mixed profiles.

    Arguments
    ---------
    mixtures : int > 0
        Number of mixtures to evaluate the regret of.
    """
    utils.check(mixtures > 0, 'mixtures must be greater than 0')
    mixes = game.random_mixtures(mixtures)
    regs = np.fromiter((regret.mixture_regret(game, mix)  # pragma: no branch
                        for mix in mixes), float, mixtures)
    return mixes[np.nanargmin(regs)]


def replicator_dynamics(game, mix, *, max_iters=10000, converge_thresh=1e-8,
                        slack=1e-3):
    """Replicator Dynamics

    Run replicator dynamics on a game starting at mix. Replicator dynamics may
    not converge, and so the resulting mixture may not actually represent a
    nash equilibrium.

    Arguments
    ---------
    game : Game
        The game to run replicator dynamics on. Game must support
        `deviation_payoffs`.
    mix : mixture
        The mixture to initialize replicator dynamics with.
    max_iters : int
        Replicator dynamics may never converge and this prevents replicator
        dynamics from running forever.
    converge_thresh : float
        This will terminate early if successive updates differ with a norm
        smaller than `converge_thresh`.
    slack : float
        For repliactor dynamics to operate, it must know the minimum and
        maximum payoffs for a role such that deviations always have positive
        probability. This is the proportional slack that given relative to the
        minimum and maximum payoffs. This has an effect on convergence, but the
        actual effect isn't really know.
    """
    minp = game.min_role_payoffs()
    maxp = game.max_role_payoffs()

    for _ in range(max_iters):
        old_mix = mix.copy()
        dev_pays = game.deviation_payoffs(mix)
        np.minimum(minp, np.minimum.reduceat(dev_pays, game.role_starts), minp)
        np.maximum(maxp, np.maximum.reduceat(dev_pays, game.role_starts), maxp)
        resid = slack * (maxp - minp)
        resid[np.isclose(resid, 0)] = slack
        offset = np.repeat(minp - resid, game.num_role_strats)
        mix *= dev_pays - offset
        mix /= np.add.reduceat(
            mix, game.role_starts).repeat(game.num_role_strats)
        if linalg.norm(mix - old_mix) <= converge_thresh:
            break

    # Probabilities are occasionally negative
    return game.mixture_project(mix)


def regret_minimize(game, mix, *, gtol=1e-8):
    """A pickleable object to find Nash equilibria

    This method uses constrained convex optimization to to attempt to solve a
    proxy for the nonconvex regret minimization. Since this may converge to a
    local optimum, it may return a mixture that is not an approximate
    equilibrium.

    Arguments
    ---------
    game : Game
        The game to run replicator dynamics on. Game must support
        `deviation_payoffs`.
    mix : mixture
        The mixture to initialize replicator dynamics with.
    gtol : float
        The gradient tolerance used for optimization convergence. See
        `scipy.optimize.minimize`.
    """
    scale = np.repeat(game.max_role_payoffs() - game.min_role_payoffs(),
                      game.num_role_strats)
    scale[np.isclose(scale, 0)] = 1  # In case payoffs are the same
    offset = game.min_role_payoffs().repeat(game.num_role_strats)

    def grad(mixture):
        """Gradient of the objective function"""
        # We assume that the initial point is in a constant sum subspace, and
        # so project the gradient so that any gradient step maintains that
        # constant step. Thus, sum to 1 is not one of the penalty terms

        # Because deviation payoffs uses log space, we max with 0 just for the
        # payoff calculation
        dev_pay, dev_jac = game.deviation_payoffs(
            np.maximum(mixture, 0), jacobian=True, full_jacobian=True)

        # Normalize
        dev_pay = (dev_pay - offset) / scale
        dev_jac /= scale[:, None]

        # Gains from deviation (objective)
        gains = np.maximum(
            dev_pay - np.add.reduceat(
                mixture * dev_pay,
                game.role_starts).repeat(game.num_role_strats),
            0)
        obj = gains.dot(gains) / 2

        gains_jac = (dev_jac - dev_pay - np.add.reduceat(
            mixture[:, None] * dev_jac, game.role_starts).repeat(
                game.num_role_strats, 0))
        grad = gains.dot(gains_jac)

        # Project grad so steps stay in the simplotope
        grad -= np.repeat(np.add.reduceat(grad, game.role_starts) /
                          game.num_role_strats, game.num_role_strats)

        return obj, grad

    with warnings.catch_warnings():
        # XXX For some reason, line-search in optimize throws a
        # run-time warning when things get very small negative.  This
        # is potentially a error with the way we compute gradients, but
        # it's not reproducible, so we ignore it.
        warnings.simplefilter(
            'ignore', optimize.linesearch.LineSearchWarning)
        mix = optimize.minimize(
            grad, mix, jac=True, bounds=[(0, 1)] * game.num_strats,
            options={'gtol': gtol}).x
        # Project it onto the simplex, it might not be due to the penalty
        return game.mixture_project(mix)


def fictitious_play(game, mix, *, max_iters=10000, converge_thresh=1e-8):
    """Run fictitious play on a mixture

    In fictitious play, players continually best respond to the empirical
    distribution of their opponents at each round.
    """
    empirical = mix.copy()
    for i in range(2, max_iters + 2):
        update = (game.best_response(empirical) - empirical) / i
        empirical += update
        if np.linalg.norm(update) < converge_thresh:
            break
    return empirical


# TODO Implement regret based equilibria finding, i.e. running a zero regret
# algorithm on payoffs.
# TODO Implement other equilibria finding methods that are found in gambit


def scarfs_algorithm(game, mix, *, regret_thresh=1e-3, disc=8):
    """Uses fixed point method to find nash eqm

    This is guaranteed to find an equilibrium with regret below regret_thresh,
    however, it's guaranteed convergence is assured by potentially exponential
    running time, and therefore is not recommended unless you're willing to
    wait. The underlying algorithm is solving for an approximate fixed point.

    Arguments
    ---------
    game : Game
        The game to run replicator dynamics on. Game must support
        `deviation_payoffs`.
    mix : mixture
        The mixture to initialize replicator dynamics with.
    regret_thresh : float
        The maximum regret of the returned mixture.
    disc : int
        The initial discretization of the mixture. A lower initial
        discretization means fewer possible starting points for search in the
        mixture space, but is likely to converge faster as the search at higher
        discretization will be seeded with an approximate equilibrium from a
        lower discretization. For example, with `disc=2` there are only
        `game.num_strats - game.num_roles + 1` possible starting points.
    """
    def eqa_func(mixture):
        """Equilibrium fixed point function"""
        mixture = game.mixture_from_simplex(mixture)
        gains = np.maximum(regret.mixture_deviation_gains(game, mixture), 0)
        result = (mixture + gains) / (1 + np.add.reduceat(
            gains, game.role_starts).repeat(game.num_role_strats))
        return game.mixture_to_simplex(result)

    disc = min(disc, 8)
    reg = regret.mixture_regret(game, mix)
    while reg > regret_thresh:
        mix = game.mixture_from_simplex(fixedpoint.fixed_point(
            eqa_func, game.mixture_to_simplex(mix), disc=disc))
        reg = regret.mixture_regret(game, mix)
        disc *= 2

    return mix


_AVAILABLE_METHODS = {
    'replicator': replicator_dynamics,
    'fictitious': fictitious_play,
    'optimize': regret_minimize,
    'scarf': scarfs_algorithm,
}


def _find_eqm(function, game, mix, **kwargs):
    """Pickleable function to find an equilibrium"""
    return function(game, mix, **kwargs)


def mixed_nash( # pylint: disable=too-many-locals
        game, *, regret_thresh=1e-3, dist_thresh=0.1, grid_points=2,
        random_restarts=0, processes=0, min_reg=False, at_least_one=False,
        **methods):
    """Finds role-symmetric mixed Nash equilibria

    This is the intended front end for nash equilibria finding, wrapping the
    individual methods in a convenient front end that also support parallel
    execution. Scipy optimize, and hence nash finding with the optimize method
    is NOT thread safe. This can be mitigated by running nash finding in a
    separate process (by setting processes > 0) if the game is pickleable.

    Arguments
    ---------
    regret_thresh : float, optional
        The threshold to consider an equilibrium found.
    dist_thresh : float, optional
        The threshold for considering equilibria distinct.
    grid_points : int > 1, optional
        The number of grid points to use for mixture seeds. two implies just
        pure mixtures, more will be denser, but scales exponentially with the
        dimension.
    random_restarts : int, optional
        The number of random initializations.
    processes : int or None, optional
        Number of processes to use when finding Nash equilibria. If 0 (default)
        run nash finding in the current process. This will work with any game
        but is not thread safe for the optimize method. If greater than zero or
        none, the game must be pickleable and nash finding will be run in
        `processes` processes. Passing None will use the number of current
        processors.
    min_reg : bool, optional
        If True, and no equilibria are found with the methods specified, return
        the point with the lowest empirical regret. This is ignored if
        at_least_one is True
    at_least_one : bool, optional
        If True, always return an equilibrium. This will use the fixed point
        method with increasingly smaller tolerances until an equilibrium with
        small regret is found. This may take an exceedingly long time to
        converge, so use with caution.
    **methods : {'replicator', 'optimize', 'scarf', 'fictitious'}={options}
        All methods to use can be specified as key word arguments to additional
        options for that method, e.g. mixed_nash(game,
        replicator={'max_iters':100}). To use the default options for a method,
        simply pass a falsey value i.e. {}, None, False. If no methods are
        specified, this will use both replicator dynamics and regret
        optimization as they tend to be reasonably fast and find different
        equilibria. Scarfs algorithm is almost never recommended to be passed
        here, as it will be called if at_least_one is True and only after
        failing with a faster method and only called once.

    Returns
    -------
    eqm : ndarray
        A two dimensional array with mixtures that have regret below
        `regret_thresh` and have norm difference of at least `dist_thresh`.
    """
    umix = game.uniform_mixture()
    utils.check(
        not np.isnan(game.deviation_payoffs(umix)).any(),
        'Nash finding only works on game with full deviation data')
    utils.check(
        processes is None or processes >= 0,
        'processes must be non-negative or None')
    utils.check(
        all(m in _AVAILABLE_METHODS for m in methods),
        'specified a invalid method {}', methods)

    initial_points = list(itertools.chain(
        [umix],
        game.grid_mixtures(grid_points),
        game.biased_mixtures(),
        game.role_biased_mixtures(),
        game.random_mixtures(random_restarts)))
    equilibria = collect.mcces(dist_thresh)
    best = [np.inf, -1, None]
    chunksize = len(initial_points) if processes == 1 else 4

    # Initialize pickleable methods
    methods = methods or {'replicator': {}, 'optimize': {}}
    methods = (
        functools.partial(
            _find_eqm, _AVAILABLE_METHODS[meth], game, **(opts or {}))
        for meth, opts in methods.items())

    # what to do with each candidate equilibrium
    def process(i, eqm):
        """Process an equilibrium"""
        reg = regret.mixture_regret(game, eqm)
        if reg < regret_thresh:
            equilibria.add(eqm, reg)
        best[:] = min(best, [reg, i, eqm])

    if processes == 0:
        for i, (meth, init) in enumerate(itertools.product(
                methods, initial_points)):
            process(i, meth(init))
    else:
        with multiprocessing.Pool(processes) as pool:
            for i, eqm in enumerate(itertools.chain.from_iterable(
                    pool.imap_unordered(m, initial_points, chunksize=chunksize)
                    for m in methods)):
                process(i, eqm)

    if at_least_one and not equilibria:
        # Initialize at best found
        eqm = scarfs_algorithm(game, best[2], regret_thresh=regret_thresh)
        reg = regret.mixture_regret(game, eqm)
        equilibria.add(eqm, reg)
    elif min_reg and not equilibria:
        reg, _, eqm = best
        equilibria.add(eqm, reg)

    if equilibria: # pylint: disable=no-else-return
        return np.array([e for e, _ in equilibria])
    else:
        return np.empty((0, game.num_strats))
