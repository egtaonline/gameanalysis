"""Module for computing nash equilibria"""
import itertools
import multiprocessing

import numpy as np
from numpy import linalg
from scipy import optimize

from gameanalysis import collect
from gameanalysis import fixedpoint
from gameanalysis import regret


_TINY = np.finfo(float).tiny


def pure_nash(game, epsilon=0):
    """Returns an array of all pure nash profiles"""
    eqa = [prof[None] for prof in game.profiles
           if regret.pure_strategy_regret(game, prof) <= epsilon]
    if eqa:
        return np.concatenate(eqa)
    else:
        return np.empty((0, game.num_role_strats))


def min_regret_profile(game):
    """Finds the profile with the confirmed lowest regret

    An error will be raised if there are no profiles with a defined regret.
    """
    regs = np.fromiter((regret.pure_strategy_regret(game, prof)
                        for prof in game.profiles), float, game.num_profiles)
    return game.profiles[np.nanargmin(regs)]


def min_regret_grid_mixture(game, points):
    """Finds the mixed profile with the confirmed lowest regret

    The search is done over a grid with `points` per dimensions.

    Arguments
    ---------
    points : int > 1
        Number of points per dimension to search.
    """
    mixes = game.grid_mixtures(points)
    regs = np.fromiter((regret.mixture_regret(game, mix)
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
    mixes = game.random_mixtures(mixtures)
    regs = np.fromiter((regret.mixture_regret(game, mix)
                        for mix in mixes), float, mixtures)
    return mixes[np.nanargmin(regs)]


class RegretOptimizer(object):
    """A pickleable object to find Nash equilibria

    This method uses constrained convex optimization to to attempt to solve a
    proxy for the nonconvex regret minimization."""

    def __init__(self, game, gtol=1e-8):
        self.game = game
        self.scale = game.role_repeat(game.max_payoffs() - game.min_payoffs())
        self.scale[self.scale == 0] = 1  # In case payoffs are the same
        self.offset = game.role_repeat(game.min_payoffs())
        self.gtol = gtol

    def grad(self, mix, penalty):
        # We assume that the initial point is in a constant sum subspace, and
        # so project the gradient so that any gradient step maintains that
        # constant step. Thus, sum to 1 is not one of the penalty terms

        # Because deviation payoffs uses log space, we max with 0 just for the
        # payoff calculation
        dev_pay, dev_jac = self.game.deviation_payoffs(
            np.maximum(mix, 0), jacobian=True, assume_complete=True)

        # Normalize
        dev_pay = (dev_pay - self.offset) / self.scale
        dev_jac /= self.scale[:, None]

        # Gains from deviation (objective)
        gains = np.maximum(dev_pay - self.game.role_reduce(mix * dev_pay,
                                                           keepdims=True), 0)
        obj = np.sum(gains ** 2) / 2

        gains_jac = (dev_jac - dev_pay - self.game.role_reduce(
            mix[:, None] * dev_jac, 0, keepdims=True))
        grad = np.sum(gains[:, None] * gains_jac, 0)

        # Penalty terms for obj and gradient
        obj += penalty * np.sum(np.minimum(mix, 0) ** 2) / 2
        grad += penalty * np.minimum(mix, 0)

        # Project grad so steps stay in the appropriate space
        grad -= self.game.role_repeat(self.game.role_reduce(grad) /
                                      self.game.num_strategies)

        return obj, grad

    def __call__(self, mix):
        # Pass in lambda, and make penalty not a member

        result = None
        penalty = 1
        for _ in range(10):
            # First get an unconstrained result from the optimization
            opt = optimize.minimize(lambda m: self.grad(m, penalty), mix,
                                    method='CG', jac=True,
                                    options={'gtol': self.gtol})
            mix = opt.x
            # Project it onto the simplex, it might not be due to the penalty
            result = self.game.mixture_project(mix)
            if np.allclose(mix, result):
                break
            # Increase constraint penalty
            penalty *= 2

        return result


class ReplicatorDynamics(object):
    """Replicator dynamics

    This will run at most max_iters of replicator dynamics and return unless
    the difference between successive mixtures is less than converge_thresh.
    This is an object to support pickling. Replicator Dynamics needs minimum
    and maximum payoffs in order to project successive iterations into the
    simplex. If these aren't known then they should return inf and -inf
    respectively. Otherwise they can be conservative bounds.
    """

    def __init__(self, game, max_iters=10000, converge_thresh=1e-8,
                 slack=1e-3):
        self.game = game
        self.max_iters = max_iters
        self.converge_thresh = converge_thresh
        self.slack = slack

    def __call__(self, mix):
        minp = self.game.min_payoffs()
        maxp = self.game.max_payoffs()

        for _ in range(self.max_iters):
            old_mix = mix
            dev_pays = self.game.deviation_payoffs(mix, assume_complete=True)
            minp = np.minimum(minp, self.game.role_reduce(dev_pays,
                                                          ufunc=np.minimum))
            maxp = np.maximum(maxp, self.game.role_reduce(dev_pays,
                                                          ufunc=np.maximum))
            slack = self.slack * (maxp - minp)
            slack[slack == 0] = self.slack
            offset = self.game.role_repeat(minp - slack)
            mix = (dev_pays - offset) * mix
            mix /= self.game.role_reduce(mix, keepdims=True)
            if linalg.norm(mix - old_mix) <= self.converge_thresh:
                break

        # Probabilities are occasionally negative
        return self.game.mixture_project(mix)


_AVAILABLE_METHODS = {
    'replicator': ReplicatorDynamics,
    'optimize': RegretOptimizer,
}


def mixed_nash(game, regret_thresh=1e-3, dist_thresh=1e-3, grid_points=2,
               random_restarts=0, processes=1, min_reg=False,
               at_least_one=False, **methods):
    """Finds role-symmetric mixed Nash equilibria

    Arguments
    ---------
    regret_thresh : float
        The threshold to consider an equilibrium found.
    dist_thresh : float
        The threshold for considering equilibria distinct.
    grid_points : int > 1
        The number of grid points to use for mixture seeds. two implies just
        pure mixtures, more will be denser, but scales exponentially with the
        dimension.
    random_restarts : int
        The number of random initializations.
    processes : int or None
        Number of processes to use when finding Nash equilibria. If greater
        than one, the game will need to be pickleable. Passing None will use
        the number of current processors.
    min_reg : bool
        If True, and no equilibria are found with the methods specified, return
        the point with the lowest empirical regret. This is ignored if
        at_least_one is True
    at_least_one : bool
        If True, always return an equilibrium. This will use the fixed point
        method with increasingly smaller tolerances until an equilibrium with
        small regret is found. This may take an exceedingly long time to
        converge, so use with caution.
    **methods : {'replicator', 'optimize'}={options}
        All methods to use can be specified as key word arguments to additional
        options for that method, e.g. mixed_nash(game,
        replicator={'max_iters':100}). To use the default options for a method,
        simply pass a falsey value i.e. {}, None, False. If no methods are
        specified, this will use both replicator dynamics and regret
        optimization as they tend to be reasonably fast and find different
        equilibria.

    Returns
    -------
    eqm : ndarray
        A two dimensional array with mixtures that have regret below
        `regret_thresh` and have norm difference of at least `dist_thresh`.
    """
    assert game.is_complete(), "Nash finding only works on complete games"""
    assert all(m in _AVAILABLE_METHODS for m in methods), \
        "specified a invalid method {}".format(methods)

    initial_points = list(itertools.chain(
        [game.uniform_mixture()],
        game.grid_mixtures(grid_points),
        game.biased_mixtures(),
        game.role_biased_mixtures(),
        game.random_mixtures(random_restarts)))
    equilibria = collect.WeightedSimilaritySet(
        lambda a, b: linalg.norm(a - b) < dist_thresh)
    best = [np.inf, -1, None]
    chunksize = len(initial_points) if processes == 1 else 4

    methods = methods or {'replicator': None, 'optimize': None}
    methods = (_AVAILABLE_METHODS[meth](game, **(opts or {}))
               for meth, opts in methods.items())

    # what to do with each candidate equilibrium
    def process(i, eqm):
        reg = regret.mixture_regret(game, eqm)
        if reg < regret_thresh:
            equilibria.add(eqm, reg)
        best[:] = min(best, [reg, i, eqm])

    if processes == 1:
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
        eqm = game.uniform_mixture()
        reg = regret.mixture_regret(game, eqm)
        disc = 8
        while reg > regret_thresh:
            eqm = _fixed_point_nash(game, eqm, disc)
            reg = regret.mixture_regret(game, eqm)
            disc *= 2
        equilibria.add(eqm, reg)
    elif min_reg and not equilibria:
        reg, _, eqm = best
        equilibria.add(eqm, reg)

    if equilibria:
        return np.concatenate([x[0][None] for x in equilibria])
    else:
        return np.empty((0, game.num_role_strats))


def _fixed_point_nash(game, mix, disc):
    """Uses fixed point method to find nash eqm

    This is guaranteed to find an equilibrium that's within tol od a true
    equilibrium. Therefore, by making tol arbitrarily small, this will find an
    approximate equilibrium. However, it's guaranteed convergence is assured by
    potentially exponential time, and therefore is not recommended unless
    you're willing to wait.
    """
    def eqa_func(mix):
        mix = game.from_simplex(mix)
        gains = np.maximum(regret.mixture_deviation_gains(game, mix), 0)
        result = ((mix + gains) / (1 + game.role_reduce(gains, keepdims=True)))
        return game.to_simplex(result)

    return game.from_simplex(fixedpoint.fixed_point(
        eqa_func, game.to_simplex(mix), disc=disc))
