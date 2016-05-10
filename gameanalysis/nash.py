"""Module for computing nash equilibria"""
import itertools
import math
import multiprocessing

import numpy as np
from numpy import linalg
from scipy import optimize

from gameanalysis import regret


_TINY = np.finfo(float).tiny


def pure_nash(game, epsilon=0, as_array=False):
    """Returns a generator of all pure-strategy epsilon-Nash equilibria."""
    return (game.as_profile(profile, as_array=as_array, verify=False)
            for profile in game.profiles(as_array=None)
            if regret.pure_strategy_regret(game, profile) <= epsilon)


def min_regret_profile(game):
    """Finds the profile with the confirmed lowest regret.

    An error will be raised if there are no profiles with a defined regret.
    """
    return min((r, i, p) for r, i, p
               in ((regret.pure_strategy_regret(game, prof), i, prof)
                   for i, prof in enumerate(game))
               if not math.isnan(r))[2]


def min_regret_grid_mixture(game, points, as_array=False):
    """Finds the mixed profile with the confirmed lowest regret

    The search is done over a grid with `points` per dimensions.

    Arguments
    ---------
    points : int > 1
        Number of points per dimension to search.
    """
    return game.as_mixture(
        min((regret.mixture_regret(game, mix), i, mix)
            for i, mix
            in enumerate(game.grid_mixtures(points, as_array=True)))[2],
        as_array=as_array)


def min_regret_rand_mixture(game, mixtures, as_array=False):
    """Finds the mixed profile with the confirmed lowest regret

    The search is done over a random sampling of `mixtures` mixed profiles.

    Arguments
    ---------
    mixtures : int > 0
        Number of mixtures to evaluate the regret of.
    """
    return game.as_mixture(
        min((regret.mixture_regret(game, mix), i, mix)
            for i, mix
            in enumerate(game.random_mixtures(mixtures, as_array=True)))[2],
        as_array=as_array)


def mixed_nash(game, regret_thresh=1e-3, dist_thresh=1e-3, random_restarts=0,
               processes=None, at_least_one=False,
               as_array=False, **methods):
    """Finds role-symmetric, mixed Nash equilibria

    This method first tries replicator dynamics, but falls back to brute force
    if replicator dynamics fails to find an equilibrium.

    Arguments
    ---------
    regret_thresh : float
        The threshold to consider an equilibrium found.
    dist_thresh : float
        The threshold for considering equilibria distinct.
    random_restarts : int
        The number of random initializations for replicator dynamics.
    processes : int
        Number of processes to use when running replicator dynamics. If None,
        all processes are used.
    methods : [str] or {str: {...}}
        The methods to use to converge to an equilibrium. Methods should be an
        iterable of strings. Optionally, it can be a dictionary with extra
        options for each of the methods. If None, defaults to using all
        methods. This will take longer than using only one, but will find the
        most equilibria.
    at_least_one : bool
        Returns the minimum regret mixture found by replicator dynamics if no
        equilibria were within the regret threshold
    as_array : bool
        If true returns equilibria in array form.

    Returns
    -------
    eqm : (Mixture)
        A generator over low regret mixtures
    """
    # TODO could change dist_thresh to specify amount of quantization, which
    # would allow hashing
    initial_seed_funcs = [
        lambda: [game.uniform_mixture(as_array=True)],
        lambda: game.pure_mixtures(as_array=True),
        lambda: game.biased_mixtures(as_array=True),
        lambda: game.role_biased_mixtures(as_array=True),
        lambda: game.random_mixtures(random_restarts, as_array=True)]

    def initial_points():
        return itertools.chain.from_iterable(f() for f in initial_seed_funcs)

    available_methods = {
        'replicator': ReplicatorDynamics,
        'optimize': RegretOptimizer,
    }
    methods = methods or {k: {} for k in available_methods.keys()}
    methods = [available_methods[m](game, **(p or {}))
               for m, p in methods.items()]

    equilibria = []
    best = (np.inf, -1, None)  # Best convergence so far

    with multiprocessing.Pool(processes) as pool:
        for i, eqm in enumerate(itertools.chain.from_iterable(
                pool.imap_unordered(m, initial_points()) for m in methods)):
            reg = regret.mixture_regret(game, eqm)
            if (reg <= regret_thresh and
                    all(linalg.norm(e - eqm, 2) >= dist_thresh
                        for e in equilibria)):
                equilibria.append(eqm)
                yield game.as_mixture(eqm, as_array=as_array, verify=False)
            best = min(best, (reg, i, eqm))

    if not equilibria and at_least_one:
        yield game.as_mixture(best[2], as_array=as_array, verify=False)


# Everything but init is called on pickled object in another process, so it
# will not be included in coverage.
class RegretOptimizer(object):
    def __init__(self, game, gtol=1e-8):
        self.game = game.normalize()
        self.gtol = gtol
        self.penalty = 1

    def grad(self, mix):  # pragma: no cover
        # We assume that the initial point is in a constant sum subspace, and
        # so project the gradient so that any gradient step maintains that
        # constant step. Thus, sum to 1 is not one of the penalty terms
        dev_pay, dev_jac = self.game.deviation_payoffs(mix, verify=False,
                                                       jacobian=True)
        gains = np.maximum(dev_pay - self.game.role_reduce(mix * dev_pay,
                                                           keepdims=True), 0)
        obj = 0.5 * np.sum(gains ** 2)

        dev_diag = np.zeros((self.game.num_role_strats,
                             self.game.astrategies.size))
        dev_diag[np.arange(self.game.num_role_strats),
                 np.arange(self.game.astrategies.size).repeat(
                     self.game.astrategies)] = dev_pay
        dev_diag = dev_diag.repeat(self.game.astrategies, axis=1)
        product_rule = (dev_jac -
                        dev_diag -
                        self.game.role_reduce(dev_jac * mix, 1, keepdims=True))
        grad = np.sum(gains[None] * product_rule, 1)

        # Penalty terms for obj and gradient
        obj += self.penalty * 0.5 * np.sum(np.minimum(mix, 0) ** 2)
        grad += self.penalty * np.minimum(mix, 0)

        # Project grad so steps stay in the appropriate space
        grad -= np.repeat(self.game.role_reduce(grad) / self.game.astrategies,
                          self.game.astrategies)

        return obj, grad

    def __call__(self, mix):  # pragma: no cover
        self.penalty = 1  # reset

        result = None
        closeness = 1
        while closeness > 1e-3 and self.penalty <= (1 << 10):
            # First get an unconstrained result from the optimization
            opt = optimize.minimize(self.grad, mix, method='CG', jac=True,
                                    options={'gtol': self.gtol})
            mix = opt.x
            # Project it onto the simplex
            result = self.game.simplex_project(mix)
            # Maximum projection error over roles
            closeness = math.sqrt(self.game.role_reduce((mix - result) ** 2)
                                  .max())
            # Increase constraint penalty
            self.penalty *= 2

        return result


class ReplicatorDynamics(object):
    """Replicator dynamics

    This will run at most max_iters of replicators dynamics and return unless
    the difference between successive mixtures is less than converge_thresh.
    This is an object to support pickling.
    """
    def __init__(self, game, max_iters=10000, converge_thresh=1e-8):
        self.game = game
        self.max_iters = max_iters
        self.converge_thresh = converge_thresh

    def __call__(self, mix):  # pragma: no cover
        for i in range(self.max_iters):
            old_mix = mix
            mix = (self.game.deviation_payoffs(mix, as_array=True)
                   - self.game.min_payoffs(True).repeat(self.game.astrategies)
                   + _TINY) * mix
            mix /= self.game.role_reduce(mix, keepdims=True)
            if linalg.norm(mix - old_mix) <= self.converge_thresh:
                break

        # Probabilities are occasionally negative
        return self.game.simplex_project(mix)
