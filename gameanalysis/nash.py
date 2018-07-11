"""Module for computing nash equilibria"""
import contextlib
import functools
import itertools
import multiprocessing
import sys
import warnings

import numpy as np
from scipy import optimize

from gameanalysis import collect
from gameanalysis import fixedpoint
from gameanalysis import regret
from gameanalysis import utils


# FIXME Remove
def pure_nash(game, *, epsilon=0):
    """Returns an array of all pure nash profiles

    This is the old syntax.
    """
    return pure_equilibria(game, epsilon=epsilon)


def pure_equilibria(game, *, epsilon=0):
    """Returns an array of all pure nash profiles"""
    eqa = [prof for prof in game.profiles()
           if regret.pure_strategy_regret(game, prof) <= epsilon]
    return np.stack(eqa) if eqa else np.empty((0, game.num_strats))


def _nan_to_inf(val):
    """Convert nans to inf for min_reg"""
    return np.inf if np.isnan(val) else val


def min_regret_profile(game):
    """Finds the profile with the confirmed lowest regret

    An error will be raised if there are no profiles with a defined regret.
    """
    utils.check(not game.is_empty(), 'Game must have a profile')
    reg, _, prof = min(
        (_nan_to_inf(regret.pure_strategy_regret(game, prof)), i, prof)
        for i, prof in enumerate(game.profiles()))
    utils.check(not np.isinf(reg), 'No profiles had valid regret')
    return prof


# TODO Remove
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


# TODO Remove
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


class _Sentinel(Exception):
    """A sentinel for timeouts"""
    pass


def _iter_nash(
        *, def_max_iters=10000, def_converge_thresh=1e-8, def_converge_disc=1,
        def_timeout=None, first_check=1000):
    """Decorator for iterative nash finding methods"""
    # TODO potentially also take regret_thresh and check every that regret is
    # below thresh at most every second?
    def wrap(func):
        """The actual decorator"""
        @functools.wraps(func)
        def wrapped( # pylint: disable=too-many-arguments
                game, prof, *args, max_iters=def_max_iters,
                timeout=def_timeout, converge_thresh=def_converge_thresh,
                converge_disc=def_converge_disc, **kwargs):
            """The wrapped nash function"""
            converge_thresh *= np.sqrt(2 * game.num_roles)
            smooth_update = np.zeros(game.num_strats)
            mix = last_mix = game.trim_mixture_support(prof, thresh=0)
            with contextlib.suppress(_Sentinel), utils.timeout(
                timeout, _Sentinel):
                for i, mix in zip(
                        range(max_iters), func(game, prof, *args, **kwargs)):
                    smooth_update *= 1 - converge_disc
                    smooth_update += converge_disc * (mix - last_mix)
                    if (i * converge_disc > first_check and
                            np.linalg.norm(smooth_update) < converge_thresh):
                        break
                    last_mix = mix.copy()
            return game.mixture_project(mix)
        return wrapped
    return wrap


@_iter_nash(first_check=100)
def replicator_dynamics(game, mix, *, slack=1e-3):
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
    slack : float
        For repliactor dynamics to operate, it must know the minimum and
        maximum payoffs for a role such that deviations always have positive
        probability. This is the proportional slack that given relative to the
        minimum and maximum payoffs. This has an effect on convergence, but the
        actual effect isn't really know.
    """
    mix = mix.copy()
    minp = game.min_role_payoffs().copy()
    maxp = game.max_role_payoffs().copy()

    while True:
        dev_pays = game.deviation_payoffs(mix)
        np.minimum(minp, np.minimum.reduceat(dev_pays, game.role_starts), minp)
        np.maximum(maxp, np.maximum.reduceat(dev_pays, game.role_starts), maxp)
        resid = slack * (maxp - minp)
        resid[np.isclose(resid, 0)] = slack
        offset = np.repeat(minp - resid, game.num_role_strats)
        mix *= dev_pays - offset
        mix /= np.add.reduceat(
            mix, game.role_starts).repeat(game.num_role_strats)
        yield mix


@_iter_nash(
    def_max_iters=100000, def_converge_thresh=1e-6, def_converge_disc=0.2,
    first_check=10)
def regret_matching(game, profile, *, slack=0.1): # pylint: disable=too-many-locals
    """Regret matching

    Run regret matching. This selects new strategies to play proportionally to
    the gain they receive from deviating from the current profile.

    Parameters
    ----------
    game : Game
        The game to run replicator dynamics on. Game must support
        `deviation_payoffs`.
    profile : array_like
        The initial profile to start with.
    slack : float, optional
        The amount to make sure agents will always play their last played
        strategy. (0, 1)
    """
    strat_players = game.num_role_players.repeat(
        game.num_role_strats).astype(float)

    profile = np.asarray(profile, int)
    mean_gains = np.zeros(game.num_devs)
    mean_mix = profile / strat_players
    mus = np.full(game.num_roles, np.finfo(float).tiny)

    for i in itertools.count(1): # pragma: no branch
        # Regret matching
        gains = regret.pure_strategy_deviation_gains(game, profile)
        gains *= profile.repeat(game.num_strat_devs)
        mean_gains += (gains - mean_gains) / i
        np.maximum(np.maximum.reduceat(
            np.add.reduceat(np.maximum(gains, 0), game.dev_strat_starts),
            game.role_starts), mus, mus)

        # For each strategy sample from regret matching distribution
        new_profile = np.zeros(game.num_strats, int)
        for rgains, prof, nprof, norm, strats in zip(
                np.split(np.maximum(mean_gains, 0), game.dev_role_starts[1:]),
                np.split(profile, game.role_starts[1:]),
                np.split(new_profile, game.role_starts[1:]),
                mus * (1 + slack),
                game.num_role_strats):
            probs = rgains / norm
            probs[np.arange(0, probs.size, strats + 1)] = 1 - np.add.reduceat(
                probs, np.arange(0, probs.size, strats))
            for count, prob in zip(prof, np.split(probs, strats)):
                nprof += np.random.multinomial(count, prob)

        # Test for convergence
        profile = new_profile
        mean_mix += (profile / strat_players - mean_mix) / (i + 1)
        yield mean_mix


def _regret_matching_mix(game, mix, **kwargs):
    """Regret matching that takes a mixture"""
    return regret_matching(game, game.max_prob_prof(mix), **kwargs)


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
    gtol : float, optional
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
        return game.mixture_project(mix)


@_iter_nash(def_converge_thresh=1e-6)
def fictitious_play(game, mix):
    """Run fictitious play on a mixture

    In fictitious play, players continually best respond to the empirical
    distribution of their opponents at each round. This tends to have very slow
    convergence.

    Parameters
    ----------
    game : RsGame
        The game to compute an equilibrium of.
    mix : array_like
        The initial mixture to respond to.
    """
    empirical = mix.copy()
    for i in itertools.count(2): # pragma: no branch
        empirical += (game.best_response(empirical) - empirical) / i
        yield empirical


@_iter_nash()
def _multiplicative_weights(game, mix, func, epsilon):
    """Generic multiplicative weights algorithm

    Parameters
    ----------
    game : RsGame
        The game to compute an equilibrium of.
    mix : ndarray
        The initial mixture for searching.
    func : (RsGame, ndarray) -> ndarray
        Function that takes a game and a mixture and returns an unbiased
        estimate of the payoff to each strategy when opponents play according
        to the mixture.
    epsilon : float
        The rate of update for new payoffs. Convergence results hold when
        epsilon in [0, 3/5].
    """
    average = mix.copy()
    # This is done in log space to prevent weights zeroing out for limit cycles
    with np.errstate(divide='ignore'):
        log_weights = np.log(mix)
    learning = np.log(1 + epsilon)

    for i in itertools.count(2): # pragma: no branch
        pays = func(game, np.exp(log_weights))
        log_weights += pays * learning
        log_weights -= np.logaddexp.reduceat(
            log_weights, game.role_starts).repeat(game.num_role_strats)
        average += (np.exp(log_weights) - average) / i
        yield average


def _mw_dist(game, mix):
    """Distributional multiplicative weights payoff function"""
    return game.deviation_payoffs(mix)


def multiplicative_weights_dist(
        game, mix, *, epsilon=0.5, max_iters=10000, converge_thresh=1e-8,
        **kwargs):
    """Compute an equilibrium using the distribution multiplicative weights

    This version of multiplicative weights takes the longest per iteration, but
    also has less variance and likely converges better.

    Parameters
    ----------
    game : RsGame
        The game to compute an equilibrium of.
    mix : ndarray
        The initial mixture for searching.
    epsilon : float, optional
        The rate of update for new payoffs. Convergence results hold when
        epsilon in [0, 3/5].
    """
    return _multiplicative_weights( # pylint: disable=unexpected-keyword-arg
        game, mix, _mw_dist, epsilon, max_iters=max_iters,
        converge_thresh=converge_thresh, converge_disc=1, **kwargs)


def _mw_stoch(game, mix):
    """Stochastic multiplicative weights payoff function"""
    prof = game.random_profile(mix)
    devs = regret.pure_strategy_deviation_pays(game, prof)
    counts = prof.repeat(game.num_strat_devs)
    sum_dev = np.bincount(game.dev_to_indices, devs * counts, game.num_strats)
    return sum_dev / game.num_role_players.repeat(game.num_role_strats)


def multiplicative_weights_stoch(
        game, mix, *, epsilon=0.5, max_iters=100000, converge_thresh=1e-7,
        converge_disc=0.2, **kwargs):
    """Compute an equilibrium using the stochastic multiplicative weights

    This version of multiplicative weights takes a medium amount of time per
    iteration and converges better than the bandit version.

    Parameters
    ----------
    game : RsGame
        The game to compute an equilibrium of.
    mix : ndarray
        The initial mixture for searching.
    epsilon : float, optional
        The rate of update for new payoffs. Convergence results hold when
        epsilon in [0, 3/5].
    """
    return _multiplicative_weights( # pylint: disable=unexpected-keyword-arg
        game, mix, _mw_stoch, epsilon, max_iters=max_iters,
        converge_thresh=converge_thresh, converge_disc=converge_disc, **kwargs)


def _mw_bandit(min_prob, game, mix):
    """Multi-armed bandit multiplicative weights payoff function"""
    # Minimum probability to force exploration
    mix = game.minimum_prob(mix, min_prob=min_prob)
    prof = game.random_profile(mix)
    pay = (game.get_payoffs(prof) * prof /
           game.num_role_players.repeat(game.num_role_strats))
    return pay / mix


def multiplicative_weights_bandit(
        game, mix, *, epsilon=0.5, max_iters=100000, converge_thresh=1e-7,
        converge_disc=0.1, min_prob=1e-3, **kwargs):
    """Compute an equilibrium using the bandit multiplicative weights

    This version of multiplicative weights takes the shortest amount of time
    per iteration but is less likely to converge to an equilibrium.

    Parameters
    ----------
    game : RsGame
        The game to compute an equilibrium of.
    mix : ndarray
        The initial mixture for searching.
    epsilon : float, optional
        The rate of update for new payoffs. Convergence results hold when
        epsilon in [0, 3/5].
    min_prob : float, optional
        The minimum probability a mixture is given when sampling from a
        profile. This is necessary to prevent the bandit from getting stuck in
        pure mixtures, and to bound the potential payoffs, as payoffs are
        divided by the probability to get unbiased estimates. However, larger
        settings of this will bias the overall result. (0, 1)
    """
    return _multiplicative_weights( # pylint: disable=unexpected-keyword-arg
        game, mix, functools.partial(_mw_bandit, min_prob), epsilon,
        max_iters=max_iters, converge_thresh=converge_thresh,
        converge_disc=converge_disc, **kwargs)


# TODO Implement other equilibria finding methods that are found in gambit


@_iter_nash(def_converge_thresh=0, def_max_iters=sys.maxsize)
def scarfs_algorithm(game, mix, *, regret_thresh=1e-2, disc=8):
    """Uses fixed point method to find nash eqm

    This is guaranteed to find an equilibrium with regret below regret_thresh
    if given enough time. However, it's guaranteed convergence is assured by
    potentially exponential running time, and therefore is not recommended
    unless you're willing to wait. The underlying algorithm is solving for an
    approximate Nash fixed point with greater and great approximation until its
    regret is below the threshold.

    Arguments
    ---------
    game : Game
        The game to run replicator dynamics on. Game must support
        `deviation_payoffs`.
    mix : mixture
        The mixture to initialize replicator dynamics with.
    regret_thresh : float, optional
        The maximum regret of the returned mixture.
    disc : int, optional
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
        yield mix

    # Two yields in a row means convergence
    yield mix


def _noop(_game, mix):
    """A noop for checking regret"""
    return mix


def _initial_mixtures(game):
    """Return generator of initial mixtures"""
    return itertools.chain(
        [game.uniform_mixture()],
        game.biased_mixtures(),
        game.role_biased_mixtures())


_STYLES = frozenset([
    'fast', 'fast*', 'more', 'more*', 'best', 'best*', 'one'])
_STYLES_STR = ', '.join(_STYLES)


def _serial_nash_func(game, spec):
    """Serializable function for nash finding"""
    func, mix, req = spec
    return req, func(game, mix)


def _required(game):
    """Required methods for due diligence"""
    return itertools.chain(
        ((regret_minimize, mix) for mix in itertools.chain(
            _initial_mixtures(game),
            game.pure_mixtures())),
        ((_noop, mix) for mix in game.pure_mixtures()))


def _more(game, reg):
    """Extra methods for `more`"""
    return itertools.chain.from_iterable(
        ((func, mix) for mix in _initial_mixtures(game)) for func in [
            functools.partial(scarfs_algorithm, timeout=60, regret_thresh=reg),
            replicator_dynamics, multiplicative_weights_dist, fictitious_play,
            _regret_matching_mix, multiplicative_weights_stoch,
            multiplicative_weights_bandit])


def _best(game, reg):
    """Extra methods for `best`"""
    return itertools.chain(
        _more(game, reg),
        [(functools.partial(
            scarfs_algorithm, regret_thresh=reg, timeout=30 * 60),
          game.uniform_mixture())])


def _one(game, reg):
    """Extra methods for `one`"""
    return itertools.chain(
        _more(game, reg),
        [(functools.partial(scarfs_algorithm, regret_thresh=reg),
          game.uniform_mixture())])


def mixed_equilibria( # pylint: disable=too-many-locals
        game, style='best', *, regret_thresh=1e-2, dist_thresh=0.1,
        processes=None):
    """Compute mixed equilibria

    Parameters
    ----------
    game : RsGame
        Game to compute equilibria of.
    style : str, optional
        The style of equilibria funding to run. Available styles are:

        fast   - run minimal algorithms and return nothing on failure
        more   - run minimal and if nothing run other reasonable algorithms
        best   - run extra and if nothing run exponential with timeout
        one    - run extra and if nothing run exponential
        <any>* - if nothing found, return minimum regret
    regret_thresh : float, optional
        Minimum regret for a mixture to count as an equilibrium.
    dist_thresh : float, optional
        Minimum role norm for equilibria to be considered distinct. [0, 1]
    processes : int, optional
        Number of processes to compute equilibria with. If None, all available
        processes will be used.
    """
    utils.check(style in _STYLES, 'style {} not one of {}', style, _STYLES_STR)
    utils.check(
        processes is None or processes > 0,
        'processes must be positive or None')
    # TODO Is there a better interface for checking dev payoffs
    utils.check(
        not np.isnan(game.deviation_payoffs(game.uniform_mixture())).any(),
        'Nash finding only works on game with full deviation data')

    seq = 0
    req = 0
    best = [np.inf, 0, None]

    equilibria = collect.mcces(dist_thresh * np.sqrt(2 * game.num_roles))
    func = functools.partial(_serial_nash_func, game)
    extra = {
        'fast': lambda _, __: (),
        'more': _more,
        'best': _best,
        'one': _one,
    }[style.rstrip('*')](game, regret_thresh)

    def process_req(tup):
        """Count required methods"""
        nonlocal req
        req += 1
        return tup + (True,)

    with multiprocessing.Pool(processes) as pool:
        for preq, eqm in pool.imap_unordered(func, itertools.chain(
                map(process_req, _required(game)),
                (tup + (False,) for tup in extra))):
            seq += 1
            req -= preq
            reg = regret.mixture_regret(game, eqm)
            best[:] = min(best, [reg, seq, eqm[None]])
            if reg < regret_thresh:
                equilibria.add(eqm, reg)
            if not req and equilibria:
                return np.stack([e for e, _ in equilibria])

    assert not req
    return best[-1] if style.endswith('*') else np.empty((0, game.num_strats))


_AVAILABLE_METHODS = {
    'replicator': replicator_dynamics,
    'fictitious': fictitious_play,
    'matching': _regret_matching_mix,
    'optimize': regret_minimize,
    'regret_dist': multiplicative_weights_dist,
    'regret_stoch': multiplicative_weights_stoch,
    'regret_bandit': multiplicative_weights_bandit,
}


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

    This is the old style nash finding and provides more options. For new
    methods, mixture_equilibria is the preferred interface.

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
        functools.partial(_AVAILABLE_METHODS[meth], game, **(opts or {}))
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

    if equilibria: # pylint: disable=no-else-return
        return np.array([e for e, _ in equilibria])
    elif at_least_one:
        return scarfs_algorithm( # pylint: disable=unsubscriptable-object
            game, best[-1], regret_thresh=regret_thresh)[None]
    elif min_reg:
        return best[-1][None] # pylint: disable=unsubscriptable-object
    else:
        return np.empty((0, game.num_strats))
