"Module for using bootstrap in analysis"
import itertools
import multiprocessing
from collections import abc

import numpy as np
import numpy.random as rand

from gameanalysis import regret


def game_function(game, function, num_resamples, percentiles=None,
                  num_returned=None, processes=None):
    """Bootstrap the value of a function over a sample game

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    function : f(Game) -> float or f(Game) -> [float]
        The function of the game to compute. It must be pickleable, and it must
        return either a float or an iterable of floats. If an iterable of
        floats, this bootstrap all indices of the return value independently.
    num_resamples : int
        The number of bootstrap samples. Higher will take longer but also give
        better accuracy.
    percentiles : int or [int]
        The percentiles to compute on the resulting data in [0, 100]. Standard
        percentiles are 95, or [2.5, 97.5]. By default, return all samples.
    num_returned : int (optional)
        The number of float values your function returns. The return value must
        be iterable if this is specified. Default will calculate it.
    processes : int (optional)
        The number of processes to use for computation. By default this is the
        number of cores.

    Returns
    -------
    bootstrap_percentiles : ndarray
        An ndarray of the percentiles from bootstrapping. The shape will depend
        on the number of percentiles and the number of values returned from
        your function.

    Note: if you want all of the bootstrap samples, setting percentiles to
    np.linspace(0, 100, num_resamples) will essentially return a sorted version
    of the bootstrap data.
    """
    func = _BootstrapPickleable(game, function)
    with multiprocessing.Pool(processes) as pool:
        gen = pool.imap_unordered(func, itertools.repeat(None, num_resamples))
        if num_returned is None:
            first = next(gen)
            if not isinstance(first, abc.Sized):
                num_returned = 1
                gen = itertools.chain([first], gen)
            else:
                num_returned = len(first)
                gen = itertools.chain.from_iterable(
                    itertools.chain([first], gen))
        else:
            gen = itertools.chain.from_iterable(gen)

        results = np.fromiter(gen, float, num_returned * num_resamples)
    results = results.reshape((num_resamples, num_returned))
    if percentiles is None:
        results.sort(0)
        return results.T
    else:
        return np.percentile(results, percentiles, 0).T


class _BootstrapPickleable(object):
    def __init__(self, game, function):
        self.game = game
        self.function = function

    def __call__(self, _):
        self.game.resample()
        return self.function(self.game)


def profile_function(game, function, profiles, num_resamples, percentiles,
                     processes=None):
    """Compute a function over profiles

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    function : f(Game, profile) -> float
        The function of the game profile pair to compute. It must be
        pickleable, and it must return a float (e.g. regret.mixture_regret).
    profiles : [Profile] or [Mixture]
        The profiles to compute bootstrap bounds over for function.
    num_resamples : int
        The number of bootstrap samples. Higher will take longer but also give
        better accuracy.
    percentiles : int or [int]
        The percentiles to compute on the resulting data in [0, 100]. Standard
        percentiles are 95, or [2.5, 97.5]. By default, return all samples.
    processes : int (optional)
        The number of processes to use for computation. By default this is the
        number of cores.

    Returns
    -------
    bootstrap_percentiles : ndarray
        An ndarray of the percentiles from bootstrapping for each profile. The
        shape will depend on the number of percentiles and the number of
        profiles.
    """
    func = _ProfilePickleable(profiles, function)
    return game_function(game, func, num_resamples, percentiles, len(profiles),
                         processes)


class _ProfilePickleable(object):
    def __init__(self, profiles, function):
        self.profiles = profiles
        self.function = function

    def __call__(self, game):
        return [self.function(game, prof) for prof in self.profiles]


def mixture_regret(game, profiles, num_resamples, percentiles, processes=None):
    """Compute percentile bounds on mixture regret

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    profiles : [Mixture]
        The profiles to compute mixture regret bounds for.
    num_resamples : int
        The number of bootstrap samples. Higher will take longer but also give
        better accuracy.
    percentiles : int or [int]
        The percentiles to compute on the resulting data in [0, 100]. Standard
        percentiles are 95, or [2.5, 97.5]. By default, return all samples.
    processes : int (optional)
        The number of processes to use for computation. By default this is the
        number of cores.

    Returns
    -------
    regret_percentiles : ndarray
        An ndarray of the percentiles for bootstrap regret for each profile.
    """
    return profile_function(game, regret.mixture_regret, profiles,
                            num_resamples, percentiles, processes)


def mixture_welfare(game, profiles, num_resamples, percentiles,
                    processes=None):
    """Compute percentile bounds on mixture welfare

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    profiles : [Mixture]
        The profiles to compute mixture welfare bounds for.
    num_resamples : int
        The number of bootstrap samples. Higher will take longer but also give
        better accuracy.
    percentiles : int or [int]
        The percentiles to compute on the resulting data in [0, 100]. Standard
        percentiles are 95, or [2.5, 97.5]. By default, return all samples.
    processes : int (optional)
        The number of processes to use for computation. By default this is the
        number of cores.

    Returns
    -------
    bootstrap_percentiles : ndarray
        An ndarray of the percentiles for bootstrap welfare for each profile.
    """
    return profile_function(game, regret.mixed_social_welfare, profiles,
                            num_resamples, percentiles, processes)


def mean(data, num_resamples, percentiles=None):
    """Compute bootstrap bounds for the mean of a data set

    One particular use is compute bootstrap bounds on social welfare of a
    mixture if all of the samples are iid draws of welfare from the mixture.

    Parameters
    ----------
    data : [float] or ndarray
        The data to get bootstrap estimates around the mean of.
    num_resamples : int
        The number of bootstrap samples. Higher will take longer but also give
        better accuracy.
    percentiles : int or [int]
        The percentiles to compute on the resulting data in [0, 100]. Standard
        percentiles are 95, or [2.5, 97.5].
    """
    data = np.asarray(data, float).ravel()
    samples = rand.multinomial(data.size, np.ones(data.size) / data.size,
                               num_resamples)
    result = samples.dot(data) / data.size
    if percentiles is None:
        result.sort()
        return result
    else:
        return np.percentile(result, percentiles)


def sample_regret(mixture_payoffs, deviation_payoffs, num_resamples,
                  percentiles=None):
    """Compute bootstrap bounds on the mixture regret with samples

    Parameters
    ----------
    mixture_payoffs : {role: [payoff]}
        A sample of payoffs by role. The distribution must come from the
        desired mixture
    deviation payoffs : {role: {strat: [payoff]}} or {role: [[payoff]]}
        The payoff to the deviator when everyone else is played according to
        the mixture. The strategy payoffs can either by a mapping of strategy
        names, or just a list of payoffs.
    num_resamples : int
        The number of bootstrap samples. Higher will take longer but also give
        better accuracy.
    percentiles : int or [int]
        The percentiles to compute on the resulting data in [0, 100]. Standard
        percentiles are 95, or [2.5, 97.5].

    Returns
    -------

    Note: The lengths of every list must be the same
    """
    num_samples = len(next(iter(mixture_payoffs.values())))
    num_roles = len(mixture_payoffs)
    ordered_devs = [deviation_payoffs[r] for r in mixture_payoffs]
    dev_len = np.fromiter(map(len, ordered_devs), int, num_roles)
    dev_red = np.concatenate(([0], np.cumsum(dev_len[:-1])))
    mix = np.asarray(list(mixture_payoffs.values()), float)
    devs = np.asarray(list(itertools.chain.from_iterable(
        (list(x.values() if isinstance(x, abc.Mapping) else x)
         for x in ordered_devs))), float)
    samples = rand.multinomial(num_samples,
                               np.ones(num_samples) / num_samples,
                               num_resamples).T / num_samples
    dev_samples = np.maximum.reduceat(devs.dot(samples), dev_red)
    mix_samples = mix.dot(samples)
    result = np.max(dev_samples - mix_samples, 0)
    if percentiles is None:
        result.sort()
        return result
    else:
        return np.percentile(result, percentiles)
