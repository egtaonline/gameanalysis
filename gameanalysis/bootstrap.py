"Module for using bootstrap in analysis"
import multiprocessing

import numpy as np

from gameanalysis import regret


def game_function(game, function, num_resamples, num_returned, *,
                  percentiles=None, processes=None):
    """Bootstrap the value of a function over a sample game

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    function : f(Game) -> float or f(Game) -> [float]
        The function of the game to compute. It must be pickleable unless
        processes is 1, and it must return either a float or an iterable of
        floats. If an iterable of floats, this bootstrap all indices of the
        return value independently.
    num_resamples : int
        The number of bootstrap samples. Higher will take longer but also give
        better accuracy.
    num_returned : int
        The number of float values your function returns.
    percentiles : int or [int]
        The percentiles to compute on the resulting data in [0, 100]. Standard
        percentiles are 95, or [2.5, 97.5]. By default, return all samples.
    processes : int (optional)
        The number of processes to use for computation. By default this is the
        number of cores.

    Returns
    -------
    bootstrap_percentiles : ndarray
        An ndarray of the percentiles from bootstrapping. The shape will depend
        on the number of percentiles and the number of values returned from
        your function.
    """
    results = np.empty((num_resamples, num_returned))
    func = _BootstrapPickleable(game, function)
    chunksize = num_resamples if processes == 1 else 4
    with multiprocessing.Pool(processes) as pool:
        for i, res in enumerate(pool.imap_unordered(
                func, range(num_resamples), chunksize=chunksize)):
            results[i] = res

    if percentiles is None:
        results.sort(0)
        return results.T
    else:
        return np.percentile(results, percentiles, 0).T


class _BootstrapPickleable(object):
    """A pickleable game function combo"""

    def __init__(self, game, function):
        self.game = game
        self.function = function

    def __call__(self, _):
        return self.function(self.game.resample())


def profile_function(game, function, profiles, num_resamples, *,
                     percentiles=None, processes=None):
    """Compute a function over profiles

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    function : Game, profile -> float
        The function of the game profile pair to compute. It must be
        pickleable, and it must return a float (e.g. regret.mixture_regret).
    profiles : ndarray
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
    if profiles.ndim == 1:
        profiles = profiles[None]
    func = _ProfilePickleable(profiles, function)
    return game_function(game, func, num_resamples, profiles.shape[0],
                         percentiles=percentiles, processes=processes)


class _ProfilePickleable(object):

    def __init__(self, profiles, function):
        self.profiles = profiles
        self.function = function

    def __call__(self, game):
        return [self.function(game, prof) for prof in self.profiles]


def mixture_regret(game, mixtures, num_resamples, *, percentiles=None,
                   processes=None):
    """Compute percentile bounds on mixture regret

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    mixtures : ndararay
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
    return profile_function(game, regret.mixture_regret, mixtures,
                            num_resamples, percentiles=percentiles,
                            processes=processes)


def mixture_welfare(game, mixtures, num_resamples, *, percentiles=None,
                    processes=None):
    """Compute percentile bounds on mixture welfare

    Parameters
    ----------
    game : SampleGame
        The sample game to bootstrap the function value over.
    mixtures : ndarray
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
    return profile_function(game, regret.mixed_social_welfare, mixtures,
                            num_resamples, percentiles=percentiles,
                            processes=processes)
