import itertools
from collections import abc

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import utils


def default_distribution(shape=None):
    return rand.uniform(-1, 1, shape)


def role_symmetric_game(num_role_players, num_role_strats,
                        distribution=default_distribution):
    """Generate a random role symmetric game

    Parameters
    ----------
    num_roles : int > 0
        The number of roles in the game.
    num_role_players : int or [int], len == num_roles
        The number of players, same for each role if a scalar, or a list, one
        for each role.
    num_role_strats : int or [int], len == num_roles
        The number of strategies, same for each role if a scalar, or a list,
        one for each role.
    distribution : (shape) -> ndarray (shape)
        Payoff distribution.
    """
    game = rsgame.emptygame(num_role_players, num_role_strats)
    profiles = game.all_profiles()
    mask = profiles > 0
    payoffs = np.zeros(profiles.shape)
    payoffs[mask] = distribution(mask.sum())
    return paygame.game_replace(game, profiles, payoffs)


def independent_game(num_role_strats, distribution=default_distribution):
    """Generate a random independent (asymmetric) game

    All payoffs are generated independently from distribution.

    Parameters
    ----------
    num_role_strats : int or [int], len == num_role_players
        The number of strategies for each player. If an int, then it's a one
        player game.
    distribution : (shape) -> ndarray (shape)
        The distribution to sample payoffs from. Must take a single shape
        argument and return an ndarray of iid values with that shape.
    """
    return matgame.matgame(distribution(
        tuple(num_role_strats) + (len(num_role_strats),)))


def covariant_game(num_role_strats, mean_dist=lambda shape: np.zeros(shape),
                   var_dist=lambda shape: np.ones(shape),
                   covar_dist=default_distribution):
    """Generate a covariant game

    Covariant games are asymmetric games where payoff values for each profile
    drawn according to multivariate normal.

    The multivariate normal for each profile has a constant mean drawn from
    `mean_dist`, constant variance drawn from`var_dist`, and constant
    covariance drawn from `covar_dist`.

    Parameters
    ----------
    mean_dist : (shape) -> ndarray (shape)
        Distribution from which mean payoff for each profile is drawn.
        (default: lambda: 0)
    var_dist : (shape) -> ndarray (shape)
        Distribution from which payoff variance for each profile is drawn.
        (default: lambda: 1)
    covar_dist : (shape) -> ndarray (shape)
        Distribution from which the value of the off-diagonal covariance matrix
        entries for each profile is drawn. (default: uniform [-1, 1])
    """
    # Create sampling distributions and sample from them
    num_role_strats = list(num_role_strats)
    num_players = len(num_role_strats)
    shape = num_role_strats + [num_players]
    var = covar_dist(shape + [num_players])
    diag = var.diagonal(0, num_players, num_players + 1)
    diag.setflags(write=True)  # Hack
    np.copyto(diag, var_dist(shape))

    # The next couple of lines do multivariate Gaussian sampling for all
    # payoffs simultaneously
    u, s, v = np.linalg.svd(var)
    payoffs = rand.normal(size=shape)
    payoffs = (payoffs[..., None] * (np.sqrt(s)[..., None] * v)).sum(-2)
    payoffs += mean_dist(shape)
    return matgame.matgame(payoffs)


def two_player_zero_sum_game(num_role_strats,
                             distribution=default_distribution):
    """Generate a two-player, zero-sum game"""
    # Generate player 1 payoffs
    num_role_strats = np.broadcast_to(num_role_strats, 2)
    p1_payoffs = distribution(num_role_strats)[..., None]
    return matgame.matgame(np.concatenate([p1_payoffs, -p1_payoffs], -1))


def sym_2p2s_game(a=0, b=1, c=2, d=3, distribution=default_distribution):
    """Create a symmetric 2-player 2-strategy game of the specified form.

    Four payoff values get drawn from U(min_val, max_val), and then are
    assigned to profiles in order from smallest to largest according to the
    order parameters as follows:


    +---+-----+-----+
    |   | s0  | s1  |
    +---+-----+-----+
    |s0 | a,a | b,c |
    +---+-----+-----+
    |s1 | c,b | d,d |
    +---+-----+-----+


    So a=2,b=0,c=3,d=1 gives a prisoners' dilemma; a=0,b=3,c=1,d=2 gives a game
    of chicken.

    distribution must accept a size parameter a la numpy distributions.
    """
    # Generate payoffs
    payoffs = distribution(4)
    payoffs.sort()
    counts = [[2, 0], [1, 1], [0, 2]]
    values = [[payoffs[a], 0], [payoffs[b], payoffs[c]], [0, payoffs[d]]]
    return paygame.game(2, 2, counts, values)


def prisoners_dilemma(distribution=default_distribution):
    """Return a random prisoners dilemma game"""
    return sym_2p2s_game(2, 0, 3, 1, distribution)


def sym_2p2s_known_eq(eq_prob):
    """Generate a symmetric 2-player 2-strategy game

    This game has a single mixed equilibrium where strategy one is played with
    probability eq_prob.
    """
    profiles = [[2, 0], [1, 1], [0, 2]]
    payoffs = [[0, 0], [eq_prob, 1 - eq_prob], [0, 0]]
    return paygame.game(2, 2, profiles, payoffs)


# TODO There are nash finding methods that rely on approximating games as poly
# matrix games. Potentially we can implement a polymatrix / rs polymatrix first
# and then the nash will follow nicely from that
def polymatrix_game(num_players, num_strats, matrix_game=independent_game,
                    players_per_matrix=2):
    """Creates a polymatrix game using the specified k-player matrix game function.

    Each player's payoff in each profile is a sum over independent games played
    against each set of opponents. Each k-tuple of players plays an instance of
    the specified random k-player matrix game.

    Parameters
    ----------
    num_players : int
        The number of players.
    num_strats : int
        The number of strategies per player.
    matrix_game : (players_per_matrix, num_strats) -> Game, optional
        A function to generate games between sub groups of players.
    players_per_matrix : int, optional
        The number of players that interact simultaneously.

    Notes
    -----
    The actual roles and strategies of matrix game are ignored.
    """
    payoffs = np.zeros([num_strats] * num_players + [num_players])
    for players in itertools.combinations(range(num_players),
                                          players_per_matrix):
        sub_payoffs = matgame.matgame_copy(matrix_game(
            [num_strats] * players_per_matrix)).payoff_matrix()
        new_shape = np.array([1] * num_players + [players_per_matrix])
        new_shape[list(players)] = num_strats
        payoffs[..., list(players)] += sub_payoffs.reshape(new_shape)

    return matgame.matgame(payoffs)


def rock_paper_scissors(win=1, loss=-1):
    """Return an instance of rock paper scissors"""
    if isinstance(win, abc.Iterable):
        win = list(win)
    else:
        win = [win] * 3
    if isinstance(loss, abc.Iterable):
        loss = list(loss)
    else:
        loss = [loss] * 3
    assert (all(l < 0 for l in loss) and all(0 < w for w in win) and
            len(loss) == 3 and len(win) == 3)
    profiles = [[2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]]
    payoffs = [[0., 0., 0.],
               [loss[0], win[0], 0.],
               [win[1], 0., loss[1]],
               [0., 0., 0.],
               [0., loss[2], win[2]],
               [0., 0., 0.]]
    return paygame.game_names(['all'], 2, [['paper', 'rock', 'scissors']],
                              profiles, payoffs)


def travellers_dilemma(players=2, max_value=100):
    """Return an instance of travellers dilemma

    Strategies range from 2 to max_value, thus there will be max_value - 1
    strategies."""
    assert players > 1, "players must be more than one"
    assert max_value > 2, "max value must be more than 2"
    game = rsgame.emptygame(players, max_value - 1)
    profiles = game.all_profiles()
    payoffs = np.zeros(profiles.shape)
    mins = np.argmax(profiles, -1)
    mask = profiles > 0
    payoffs[mask] = mins.repeat(mask.sum(-1))
    rows = np.arange(profiles.shape[0])
    ties = profiles[rows, mins] > 1
    lowest_pays = mins + 4
    lowest_pays[ties] -= 2
    payoffs[rows, mins] = lowest_pays
    return paygame.game_replace(game, profiles, payoffs)


# TODO We could implement this with drop_profiles if we made a "RandomGame"
# class where all payoffs were simply drawn from distribution
def add_profiles(game, prob_or_count=1.0, distribution=default_distribution):
    """Add profiles to a base game

    Parameters
    ----------
    prob_or_count : float or int, optional
        If a float, the probability to add a profile from the full game. If an
        int, the number of profiles to add.
    distribution : (shape) -> ndarray, optional
        Distribution function to draw payoffs from.
    """
    # First turn input into number of profiles to compute
    num_profs = game.num_all_profiles
    if isinstance(prob_or_count, float):
        assert 0 <= prob_or_count <= 1
        if num_profs <= np.iinfo(int).max:
            num = rand.binomial(num_profs, prob_or_count)
        else:
            num = round(float(num_profs * prob_or_count))
    else:
        assert 0 <= prob_or_count <= num_profs
        num = prob_or_count

    # Generate profiles based number and size of game

    # Ratio of the expected number of profiles we'd have to draw at random to
    # produce num unique relative to the number of total profiles
    ratio = sps.digamma(float(num_profs)) - sps.digamma(float(num_profs - num))
    if num == num_profs:
        profiles = game.all_profiles()
    elif num == 0:
        profiles = np.empty((0, game.num_strats), int)
    elif ratio >= 1:
        inds = rand.choice(num_profs, num, replace=False)
        profiles = game.all_profiles()[inds]
    else:
        # TODO use a set and hashed arrays instead?
        profiles = np.empty((0, game.num_strats), int)
        num_per = max(round(float(ratio * num_profs)), num)  # Max => underflow
        mix = game.uniform_mixture()
        while profiles.shape[0] < num:
            profiles = np.concatenate([profiles,
                                       game.random_profiles(num_per, mix)])
            profiles = utils.unique_axis(profiles)
        inds = rand.choice(profiles.shape[0], num, replace=False)
        profiles = profiles[inds]

    # Fill out game with profiles
    payoffs = np.zeros(profiles.shape)
    mask = profiles > 0
    payoffs[mask] = distribution(mask.sum())
    return paygame.game_replace(game, profiles, payoffs)


def drop_profiles(game, keep_prob_or_count=1.0):
    """Drop profiles from an existing game

    Parameters
    ----------
    keep_prob_or_count : float or int, optional
        If a float, the probability to keep a profile from the full game. If an
        int, the number of profiles to keep.
    """
    # First turn input into number of profiles to compute
    num_profs = game.num_profiles
    if isinstance(keep_prob_or_count, float):
        assert 0 <= keep_prob_or_count <= 1
        if num_profs <= np.iinfo(int).max:
            num = rand.binomial(num_profs, keep_prob_or_count)
        else:
            num = round(float(num_profs * keep_prob_or_count))
    else:
        assert 0 <= keep_prob_or_count <= num_profs
        num = keep_prob_or_count

    # Generate profiles based number and size of game

    # Ratio of the expected number of profiles we'd have to draw at random to
    # produce num unique relative to the number of total profiles
    ratio = sps.digamma(float(num_profs)) - sps.digamma(float(num_profs - num))
    if num == num_profs:
        profiles = game.profiles()
        payoffs = game.payoffs()
    elif num == 0:
        profiles = np.empty((0, game.num_strats), int)
        payoffs = np.empty((0, game.num_strats))
    elif ratio >= 1:
        inds = rand.choice(num_profs, num, replace=False)
        profiles = game.profiles()[inds]
        payoffs = game.payoffs()[inds]
    else:
        # TODO use a set and hashed arrays instead?
        profiles = np.empty((0, game.num_strats), int)
        num_per = max(round(float(ratio * num_profs)), num)  # Max => underflow
        mix = game.uniform_mixture()
        while profiles.shape[0] < num:
            profiles = np.concatenate([profiles,
                                       game.random_profiles(num_per, mix)])
            profiles = utils.unique_axis(profiles)
        inds = rand.choice(profiles.shape[0], num, replace=False)
        profiles = profiles[inds]
        payoffs = game.get_payoffs(profiles)

    return paygame.game_replace(game, profiles, payoffs)


# TODO This would probably be more "cohesive" if instead of num_samples, this
# was sample prob, and the number of samples was drawn from an exponential
# distribution with prob. Doing this would require figuring out how to
# efficiently sample from the distribution, of given that I pull N random
# exponential variables, what's the histogram / how do I sample from the
# histogram instead of doing N random draws.
def add_noise(game, min_samples, max_samples=None, noise=default_distribution):
    """Generate sample game by adding noise to game payoffs

    Arguments
    ---------
    game : Game
        A Game or SampleGame (only current payoffs are used)
    min_samples : int
        The minimum number of observations to create per profile
    max_samples : int
        The maximum number of observations to create per profile. If None, it's
        the same as min_samples.
    noise : shape -> ndarray
        A noise generating function. The function should take a single shape
        parameter, and return a number of samples equal to shape. In order to
        preserve mixed equilibria, noise should also be zero mean (aka
        unbiased)
    """
    if game.is_empty():
        return paygame.samplegame_copy(game)

    perm = rand.permutation(game.num_profiles)
    profiles = game.profiles()[perm]
    payoffs = game.payoffs()[perm]
    if max_samples is None:
        max_samples = min_samples
    assert 0 <= min_samples <= max_samples, "invalid sample numbers"
    max_samples += 1
    num_values = max_samples - min_samples
    samples = rand.multinomial(profiles.shape[0],
                               np.ones(num_values) / num_values)
    mask = samples > 0
    observations = np.arange(min_samples, max_samples)[mask]
    splits = samples[mask][:-1].cumsum()

    sample_payoffs = []
    new_profiles = []
    for num, prof, pay in zip(observations, np.split(profiles, splits),
                              np.split(payoffs, splits)):
        if num == 0:
            continue
        supp = prof > 0
        spay = np.zeros((pay.shape[0], num, game.num_strats))
        pview = np.rollaxis(spay, 1, 3)
        pview[supp] = pay[supp, None] + noise((supp.sum(), num))

        new_profiles.append(prof)
        sample_payoffs.append(spay)

    if new_profiles:
        new_profiles = np.concatenate(new_profiles)
    else:  # No data
        new_profiles = np.empty((0, game.num_strats), dtype=int)
    return paygame.samplegame_replace(game, new_profiles, sample_payoffs)


def width_gaussian(max_width, num_profiles, num_samples):
    """Gaussian width distribution

    This returns standard deviations from U[0, max_width].
    """
    widths = rand.uniform(0, max_width, num_profiles)
    return rand.normal(0, widths, (num_samples, num_profiles)).T


def width_gaussian_old(scale=1):
    """Old gaussian width distribution

    This returns a valid distribution, taking a scale parameter to correct for
    the scale invariance of guassian variance.
    """
    def width_gaussian(max_width, num_profiles, num_samples):
        widths = rand.uniform(0, max_width, num_profiles)
        return rand.normal(0, np.sqrt(widths) * scale,
                           (num_samples, num_profiles)).T
    return width_gaussian


def width_bimodal(max_width, num_profiles, num_samples):
    """Bimodal width distribution

    This returns standard deviations from U[0, max_width] and half spreads from
    N[0, sqrt(max_width)].
    """
    sdevs = rand.uniform(0, max_width, num_profiles)
    spreads = rand.normal(0, max_width, num_profiles)
    draws = rand.normal(spreads, sdevs, (num_samples, num_profiles)).T
    draws *= (rand.random(draws.shape) < .5) * 2 - 1
    return draws


def width_bimodal_old(scale=1):
    """Old bimodal width distribution

    This returns a valid distribution, taking a scale parameter to correct for
    the scale invariance of guassian variance.
    """
    def width_bimodal(max_width, num_profiles, num_samples):
        variances = np.sqrt(rand.uniform(0, max_width, num_profiles)) * scale
        spreads = rand.normal(0, np.sqrt(max_width) * scale, num_profiles)
        draws = rand.normal(spreads, variances, (num_samples, num_profiles)).T
        draws *= (rand.random(draws.shape) < .5) * 2 - 1
        return draws
    return width_bimodal


def width_uniform(max_width, num_profiles, num_samples):
    """Uniform width distribution

    Generates halfwidths in U[0, max_width]
    """
    halfwidths = rand.uniform(0, max_width, num_profiles)
    return rand.uniform(-halfwidths, halfwidths, (num_samples, num_profiles)).T


def width_gumbel(max_width, num_profiles, num_samples):
    """Gumbel width distribution

    Generates scales in U[0, max_width]
    """
    scales = rand.uniform(0, max_width, num_profiles)
    return rand.gumbel(0, scales, (num_samples, num_profiles)).T


def add_noise_width(game, num_samples, max_width, noise=width_gaussian):
    """Create sample game where each profile has different noise level

    Parameters
    ----------
    game : Game
        The game to generate samples from. These samples are additive noise to
        standard payoff values.
    num_samples : int
        The number of samples to generate for each profile.
    max_width : float
        A parameter describing how much noise to generate. Larger max_width
        generates more noise.
    noise : (float, int, int) -> ndarray (optional)
        The noise generating function to use. The function must take three
        parameters: the max_width, the number of profiles, and the number of
        samples, and return an ndarray of the additive noise for each profile
        (shape: (num_profiles, num_samples)). The max_width should be used to
        generate sufficient statistics for each profile, and then each sample
        per profile should come from a distribution derived from those. For
        this to be accurate, this distribution should have expectation 0.
        Several default versions are specified in gamegen, and they're all
        prefixed with `width_`. By default, this uses `width_gaussian`.
    """
    spayoffs = game.payoffs()[:, None].repeat(num_samples, 1)
    supp = game.profiles() > 0
    view = np.rollaxis(spayoffs, 1, 3)
    view[supp] += noise(max_width, supp.sum(), num_samples)
    return paygame.samplegame_replace(game, game.profiles(), [spayoffs])
