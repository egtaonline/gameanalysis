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


def gen_profiles(game, prob=1.0, distribution=default_distribution):
    """Generate profiles given game structure

    Parameters
    ----------
    game : RsGame
        Game to generate payoffs for.
    prob : float, optional
        The probability to add a profile from the full game.
    distribution : (shape) -> ndarray, optional
        Distribution function to draw payoffs from.
    """
    # First turn input into number of profiles to compute
    num_profs = game.num_all_profiles
    assert 0 <= prob <= 1
    if num_profs <= np.iinfo(int).max:
        num = rand.binomial(num_profs, prob)
    else:
        num = round(float(num_profs * prob))
    return gen_num_profiles(game, num, distribution)


def gen_num_profiles(game, num, distribution=default_distribution):
    """Generate profiles given game structure

    Parameters
    ----------
    game : RsGame
        Game to generate payoffs for.
    count : int
        The number of profiles to generate.
    distribution : (shape) -> ndarray, optional
        Distribution function to draw payoffs from.
    """
    assert 0 <= num <= game.num_all_profiles
    profiles = sample_profiles(game, num)
    payoffs = np.zeros(profiles.shape)
    mask = profiles > 0
    payoffs[mask] = distribution(mask.sum())
    return paygame.game_replace(game, profiles, payoffs)


def game(players, strats, prob=1.0, distribution=default_distribution):
    """Generate a random role symmetric game with sparse profiles

    Parameters
    ----------
    players : int or [int]
        The number of players per role.
    strats : int or [int]
        The number of strategies per role.
    prob : float, optional
        The probability of any profile being included in the game.
    distribution : (shape) -> ndarray, optional
        Distribution function to draw payoffs from.
    """
    return gen_profiles(rsgame.emptygame(players, strats), prob, distribution)


def sparse_game(players, strats, num, distribution=default_distribution):
    """Generate a random role symmetric game with sparse profiles

    Parameters
    ----------
    players : int or [int]
        The number of players per role.
    strats : int or [int]
        The number of strategies per role.
    num : int
        The number of profiles to draw payoffs for.
    distribution : (shape) -> ndarray, optional
        Distribution function to draw payoffs from.
    """
    return gen_num_profiles(
        rsgame.emptygame(players, strats), num, distribution)


def width_gaussian(widths, num_samples):
    """Gaussian width distribution

    Samples come from iid Gaussian distributions.
    """
    return rand.normal(0, widths[:, None], (widths.size, num_samples))


def width_bimodal(widths, num_samples):
    """Bimodal Gaussian width distribution

    Samples come from a uniform mixture between two symmetric Gaussians.
    """
    squared_widths = widths ** 2
    variances = rand.uniform(0, squared_widths)
    sdevs = np.sqrt(variances)[:, None]
    spreads = np.sqrt(squared_widths - variances)[:, None]
    draws = rand.normal(spreads, sdevs, (widths.size, num_samples))
    draws *= rand.randint(0, 2, draws.shape) * 2 - 1
    return draws


def width_uniform(widths, num_samples):
    """Uniform width distribution

    Samples come from iid uniform distributions.
    """
    halfwidths = np.sqrt(3) * widths[:, None]
    return rand.uniform(-halfwidths, halfwidths, (widths.size, num_samples))


def width_gumbel(widths, num_samples):
    """Gumbel width distribution

    Samples come from iid Gumbel distributions. Distributions are randomly
    inverted since Gumbels are skewed.
    """
    scales = widths[:, None] * np.sqrt(6) / np.pi
    draws = np.random.gumbel(
        -scales * np.euler_gamma, scales, (widths.size, num_samples))
    draws *= rand.randint(0, 2, (widths.size, 1)) * 2 - 1
    return draws


def gen_noise(game, prob=0.5, min_samples=1, min_width=0, max_width=1,
              noise_distribution=width_gaussian):
    """Generate noise for profiles of a game

    This generates samples for payoff data by first generating some measure of
    distribution spread for each payoff in the game. Then, for each sample,
    noise is drawn from this distribution. As a result, some payoffs will have
    significantly more noise than other payoffs, helping to mimic real games.

    Parameters
    ----------
    game : Game
        The game to generate samples from. These samples are additive zero-mean
        noise to the payoff values.
    prob : float, optional
        The probability of continuing to add another sample to a profile. If
        this is 0, min_samples will be generated for each profile. As this
        approaches one, more samples will be generated for each profile,
        sampled from the geometric distribution of 1 - prob.
    min_samples : int, optional
        The minimum number of samples to generate for each profile. By default
        this will generate at least one sample for every profile with data.
        Setting this to zero means that a profile will only have data with
        probability `prob`.
    min_width : float, optional
        The minimum standard deviation of each payoffs samples.
    max_width : float, optional
        The maximum standard deviation of each payoffs samples.
    noise_distribution : width distribution, optional
        The noise generating function to use. This function must be a valid
        width noise distribution. A width distribution takes an array of widths
        and a number of samples, and draws that many samples for each width
        such that the standard deviation of the samples is equal to the width
        and the mean is zero.  Several default versions are specified in
        gamegen, and they're all prefixed with `width_`. By default, this uses
        `width_gaussian`.
    """
    if game.is_empty():
        return paygame.samplegame_copy(game)
    perm = rand.permutation(game.num_profiles)
    profiles = game.profiles()[perm]
    payoffs = game.payoffs()[perm]
    samples = utils.geometric_histogram(profiles.shape[0], 1 - prob)
    mask = samples > 0
    observations = np.arange(samples.size)[mask] + min_samples
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
        widths = rand.uniform(min_width, max_width, supp.sum())
        pview[supp] = pay[supp, None] + noise_distribution(widths, num)

        new_profiles.append(prof)
        sample_payoffs.append(spay)

    if new_profiles:
        new_profiles = np.concatenate(new_profiles)
    else:  # No data
        new_profiles = np.empty((0, game.num_strats), dtype=int)
    return paygame.samplegame_replace(game, new_profiles, sample_payoffs)


def samplegame(players, strats, prob=0.5, min_samples=1, min_width=0,
               max_width=1, payoff_distribution=default_distribution,
               noise_distribution=width_gaussian):
    """Generate a random role symmetric sample game

    Parameters
    ----------
    players : int or [int]
        The number of players per role.
    strats : int or [int]
        The number of strategies per role.
    prob : float, optional
        The probability of adding another sample above min_samples. These draws
        are repeated, so 0.5 will add one extra sample in expectation.
    min_samples : int, optional
        The minimum number of samples to generate for each profile. If 0, the
        game will potentially be sparse.
    min_width : float, optional
        The minimum standard deviation for each payoffs samples.
    max_width : float, optional
        The maximum standard deviation for each payoffs samples.
    payoff_distribution : (shape) -> ndarray, optional
        The distribution to sample mean payoffs from.
    noise_distribution : width distribution, optional
        The distribution used to add noise to each payoff. See `gen_noise` for
        a description of width distributions.
    """
    if min_samples == 0:
        base = game(players, strats, prob, payoff_distribution)
        min_samples = 1
    else:
        base = game(players, strats, distribution=payoff_distribution)
    return gen_noise(base, prob, min_samples, min_width, max_width,
                     noise_distribution)


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
    _, s, v = np.linalg.svd(var)
    payoffs = rand.normal(size=shape)
    payoffs = np.einsum('...i,...i,...ij->...j', payoffs, np.sqrt(s), v)
    payoffs += mean_dist(shape)
    return matgame.matgame(payoffs)


def two_player_zero_sum_game(num_role_strats,
                             distribution=default_distribution):
    """Generate a two-player, zero-sum game"""
    # Generate player 1 payoffs
    num_role_strats = np.broadcast_to(num_role_strats, 2)
    p1_payoffs = distribution(num_role_strats)[..., None]
    return matgame.matgame(np.concatenate([p1_payoffs, -p1_payoffs], -1))


def sym_2p2s_game(a, b, c, d, distribution=default_distribution):
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

    distribution must accept a size parameter a la numpy distributions.
    """
    assert {a, b, c, d} == set(range(4))
    # Generate payoffs
    payoffs = distribution(4)
    payoffs.sort()
    profs = [[2, 0], [1, 1], [0, 2]]
    pays = [[payoffs[a], 0], [payoffs[b], payoffs[c]], [0, payoffs[d]]]
    return paygame.game(2, 2, profs, pays)


def prisoners_dilemma(distribution=default_distribution):
    """Return a random prisoners dilemma game"""
    return sym_2p2s_game(2, 0, 3, 1, distribution)


def chicken(distribution=default_distribution):
    """Return a random prisoners dilemma game"""
    return sym_2p2s_game(0, 3, 1, 2, distribution)


def sym_2p2s_known_eq(eq_prob):
    """Generate a symmetric 2-player 2-strategy game

    This game has a single mixed equilibrium where strategy one is played with
    probability eq_prob.
    """
    profiles = [[2, 0], [1, 1], [0, 2]]
    payoffs = [[0, 0], [eq_prob, 1 - eq_prob], [0, 0]]
    return paygame.game(2, 2, profiles, payoffs)


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


def keep_profiles(game, keep_prob=0.5):
    """Keep random profiles from an existing game

    Parameters
    ----------
    game : RsGame
        The game to take profiles from.
    keep_prob : float, optional
        The probability of keeping a profile from the full game.
    """
    # First turn input into number of profiles to compute
    num_profs = game.num_profiles
    assert 0 <= keep_prob <= 1
    if num_profs <= np.iinfo(int).max:
        num = rand.binomial(num_profs, keep_prob)
    else:
        num = round(float(num_profs * keep_prob))
    return keep_num_profiles(game, num)


def keep_num_profiles(game, num):
    """Keep random profiles from an existing game

    Parameters
    ----------
    game : RsGame
        Game to keep profiles from.
    num : int
        The number of profiles to keep from the game.
    """
    assert 0 <= num <= game.num_profiles
    if num == 0:
        profiles = np.empty((0, game.num_strats), int)
        payoffs = np.empty((0, game.num_strats))
    elif game.is_complete():
        profiles = sample_profiles(game, num)
        payoffs = game.get_payoffs(profiles)
    else:
        inds = rand.choice(game.num_profiles, num, replace=False)
        profiles = game.profiles()[inds]
        payoffs = game.payoffs()[inds]
    return paygame.game_replace(game, profiles, payoffs)


def sample_profiles(game, num):
    """Generate unique profiles from a game

    Parameters
    ----------
    game : RsGame
        Game to generate random profiles from.
    num : int
        Number of profiles to sample from the game.
    """
    if num == game.num_all_profiles:
        return game.all_profiles()
    elif num == 0:
        return np.empty((0, game.num_strats), int)
    elif game.num_all_profiles <= np.iinfo(int).max:
        inds = rand.choice(game.num_all_profiles, num, replace=False)
        return game.profile_from_id(inds)
    else:
        # Number of times we have to re-query
        ratio = (sps.digamma(float(game.num_all_profiles)) -
                 sps.digamma(float(game.num_all_profiles - num)))
        # Max is for underflow
        num_per = max(round(float(ratio * game.num_all_profiles)), num)
        profiles = set()
        while len(profiles) < num:
            profiles.update(
                utils.hash_array(p) for p in game.random_profiles(num_per))
        profiles = np.stack([h.array for h in profiles])
        inds = rand.choice(profiles.shape[0], num, replace=False)
        return profiles[inds]
