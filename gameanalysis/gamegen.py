import itertools

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import gameio
from gameanalysis import rsgame
from gameanalysis import utils


def default_distribution(shape=None):
    return rand.uniform(-1, 1, shape)


def role_symmetric_game(num_players, num_strategies,
                        distribution=default_distribution):
    """Generate a random role symmetric game

    Parameters
    ----------
    num_roles : int > 0
        The number of roles in the game.
    num_players : int or [int], len == num_roles
        The number of players, same for each role if a scalar, or a list, one
        for each role.
    num_strategies : int or [int], len == num_roles
        The number of strategies, same for each role if a scalar, or a list,
        one for each role.
    distribution : (shape) -> ndarray (shape)
        Payoff distribution.
    """
    game = rsgame.basegame(num_players, num_strategies)
    profiles = game.all_profiles()
    mask = profiles > 0
    payoffs = np.zeros(profiles.shape)
    payoffs[mask] = distribution(mask.sum())
    return rsgame.game_copy(game, profiles, payoffs)


def independent_game(num_strategies, distribution=default_distribution):
    """Generate a random independent (asymmetric) game

    All payoffs are generated independently from distribution.

    Parameters
    ----------
    num_players : int > 0
        The number of players.
    num_strategies : int or [int], len == num_players
        The number of strategies for each player. If an int, then every player
        has the same number of strategies.
    distribution : (shape) -> ndarray (shape)
        The distribution to sample payoffs from. Must take a single shape
        argument and return an ndarray of iid values with that shape.
    """
    return role_symmetric_game(1, num_strategies, distribution)


def covariant_game(num_strategies, mean_dist=lambda shape: np.zeros(shape),
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
    num_strategies = list(num_strategies)
    num_players = len(num_strategies)
    shape = num_strategies + [num_players]
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
    return rsgame.game_matrix(payoffs)


def two_player_zero_sum_game(num_strategies,
                             distribution=default_distribution):
    """Generate a two-player, zero-sum game"""
    # Generate player 1 payoffs
    num_strategies = np.broadcast_to(num_strategies, 2)
    p1_payoffs = distribution(num_strategies)[..., None]
    return rsgame.game_matrix(np.concatenate([p1_payoffs, -p1_payoffs], -1))


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
    return rsgame.game(2, 2, counts, values)


def prisoners_dilemma(distribution=default_distribution):
    """Return a random prisoners dilemma game"""
    return normalize(sym_2p2s_game(2, 0, 3, 1, distribution))


def sym_2p2s_known_eq(eq_prob):
    """Generate a symmetric 2-player 2-strategy game

    This game has a single mixed equilibrium where strategy one is played with
    probability eq_prob.
    """
    profiles = [[2, 0], [1, 1], [0, 2]]
    payoffs = [[0, 0], [eq_prob, 1 - eq_prob], [0, 0]]
    return rsgame.game(2, 2, profiles, payoffs)


def polymatrix_game(num_players, num_strategies, matrix_game=independent_game,
                    players_per_matrix=2):
    """Creates a polymatrix game using the specified k-player matrix game function.

    Each player's payoff in each profile is a sum over independent games played
    against each set of opponents. Each k-tuple of players plays an instance of
    the specified random k-player matrix game.

    Parameters
    ----------
    num_players : int
        The number of players.
    num_strategies : int
        The number of strategies per player.
    matrix_game : (players_per_matrix, num_strategies) -> Game, optional
        A function to generate games between sub groups of players.
    players_per_matrix : int, optional
        The number of players that interact simultaneously.

    Notes
    -----
    The actual roles and strategies of matrix game are ignored.
    """
    payoffs = np.zeros([num_strategies] * num_players + [num_players])
    for players in itertools.combinations(range(num_players),
                                          players_per_matrix):
        sub_payoffs = _compact_payoffs(matrix_game([num_strategies] *
                                                   players_per_matrix))
        new_shape = np.array([1] * num_players + [players_per_matrix])
        new_shape[list(players)] = num_strategies
        payoffs[..., list(players)] += sub_payoffs.reshape(new_shape)

    return rsgame.game_matrix(payoffs)


def _compact_payoffs(game):
    """Given a game returns a compact representation of the payoffs

    In this case compact means that they're in one ndarray. This representation
    is inefficient for almost everything but an independent game with full
    data.

    Parameters
    ----------
    game : Game
        The game to generate a compact payoff matrix for

    Returns
    -------
    payoffs : ndarray; shape (s1, s2, ..., sn, n)
        payoffs[s1, s2, ..., sn, j] is the payoff to player j when player 1
        plays s1, player 2 plays s2, etc. n is the total number of players.
    """
    payoffs = np.empty(list(game.num_strategies) + [game.num_roles])
    for profile, payoff in zip(game.profiles, game.payoffs):
        # This generator expression takes a role symmetric profile with payoffs
        # and generates tuples of strategy indexes and payoffs for every player
        # when that player plays the given strategy.

        # The first line takes results in the form:
        # (((r1i1, r1p1), (r1i2, r1p2)), ((r1i1, r2p1),)) that is grouped by
        # role, then by player in the role, then grouped strategy index and
        # payoff, and turns it into a single tuple of indices and payoffs.
        perms = (zip(*itertools.chain.from_iterable(sp))
                 # This product is over roles
                 for sp in itertools.product(*[
                     # This computes all of the ordered permutations of
                     # strategies in a given role, e.g. if two players play s1
                     # and one plays s2, this iterates over all possible ways
                     # that could be expressed in an asymmetric game.
                     utils.ordered_permutations(itertools.chain.from_iterable(
                         # This iterates over the strategy counts, and
                         # duplicates strategy indices and payoffs based on the
                         # strategy counts.
                         itertools.repeat((i, v), c) for i, (c, v)
                         in enumerate(zip(p, pay))))
                     for p, pay in zip(game.role_split(profile),
                                       game.role_split(payoff))]))
        for indices, utilities in perms:
            payoffs[indices] = utilities
    return payoffs


def rock_paper_scissors(win=1, loss=-1, return_serial=False):
    """Return an instance of rock paper scissors"""
    assert win > 0 and loss < 0
    profiles = [[2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]]
    payoffs = [[0., 0., 0.],
               [loss, win, 0.],
               [win, 0., loss],
               [0., 0., 0.],
               [0., loss, win],
               [0., 0., 0.]]
    game = rsgame.game(2, 3, profiles, payoffs)
    if not return_serial:
        return game
    else:
        serial = gameio.gameserializer(['all'],
                                       [['rock', 'paper', 'scissors']])
        return game, serial


def travellers_dilemma(players=2, max_value=100):
    """Return an instance of travellers dilemma

    Strategies range from 2 to max_value, thus there will be max_value - 1
    strategies."""
    assert players > 1, "players must be more than one"
    assert max_value > 2, "max value must be more than 2"
    game = rsgame.basegame(players, max_value - 1)
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
    return rsgame.game_copy(game, profiles, payoffs, False)


def normalize(game, new_min=0, new_max=1):
    """Return a normalized game"""
    profiles = game.profiles
    scale = game.role_repeat(game.max_payoffs() - game.min_payoffs())
    offset = game.role_repeat(game.min_payoffs())
    payoffs = (game.payoffs - offset) / scale * (new_max - new_min) + new_min
    payoffs *= profiles > 0
    return rsgame.game_copy(game, profiles, payoffs, False)


def add_profiles(game, prob_or_count=1.0, distribution=default_distribution):
    """Add profiles to a base game

    Parameters
    ----------
    distribution : (shape) -> ndarray, optional
        Distribution function to draw profiles from.
    prob_or_count : float or int, optional
        If a float, the probability to add a profile from the full game. If an
        int, the number of profiles to add.
    independent : bool, optional
        If true then each profile has `prob` probability of being added, else
        `num_all_profiles * prob` profiles will be kept.
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
        profiles = np.empty((0, game.num_role_strats), int)
    elif ratio >= 1:
        inds = rand.choice(num_profs, num, replace=False)
        profiles = game.all_profiles()[inds]
    else:
        profiles = np.empty((0, game.num_role_strats), int)
        num_per = max(round(float(ratio * num_profs)), num)  # Max => underflow
        mix = game.uniform_mixture()
        while profiles.shape[0] < num:
            profiles = np.concatenate([profiles,
                                       game.random_profiles(mix, num_per)])
            profiles = utils.unique_axis(profiles)
        inds = rand.choice(profiles.shape[0], num, replace=False)
        profiles = profiles[inds]

    # Fill out game with profiles
    payoffs = np.zeros(profiles.shape)
    mask = profiles > 0
    payoffs[mask] = distribution(mask.sum())
    return rsgame.game_copy(game, profiles, payoffs, False)


def drop_profiles(game, prob, independent=True):
    """Drop profiles from a game

    If independent then each profile has prob of being removed, if not
    independent, then `num_profiles * prob` profiles will be kept."""
    if independent:
        selection = rand.random(game.num_profiles) < prob
    else:
        inds = rand.choice(np.arange(game.num_profiles),
                           round(game.num_profiles * prob), replace=False)
        selection = np.zeros(game.num_profiles, bool)
        selection[inds] = True

    if isinstance(game, rsgame.SampleGame):
        new_profiles = game.profiles[selection]
        new_sample_payoffs = [
            payoffs[mask] for payoffs, mask
            in zip(game.sample_payoffs,
                   np.split(selection, game.sample_starts[1:]))
            if np.any(mask)]
        return rsgame.samplegame_copy(game, new_profiles, new_sample_payoffs,
                                      False)
    else:
        new_profiles = game.profiles[selection]
        new_payoffs = game.payoffs[selection]
        return rsgame.game_copy(game, new_profiles, new_payoffs, False)


def drop_samples(game, prob):
    """Drop samples from a sample game

    Samples are dropped independently with probability prob."""
    sample_map = {}
    for prof, pays in zip(np.split(game.profiles, game.sample_starts[1:]),
                          game.sample_payoffs):
        num_profiles, _, num_samples = pays.shape
        perm = rand.permutation(num_profiles)
        prof = prof[perm]
        pays = pays[perm]
        new_samples, counts = np.unique(
            rand.binomial(num_samples, prob, num_profiles), return_counts=True)
        splits = counts[:-1].cumsum()
        for num, prof_samp, pay_samp in zip(
                new_samples, np.split(prof, splits), np.split(pays, splits)):
            if num == 0:
                continue
            prof, pays = sample_map.setdefault(num, ([], []))
            prof.append(prof_samp)
            pays.append(pay_samp[..., :num])

    if sample_map:
        profiles = np.concatenate(list(itertools.chain.from_iterable(
            x[0] for x in sample_map.values())), 0)
        sample_payoffs = tuple(np.concatenate(x[1]) for x
                               in sample_map.values())
    else:  # No data
        profiles = np.empty((0, game.num_role_strats), dtype=int)
        sample_payoffs = []

    return rsgame.samplegame_copy(game, profiles, sample_payoffs, False)


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
    if game.num_profiles == 0:
        return rsgame.samplegame_copy(game)

    perm = rand.permutation(game.num_profiles)
    profiles = game.profiles[perm]
    payoffs = game.payoffs[perm]
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
        mask = prof > 0
        spay = np.zeros(pay.shape + (num,))
        pview = spay.view()
        pview.shape = (-1, num)
        pview[mask.ravel()] = pay[mask, None] + noise((mask.sum(), num))

        new_profiles.append(prof)
        sample_payoffs.append(spay)

    if new_profiles:
        new_profiles = np.concatenate(new_profiles)
    else:  # No data
        new_profiles = np.empty((0, game.num_role_strats), dtype=int)
    return rsgame.samplegame_copy(game, new_profiles, sample_payoffs, False)


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
    spayoffs = game.payoffs[..., None].repeat(num_samples, -1)
    mask = game.profiles > 0
    samples = noise(max_width, mask.sum(), num_samples)
    expand_mask = np.broadcast_to(mask[..., None], mask.shape + (num_samples,))
    spayoffs[expand_mask] += samples.flat
    return rsgame.samplegame_copy(game, game.profiles, [spayoffs])


def serializer(game):
    """Generate a GameSerializer from a game"""
    role_names = ['all'] if game.is_symmetric(
    ) else utils.prefix_strings('r', game.num_roles)
    strat_names = [utils.prefix_strings('s', s) for s in game.num_strategies]
    return gameio.gameserializer(role_names, strat_names)
