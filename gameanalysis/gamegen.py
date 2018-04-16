"""Module for generating random games"""
import itertools
from collections import abc

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import aggfn
from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import utils


def default_distribution(shape=None):
    """Default distribution for payoffs"""
    return rand.uniform(-1, 1, shape)


def gen_profiles(base, prob=1.0, distribution=default_distribution):
    """Generate profiles given game structure

    Parameters
    ----------
    base : RsGame
        Game to generate payoffs for.
    prob : float, optional
        The probability to add a profile from the full game.
    distribution : (shape) -> ndarray, optional
        Distribution function to draw payoffs from.
    """
    # First turn input into number of profiles to compute
    num_profs = base.num_all_profiles
    utils.check(0 <= prob <= 1, 'probability must be in [0, 1] but was {:g}',
                prob)
    if num_profs <= np.iinfo(int).max:
        num = rand.binomial(num_profs, prob)
    else:
        num = round(float(num_profs * prob))
    return gen_num_profiles(base, num, distribution)


def gen_num_profiles(base, num, distribution=default_distribution):
    """Generate profiles given game structure

    Parameters
    ----------
    base : RsGame
        Game to generate payoffs for.
    count : int
        The number of profiles to generate.
    distribution : (shape) -> ndarray, optional
        Distribution function to draw payoffs from.
    """
    utils.check(
        0 <= num <= base.num_all_profiles,
        'num must be in [0, {:d}] but was {:d}', base.num_all_profiles, num)
    profiles = sample_profiles(base, num)
    payoffs = np.zeros(profiles.shape)
    mask = profiles > 0
    payoffs[mask] = distribution(mask.sum())
    return paygame.game_replace(base, profiles, payoffs)


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
    return gen_profiles(rsgame.empty(players, strats), prob, distribution)


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
        rsgame.empty(players, strats), num, distribution)


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


def gen_noise( # pylint: disable=too-many-arguments,too-many-locals
        base, prob=0.5, min_samples=1, min_width=0, max_width=1,
        noise_distribution=width_gaussian):
    """Generate noise for profiles of a game

    This generates samples for payoff data by first generating some measure of
    distribution spread for each payoff in the game. Then, for each sample,
    noise is drawn from this distribution. As a result, some payoffs will have
    significantly more noise than other payoffs, helping to mimic real games.

    Parameters
    ----------
    base : Game
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
    if base.is_empty():
        return paygame.samplegame_copy(base)
    perm = rand.permutation(base.num_profiles)
    profiles = base.profiles()[perm]
    payoffs = base.payoffs()[perm]
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
        spay = np.zeros((pay.shape[0], num, base.num_strats))
        pview = np.rollaxis(spay, 1, 3)
        widths = rand.uniform(min_width, max_width, supp.sum())
        pview[supp] = pay[supp, None] + noise_distribution(widths, num)

        new_profiles.append(prof)
        sample_payoffs.append(spay)

    if new_profiles:
        new_profiles = np.concatenate(new_profiles)
    else:  # No data
        new_profiles = np.empty((0, base.num_strats), dtype=int)
    return paygame.samplegame_replace(base, new_profiles, sample_payoffs)


def samplegame( # pylint: disable=too-many-arguments
        players, strats, prob=0.5, min_samples=1, min_width=0, max_width=1,
        payoff_distribution=default_distribution,
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


def covariant_game(
        num_role_strats, mean_dist=np.zeros, var_dist=np.ones,
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
    _, diag, right = np.linalg.svd(var)
    payoffs = rand.normal(size=shape)
    payoffs = np.einsum('...i,...i,...ij->...j', payoffs, np.sqrt(diag), right)
    payoffs += mean_dist(shape)
    return matgame.matgame(payoffs)


def two_player_zero_sum_game(num_role_strats,
                             distribution=default_distribution):
    """Generate a two-player, zero-sum game"""
    # Generate player 1 payoffs
    num_role_strats = np.broadcast_to(num_role_strats, 2)
    p1_payoffs = distribution(num_role_strats)[..., None]
    return matgame.matgame(np.concatenate([p1_payoffs, -p1_payoffs], -1))


def sym_2p2s_game(a, b, c, d, distribution=default_distribution): # pylint: disable=invalid-name
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
    utils.check({a, b, c, d} == set(range(4)), 'numbers must be each of 1-4')
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


def polymatrix_game(
        num_players, num_strats, matrix_game=independent_game,
        players_per_matrix=2):
    """Creates a polymatrix game

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
    utils.check(
        all(l < 0 for l in loss) and all(w > 0 for w in win) and len(loss) == 3
        and len(win) == 3,
        'win must be greater than 0 and loss must be less than zero')
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
    utils.check(players > 1, 'players must be more than one')
    utils.check(max_value > 2, 'max value must be more than 2')
    base = rsgame.empty(players, max_value - 1)
    profiles = base.all_profiles()
    payoffs = np.zeros(profiles.shape)
    mins = np.argmax(profiles, -1)
    mask = profiles > 0
    payoffs[mask] = mins.repeat(mask.sum(-1))
    rows = np.arange(profiles.shape[0])
    ties = profiles[rows, mins] > 1
    lowest_pays = mins + 4
    lowest_pays[ties] -= 2
    payoffs[rows, mins] = lowest_pays
    return paygame.game_replace(base, profiles, payoffs)


def keep_profiles(base, keep_prob=0.5):
    """Keep random profiles from an existing game

    Parameters
    ----------
    base : RsGame
        The game to take profiles from.
    keep_prob : float, optional
        The probability of keeping a profile from the full game.
    """
    # First turn input into number of profiles to compute
    num_profs = base.num_profiles
    utils.check(
        0 <= keep_prob <= 1, 'keep probability must be in [0, 1] but was {:g}',
        keep_prob)
    if num_profs <= np.iinfo(int).max:
        num = rand.binomial(num_profs, keep_prob)
    else:
        num = round(float(num_profs * keep_prob))
    return keep_num_profiles(base, num)


def keep_num_profiles(base, num):
    """Keep random profiles from an existing game

    Parameters
    ----------
    base : RsGame
        Game to keep profiles from.
    num : int
        The number of profiles to keep from the game.
    """
    utils.check(
        0 <= num <= base.num_profiles, 'num must be in [0, {:d}] but was {:d}',
        base.num_profiles, num)
    if num == 0:
        profiles = np.empty((0, base.num_strats), int)
        payoffs = np.empty((0, base.num_strats))
    elif base.is_complete():
        profiles = sample_profiles(base, num)
        payoffs = base.get_payoffs(profiles)
    else:
        inds = rand.choice(base.num_profiles, num, replace=False)
        profiles = base.profiles()[inds]
        payoffs = base.payoffs()[inds]
    return paygame.game_replace(base, profiles, payoffs)


def sample_profiles(base, num): # pylint: disable=inconsistent-return-statements
    """Generate unique profiles from a game

    Parameters
    ----------
    base : RsGame
        Game to generate random profiles from.
    num : int
        Number of profiles to sample from the game.
    """
    if num == base.num_all_profiles: # pylint: disable=no-else-return
        return base.all_profiles()
    elif num == 0:
        return np.empty((0, base.num_strats), int)
    elif base.num_all_profiles <= np.iinfo(int).max:
        inds = rand.choice(base.num_all_profiles, num, replace=False)
        return base.profile_from_id(inds)
    else:
        # Number of times we have to re-query
        ratio = (sps.digamma(float(base.num_all_profiles)) -
                 sps.digamma(float(base.num_all_profiles - num)))
        # Max is for underflow
        num_per = max(round(float(ratio * base.num_all_profiles)), num)
        profiles = set()
        while len(profiles) < num:
            profiles.update(
                utils.hash_array(p) for p in base.random_profiles(num_per))
        profiles = np.stack([h.array for h in profiles])
        inds = rand.choice(profiles.shape[0], num, replace=False)
        return profiles[inds]


def _random_inputs(prob, num_strats, num_funcs):
    """Returns a random mask without all true or all false per function"""
    vals = np.random.random((num_strats, num_funcs))
    mask = vals < prob
    inds = np.arange(num_funcs)
    mask[vals.argmin(0), inds] = True
    mask[vals.argmax(0), inds] = False
    return mask


def _random_mask(prob, num_funcs, num_strats):
    """Returns a random mask with at least one true in every row and col"""
    vals = np.random.random((num_funcs, num_strats))
    mask = vals < prob
    mask[np.arange(num_funcs), vals.argmin(1)] = True
    mask[vals.argmin(0), np.arange(num_strats)] = True
    return mask


def _random_weights(prob, num_funcs, num_strats):
    """Returns random action weights"""
    return (_random_mask(prob, num_funcs, num_strats) *
            np.random.normal(0, 1, (num_funcs, num_strats)))


def normal_aggfn(role_players, role_strats, functions, *, input_prob=0.2,
                 weight_prob=0.2):
    """Generate a random normal AgfnGame

    Each function value is an i.i.d Gaussian random walk.

    Parameters
    ----------
    role_players : int or ndarray
        The number of players per role.
    role_strats : int or ndarray
        The number of strategies per role.
    functions : int
        The number of functions to generate.
    input_prob : float, optional
        The probability of a strategy counting towards a function value.
    weight_prob : float, optional
        The probability of a function producing non-zero payoffs to a strategy.
    """
    base = rsgame.empty(role_players, role_strats)
    inputs = _random_inputs(input_prob, base.num_strats, functions)
    weights = _random_weights(weight_prob, functions, base.num_strats)

    shape = (functions,) + tuple(base.num_role_players + 1)
    funcs = np.random.normal(0, 1 / np.sqrt(base.num_players + 1), shape)
    for role in range(1, base.num_roles + 1):
        funcs.cumsum(role, out=funcs)
    mean = funcs.mean(tuple(range(1, base.num_roles + 1)))
    mean.shape = (functions,) + (1,) * base.num_roles
    funcs -= mean
    return aggfn.aggfn_replace(base, weights, inputs, funcs)


def _random_aggfn( # pylint: disable=too-many-arguments
        role_players, role_strats, functions, input_prob, weight_prob,
        role_dist):
    """Base form for structured random aggfn generation

    role_dist takes a number of functions and a number of players and returns
    an ndarray of the function values.
    """
    base = rsgame.empty(role_players, role_strats)
    inputs = _random_inputs(input_prob, base.num_strats, functions)
    weights = _random_weights(weight_prob, functions, base.num_strats)

    funcs = np.ones((functions,) + tuple(base.num_role_players + 1))
    base_shape = [functions] + [1] * base.num_roles
    for role, play in enumerate(base.num_role_players):
        role_funcs = role_dist(functions, play)
        shape = base_shape.copy()
        shape[role + 1] = play + 1
        role_funcs.shape = shape
        funcs *= role_funcs
    return aggfn.aggfn_replace(base, weights, inputs, funcs)


def poly_aggfn(
        role_players, role_strats, functions, *, input_prob=0.2,
        weight_prob=0.2, degree=4):
    """Generate a random polynomial AgfnGame

    Functions are generated by generating `degree` zeros in [0, num_players] to
    serve as a polynomial functions.

    Parameters
    ----------
    role_players : int or ndarray
        The number of players per role.
    role_strats : int or ndarray
        The number of strategies per role.
    functions : int
        The number of functions to generate.
    input_prob : float, optional
        The probability of a strategy counting towards a function value.
    weight_prob : float, optional
        The probability of a function producing non-zero payoffs to a strategy.
    degree : int or [float], optional
        Either an integer specifying the degree or a list of the probabilities
        of degrees starting from one, e.g. 3 is the same as [0, 0, 1].
    """
    if isinstance(degree, int):
        degree = (0,) * (degree - 1) + (1,)
    max_degree = len(degree)

    def role_dist(functions, play):
        """Role distribution"""
        zeros = (np.random.random((functions, max_degree)) * 1.5 - 0.25) * play
        terms = np.arange(play + 1)[:, None] - zeros[:, None]
        choices = np.random.choice(
            max_degree, (functions, play + 1), True, degree)
        terms[choices[..., None] < np.arange(max_degree)] = 1
        poly = terms.prod(2) / play ** choices

        # The prevents too many small polynomials from making functions
        # effectively constant
        scale = poly.max() - poly.min()
        offset = poly.min() + 1
        return (poly - offset) / (1 if np.isclose(scale, 0) else scale)

    return _random_aggfn(role_players, role_strats, functions, input_prob,
                         weight_prob, role_dist)


def sine_aggfn(role_players, role_strats, functions, *, input_prob=0.2,
               weight_prob=0.2, period=4):
    """Generate a random sinusodial AgfnGame

    Functions are generated by generating sinusoids with uniform random shifts
    and n periods in 0 to num_players, where n is chosen randomle between
    min_period and max_period.

    Parameters
    ----------
    role_players : int or ndarray
        The number of players per role.
    role_strats : int or ndarray
        The number of strategies per role.
    functions : int
        The number of functions to generate.
    input_prob : float, optional
        The probability of a strategy counting towards a function value.
    weight_prob : float, optional
        The probability of a function producing non-zero payoffs to a strategy.
    period : float, optional
        The loose number of periods in the payoff for each function.
    """
    def role_dist(functions, play):
        """Distribution by role"""
        # This setup makes it so that the beat frequencies approach period
        periods = ((np.arange(1, functions + 1) +
                    np.random.random(functions) / 2 - 1 / 4) *
                   period / functions)
        offset = np.random.random((functions, 1))
        return np.sin(
            (np.linspace(0, 1, play + 1) * periods[:, None] + offset) * 2 *
            np.pi)

    return _random_aggfn(role_players, role_strats, functions, input_prob,
                         weight_prob, role_dist)


def _random_monotone_polynomial(functions, players, degree):
    """Generates a random monotone polynomial table"""
    coefs = (np.random.random((functions, degree + 1)) /
             players ** np.arange(degree + 1))
    powers = np.arange(players + 1) ** np.arange(degree + 1)[:, None]
    return coefs.dot(powers)


def congestion(num_players, num_facilities, num_required, *, degree=2):
    """Generate a congestion game

    A congestion game is a symmetric game, where there are a given number of
    facilities, and each player must choose to use some amount of them. The
    payoff for each facility decreases as more players use it, and a players
    utility is the sum of the utilities for every facility.

    In this formulation, facility payoffs are random polynomials of the number
    of people using said facility.

    Parameters
    ----------
    num_players : int > 1
        The number of players.
    num_facilities : int > 1
        The number of facilities.
    num_required : 0 < int < num_facilities
        The number of required facilities.
    degree : int > 0, optional
        Degree of payoff polynomials.
    """
    utils.check(num_players > 1, 'must have more than one player')
    utils.check(num_facilities > 1, 'must have more than one facility')
    utils.check(
        0 < num_required < num_facilities,
        'must require more than zero but less than num_facilities')
    utils.check(degree > 0, 'degree must be greater than zero')

    function_inputs = utils.acomb(num_facilities, num_required)
    functions = -_random_monotone_polynomial(num_facilities, num_players,
                                             degree)

    facs = tuple(utils.prefix_strings('', num_facilities))
    strats = tuple('_'.join(facs[i] for i, m in enumerate(mask) if m)
                   for mask in function_inputs)
    return aggfn.aggfn_names(
        ['all'], num_players, [strats], function_inputs.T, function_inputs,
        functions)


def local_effect(num_players, num_strategies, *, edge_prob=0.2):
    """Generate a local effect game

    In a local effect game, strategies are connected by a graph, and utilities
    are a function of the number of players playing our strategy and the number
    of players playing a neighboring strategy, hence local effect.

    In this formulation, payoffs for others playing our strategy are negative
    quadratics, and payoffs for playing other strategies are positive cubics.

    Parameters
    ----------
    num_players : int > 1
        The number of players.
    num_strategies : int > 1
        The number of strategies.
    edge_prob : float, optional
        The probability that one strategy affects another.
    """
    utils.check(num_players > 1, "can't generate a single player game")
    utils.check(num_strategies > 1, "can't generate a single strategy game")

    local_effect_graph = np.random.rand(
        num_strategies, num_strategies) < edge_prob
    np.fill_diagonal(local_effect_graph, False)
    num_neighbors = local_effect_graph.sum()
    num_functions = num_neighbors + num_strategies

    action_weights = np.eye(num_functions, num_strategies, dtype=float)
    function_inputs = np.eye(num_strategies, num_functions, dtype=bool)
    in_act, out_act = local_effect_graph.nonzero()
    func_inds = np.arange(num_strategies, num_functions)
    function_inputs[in_act, func_inds] = True
    action_weights[func_inds, out_act] = 1

    function_table = np.empty((num_functions, num_players + 1), float)
    function_table[:num_strategies] = -_random_monotone_polynomial(
        num_strategies, num_players, 2)
    function_table[num_strategies:] = _random_monotone_polynomial(
        num_neighbors, num_players, 3)
    return aggfn.aggfn(num_players, num_strategies, action_weights,
                       function_inputs, function_table)
