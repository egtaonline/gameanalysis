import random
import itertools
from os import path
from collections import Counter

import numpy as np
import numpy.random as r

from gameanalysis import rsgame
from gameanalysis import utils


# import GameIO as IO

# from BasicFunctions import leading_zeros

# from functools import partial
# from itertools import combinations
# from bisect import bisect
# from numpy.random import uniform as U, normal, multivariate_normal, beta, gumbel
# from random import choice
# from numpy import array, arange, zeros, fill_diagonal, cumsum

# Populate word list for generating better names
_WORD_LIST_FILE = path.join(path.dirname(path.dirname(__file__)), '.wordlist.txt')
_WORD_LIST = []

try:
    with open(_WORD_LIST_FILE) as f:
        for word in f:
            _WORD_LIST.append(word[:-1])
except OSError:  # Something bad happened
    pass


def _index(iterable):
    """Returns a dictionary mapping elements to their index in the iterable"""
    return dict(map(reversed, enumerate(iterable)))


def _random_strings(number, prefix='x', padding=None, cool=False):
    """Generate random strings without repetition

    If cool (the default) these strings are generated from a word list,
    otherwise they are generated by adding a prefix to a bunch of zero padded
    integers. The padding defaults so that all of the strings are the same
    length.

    """
    if cool and _WORD_LIST:
        return random.sample(_WORD_LIST, number)
    else:
        if padding is None:
            padding = len(str(number - 1))
        return ('{}{:0{}d}'.format(prefix, i, padding) for i in range(number))


def _compact_payoffs(game):
    """Given a game returns a compact representation of the payoffs

    In this case compact means that they're in one ndarray. This representation
    is inefficient for almost everything but an independent game with full
    data.

    Parameters
    ----------
    game : rsgame.Game
        The game to generate a compact payoff matrix for

    Returns
    -------
    payoffs : ndarray; shape (s1, s2, ..., sn, n)
        payoffs[s1, s2, ..., sn, j] is the payoff to player j when player 1
        plays s1, player 2 plays s2, etc. n is the total number of players.

    strategies : [(role, [strat])]
        The first list indexes the players, and the second indexes the
        strategies for that player.

    """
    strategies = list(itertools.chain.from_iterable(
        itertools.repeat((role, list(strats)), game.players[role])
        for role, strats in game.strategies.items()))

    payoffs = np.empty([len(s) for _, s in strategies] + [len(strategies)])
    for profile, payoff in game.payoffs(as_array=True):
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
                     for p, pay in zip(profile, payoff)]))
        for inds, utils in perms:
            payoffs[inds] = utils
    return payoffs, strategies


def _gen_rs_game(num_roles, num_players, num_strategies, cool=False):
    """Create a role symmetric game"""
    try:
        num_players = list(num_players)
    except TypeError:
        num_players = [num_players] * num_roles
    try:
        num_strategies = list(num_strategies)
    except TypeError:
        num_strategies = [num_strategies] * num_roles

    assert len(num_players) == num_roles, \
        'length of num_players must equal num_roles'
    assert all(p > 0 for p in num_players), \
        'number of players must be greater than zero'
    assert len(num_strategies) == num_roles, \
        'length of num_strategies must equal num_roles'
    assert all(s > 0 for s in num_strategies), \
        'number of strategies must be greater than zero'

    # This list is necessary to maintain consistent order.
    roles = list(_random_strings(num_roles, prefix='r', cool=cool))
    strategies = {role: set(_random_strings(num_strat, prefix='s', cool=cool))
                  for role, num_strat
                  in zip(roles, num_strategies)}
    players = dict(zip(roles, num_players))
    return rsgame.EmptyGame(players, strategies)


def role_symmetric_game(num_roles, num_players, num_strategies,
                        distribution=lambda: r.uniform(-1, 1), cool=False):
    """Generate a random role symmetric game

    num_players and num_strategies can be scalers, or lists of length
    num_roles. Payoffs are drawn from distribution.

    """
    game = _gen_rs_game(num_roles, num_players, num_strategies, cool=cool)
    profile_data = [{role: [(strat, count, [distribution()])
                            for strat, count in strats.items()]
                     for role, strats in prof.items()}
                    for prof in game.all_profiles()]
    return rsgame.Game(game.players, game.strategies, profile_data)


def independent_game(num_players, num_strategies,
                     distribution=lambda: r.uniform(-1, 1), cool=False):
    """Generate an independent game

    All payoff values drawn independently according to specified
    distribution. The distribution defaults to uniform from -1 to 1.

    """
    return role_symmetric_game(num_players, 1, num_strategies, cool=cool)


def symmetric_game(num_players, num_strategies,
                   distribution=lambda: r.uniform(-1, 1), cool=False):
    """Generate a random symmetric game

    distribution defaults to uniform from -1 to 1.

    """
    return role_symmetric_game(1, num_players, num_strategies, cool=cool)


def covariant_game(num_players, num_strategies, mean_dist=lambda: 0, var=1,
                   covar_dist=lambda: r.uniform(-1, 1), cool=False):
    """Generate a covariant game

    Payoff values for each profile drawn according to multivariate normal.

    The multivariate normal for each profile has a constant mean-vector with
    value drawn from mean_dist, constant variance=var, and equal covariance
    between all pairs of players, drawn from covar_dist.

    mean_dist:  Distribution from which mean payoff for each profile is drawn.
                Defaults to constant 0.
    var:        Diagonal entries of covariance matrix
    covar_dist: Distribution from which the value of the off-diagonal
                covariance matrix entries for each profile is drawn

    Both mean_dist and covar_dist should be numpy-style random number
    generators that can return an array.

    """
    game = _gen_rs_game(num_players, 1, num_strategies, cool=cool)
    mean = np.empty(num_strategies)
    covar = np.empty((num_strategies, num_strategies))

    profile_data = []
    for prof in game.all_profiles():
        mean.fill(mean_dist())
        covar.fill(covar_dist())
        np.fill_diagonal(covar, var)
        payoffs = r.multivariate_normal(mean, covar)
        profile_data.append({role: [(utils.only(strats), 1, [payoffs[i]])]
                             for i, (role, strats) in enumerate(prof.items())})
    return rsgame.Game(game.players, game.strategies, profile_data)


def zero_sum_game(num_strategies, distribution=lambda: r.uniform(-1, 1),
                  cool=False):
    """Generate a two-player, zero-sum game

    2-player zero-sum game; player 1 payoffs drawn from given distribution

    distribution defaults to uniform between -1 and 1

    """
    game = _gen_rs_game(2, 1, num_strategies, cool=cool)
    role1, role2 = game.strategies
    profile_data = []
    for prof in game.all_profiles():
        row_strat = utils.only(prof[role1])
        col_strat = utils.only(prof[role2])
        row_payoff = distribution()
        profile_data.append({
            role1: [(row_strat, 1, [row_payoff])],
            role2: [(col_strat, 1, [-row_payoff])]})
    return rsgame.Game(game.players, game.strategies, profile_data)


def sym_2p2s_game(a=0, b=1, c=2, d=3,
                  distribution=lambda s=None: r.uniform(-1, 1, s),
                  cool=False):
    """Create a symmetric 2-player 2-strategy game of the specified form.

    Four payoff values get drawn from U(min_val, max_val), and then are
    assigned to profiles in order from smallest to largest according to the
    order parameters as follows:

       | s0  | s1  |
    ---|-----|-----|
    s0 | a,a | b,c |
    s1 | c,b | d,d |
    ---|-----|-----|

    So a=2,b=0,c=3,d=1 gives a prisoners' dilemma; a=0,b=3,c=1,d=2 gives a game
    of chicken.

    distribution must accept a size parameter a la numpy distributions.

    """
    game = _gen_rs_game(1, 2, 2, cool=cool)
    role, strats = utils.only(game.strategies.item())
    strats = list(strats)

    payoffs = sorted(distribution(4))
    profile_data = [
        {role: [(strats[0], 2, [payoffs[a]])]},
        {role: [(strats[0], 1, [payoffs[b]]),
                (strats[1], 1, [payoffs[c]])]},
        {role: [(strats[1], 2, [payoffs[d]])]}]
    return rsgame.Game(game.players, game.strategies, profile_data)


def congestion_game(num_players, num_facilities, num_required, **kwargs):
    """Generates random congestion games with num_players players and nCr(f, r)
    strategies

    Congestion games are symmetric, so all players belong to one role. Each
    strategy is a subset of size #required among the size #facilities set of
    available facilities. Payoffs for each strategy are summed over facilities.
    Each facility's payoff consists of three components:

    -constant ~ U[0, num_facilities]
    -linear congestion cost ~ U[-num_required, 0]
    -quadratic congestion cost ~ U[-1, 0]

    """
    role = 'all'
    strategies = list(itertools.combinations(range(num_facilities),
                                             num_required))

    values = r.random((num_facilities, 3))
    values[:, 0] *= num_facilities  # constant
    values[:, 1] *= -num_required   # linear
    values[:, 2] *= -1              # quadratic

    def strat_string(strat):
        """Turns strategy into a string"""
        return 'f{{{}}}'.format(', '.join(map(str, strat)))

    def to_array(facilities):
        """Turns an iterable of facilities into an array"""
        array = np.zeros(num_facilities, dtype=int)
        for index in facilities:
            array[index] += 1
        return array

    profile_data = []
    for prof in itertools.combinations_with_replacement(
            strategies, num_players):
        usage = to_array(itertools.chain.from_iterable(prof))
        payoffs = np.sum(usage[:, np.newaxis] ** np.arange(3) * values, 1)
        profile_data.append(
            {role: [(strat_string(strat), count, to_array(strat).dot(payoffs))
                    for strat, count in Counter(prof).items()]})

    return rsgame.Game({role: num_players},
                       {role: {strat_string(strat) for strat in strategies}},
                       profile_data)


def local_effect_game(num_players, num_strategies, cool=False):
    """Generates random congestion games with num_players (N) players and
    num_strategies (S) strategies.

    Local effect games are symmetric, so all players belong to role all. Each
    strategy corresponds to a node in the G(N, 2/S) (directed edros-renyi
    random graph with edge probability of 2/S) local effect graph. Payoffs for
    each strategy consist of constant terms for each strategy, and interaction
    terms for the number of players choosing that strategy and each neighboring
    strategy.

    The one-strategy terms are drawn as follows:
    -constant ~ U[-(N+S), N+S]
    -linear ~ U[-N, 0]

    The neighbor strategy terms are drawn as follows:
    -linear ~ U[-S, S]
    -quadratic ~ U[-1, 1]

    """
    game = _gen_rs_game(1, num_players, num_strategies)
    role, strategies = utils.only(game.strategies.items())
    strategies = list(strategies)
    smap = _index(strategies)

    # Generate local effects graph. This is an SxSx3 graph where the first two
    # axis are in and out nodes, and the final axis is constant, linear,
    # quadratic gains.
    #
    # XXX There's a little redundant computation here
    local_effects = np.empty((num_strategies, num_strategies, 3))
    # Fill in neighbors
    local_effects[..., 0] = 0
    local_effects[..., 1] = r.uniform(-num_strategies, num_strategies,
                                      (num_strategies, num_strategies))
    local_effects[..., 2] = r.uniform(-1, 1, (num_strategies, num_strategies))
    # Mask out some edges
    local_effects *= (r.random((num_strategies, num_strategies)) >
                      (2 / num_strategies))[..., np.newaxis]
    # Fill in self
    np.fill_diagonal(local_effects[..., 0],
                     r.uniform(-(num_players + num_strategies),
                               num_players + num_strategies,
                               num_strategies))
    np.fill_diagonal(local_effects[..., 1],
                     r.uniform(-num_players, 0, num_strategies))
    np.fill_diagonal(local_effects[..., 2], 0)

    def to_array(prof):
        """Returns an array representation of a profile"""
        array = np.zeros(num_strategies, dtype=int)
        indices, counts = zip(*prof['role'].items())
        array[list(indices)] = counts
        return array

    profile_data = []
    for prof in game.all_profiles():
        counts = to_array(prof)
        payoffs = np.sum(local_effects *
                         counts[:, np.newaxis] ** np.arange(3),
                         (1, 2))
        profile_data.append(
            {role: [(strat, count, [payoffs[smap[strat]]])
                    for strat, count in strats.items()]
             for role, strats in prof.items()})

    return rsgame.Game(game.players, game.strategies, profile_data)


def polymatrix_game(num_players, num_strategies, matrix_game=independent_game,
                    players_per_matrix=2, cool=False):
    """Creates a polymatrix game using the specified k-player matrix game function.

    Each player's payoff in each profile is a sum over independent games played
    against each set of opponents. Each k-tuple of players plays an instance of
    the specified random k-player matrix game.

    players_per_matrix: k
    matrix_game:        a function of two arguments (player_per_matrix,
                        num_strategies) that returns 2-player,
                        num_strategies-strategy games.

    Note: The actual roles and strategies of matrix game are ignored.

    """
    payoffs = np.zeros([num_strategies] * num_players + [num_players])
    for players in itertools.combinations(range(num_players),
                                          players_per_matrix):
        subgame = matrix_game(players_per_matrix, num_strategies)
        sub_payoffs, _ = _compact_payoffs(subgame)
        new_shape = np.array([1] * num_players + [players_per_matrix])
        new_shape[list(players)] = num_strategies
        payoffs[..., list(players)] += sub_payoffs.reshape(new_shape)

    game = _gen_rs_game(num_players, 1, num_strategies, cool=cool)
    indexible = [(role, list(strats)) for role, strats
                 in game.strategies.items()]

    profs = [{role: [(strats[ind], 1, value)]
              for value, ind, (role, strats)
              in zip(payoff, inds, indexible)}
             # This zip makes puts each array of player payoffs with the
             # corresponding strategy number for each player. It shouldn't be
             # hard to adjust this to allow an arbitrary number of strategies
             # per player.
             for payoff, inds
             in zip(payoffs.reshape((-1, num_players)),
                    itertools.product(range(num_strategies),
                                      repeat=num_players))]

    return rsgame.Game(game.players, game.strategies, profs)


# def add_noise(game, model, spread, samples):
#     """
#     Generate sample game with random noise added to each payoff.

#     game: a RSG.Game or RSG.SampleGame
#     model: a 2-parameter function that generates mean-zero noise
#     spread, samples: the parameters passed to the noise function
#     """
#     sg = SampleGame(game.roles, game.players, game.strategies)
#     for prof in game.knownProfiles():
#         sg.addProfile({r:[PayoffData(s, prof[r][s], game.getPayoff(prof,r,s) + \
#                 model(spread, samples)) for s in prof[r]] for r in game.roles})
#     return sg


# def gaussian_mixture_noise(max_stdev, samples, modes=2, spread_mult=2):
#     """
#     Generate Gaussian mixture noise to add to one payoff in a game.

#     max_stdev: maximum standard deviation for the mixed distributions (also
#                 affects how widely the mixed distributions are spaced)
#     samples: numer of samples to take of every profile
#     modes: number of Gaussians to mix
#     spread_mult: multiplier for the spread of the Gaussians. Distance between
#                 the mean and the nearest distribution is drawn from
#                 N(0,max_stdev*spread_mult).
#     """
#     multipliers = arange(float(modes)) - float(modes-1)/2
#     offset = normal(0, max_stdev * spread_mult)
#     stdev = beta(2,1) * max_stdev
#     return [normal(choice(multipliers)*offset, stdev) for _ in range(samples)]


# eq_var_normal_noise = partial(normal, 0)
# normal_noise = partial(gaussian_mixture_noise, modes=1)
# bimodal_noise = partial(gaussian_mixture_noise, modes=2)


# def nonzero_gaussian_noise(max_stdev, samples, prob_pos=0.5, spread_mult=1):
#     """
#     Generate Noise from a normal distribution centered up to one stdev from 0.

#     With prob_pos=0.5, this implements the previous buggy output of
#     bimodal_noise.

#     max_stdev: maximum standard deviation for the mixed distributions (also
#                 affects how widely the mixed distributions are spaced)
#     samples: numer of samples to take of every profile
#     prob_pos: the probability that the noise mean for any payoff will be >0.
#     spread_mult: multiplier for the spread of the Gaussians. Distance between
#                 the mean and the mean of the distribution is drawn from
#                 N(0,max_stdev*spread_mult).
#     """
#     offset = normal(0, max_stdev)*(1 if U(0,1) < prob_pos else -1)*spread_mult
#     stdev = beta(2,1) * max_stdev
#     return normal(offset, stdev, samples)


# def uniform_noise(max_half_width, samples):
#     """
#     Generate uniform random noise to add to one payoff in a game.

#     max_range: maximum half-width of the uniform distribution
#     samples: numer of samples to take of every profile
#     """
#     hw = beta(2,1) * max_half_width
#     return U(-hw, hw, samples)


# def gumbel_noise(scale, samples, flip_prob=0.5):
#     """
#     Generate random noise according to a gumbel distribution.

#     Gumbel distributions are skewed, so the default setting of the flip_prob
#     parameter makes it equally likely to be skewed positive or negative

#     variance ~= 1.6*scale
#     """
#     location = -0.5772*scale
#     multiplier = -1 if (U(0,1) < flip_prob) else 1
#     return multiplier * gumbel(location, scale, samples)


# def mix_models(models, rates, spread, samples):
#     """
#     Generate SampleGame with noise drawn from several models.

#     models: a list of 2-parameter noise functions to draw from
#     rates: the probabilites with which a payoff will be drawn from each model
#     spread, samples: the parameters passed to the noise functions
#     """
#     cum_rates = cumsum(rates)
#     m = models[bisect(cum_rates, U(0,1))]
#     return m(spread, samples)


# n80b20_noise = partial(mix_models, [normal_noise, bimodal_noise], [.8,.2])
# n60b40_noise = partial(mix_models, [normal_noise, bimodal_noise], [.6,.4])
# n40b60_noise = partial(mix_models, [normal_noise, bimodal_noise], [.4,.6])
# n20b80_noise = partial(mix_models, [normal_noise, bimodal_noise], [.2,.8])

# equal_mix_noise = partial(mix_models, [normal_noise, bimodal_noise, \
#         uniform_noise, gumbel_noise], [.25]*4)
# mostly_normal_noise =  partial(mix_models, [normal_noise, bimodal_noise, \
#         gumbel_noise], [.8,.1,.1])

# noise_functions = filter(lambda k: k.endswith("_noise") and not \
#                     k.startswith("add_"), globals().keys())

# def rescale_payoffs(game, min_payoff=0, max_payoff=100):
#     """
#     Rescale game's payoffs to be in the range [min_payoff, max_payoff].

#     Modifies game.values in-place.
#     """
#     game.makeArrays()
#     min_val = game.values.min()
#     max_val = game.values.max()
#     game.values -= min_val
#     game.values *= (max_payoff - min_payoff)
#     game.values /= (max_val - min_val)
#     game.values += min_payoff
