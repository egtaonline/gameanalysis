import itertools
import math
import os
import random
import warnings

from nose import tools
import numpy as np

from gameanalysis import profile
from gameanalysis import randgames
from gameanalysis import rsgame
from gameanalysis import utils
from test import testutils

TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps

# The lambda: 0 means that the payoffs are all zero, since they don't matter
SMALL_GAMES = [
    randgames.symmetric_game(2, 2),
    randgames.symmetric_game(2, 5),
    randgames.symmetric_game(5, 2),
    randgames.symmetric_game(5, 5),
    randgames.independent_game(2, 2),
    randgames.independent_game(2, 5),
    randgames.independent_game(5, 2),
    randgames.independent_game(5, 5),
    randgames.role_symmetric_game(2, [1, 2], [2, 1]),
    randgames.role_symmetric_game(2, 2, 2),
    randgames.role_symmetric_game(2, 2, 5),
    randgames.role_symmetric_game(2, 5, 2),
    randgames.role_symmetric_game(2, 5, 5),
    randgames.symmetric_game(170, 2),  # approximate devreps
    randgames.symmetric_game(180, 2),  # actual devreps
]


def generate_games(allow_big=False):
    """Returns a generator for game testing"""
    for game in SMALL_GAMES:
        yield game

    if allow_big and os.getenv('BIG_TESTS') == 'ON':  # Big Games
        yield randgames.symmetric_game(1000, 2)
        yield randgames.symmetric_game(5, 40)
        yield randgames.symmetric_game(3, 160)
        yield randgames.symmetric_game(50, 2)
        yield randgames.symmetric_game(20, 5)
        yield randgames.symmetric_game(90, 5)
        yield randgames.role_symmetric_game(2, 2, 40)
        yield randgames.symmetric_game(12, 12)


def approx_dev_reps(game):
    if game._dev_reps.dtype == object:
        return game._dev_reps
    approx = np.array(np.round(game._dev_reps), dtype=object)
    view = approx.ravel()
    for i, x in enumerate(view):
        view[i] = int(x)
    return approx


def exact_dev_reps(game):
    """Uses python ints to compute dev reps. Much slower"""
    counts = game._counts
    dev_reps = np.empty_like(counts, dtype=object)
    strat_counts = list(game.players.values())
    fcount = [math.factorial(x) for x in strat_counts]
    for dev_prof, count_prof in zip(dev_reps, counts):
        total = utils.prod(fc // utils.prod(math.factorial(x) for x in cs)
                           for fc, cs in zip(fcount, count_prof))
        for dev_role, counts_role, strat_count \
                in zip(dev_prof, count_prof, strat_counts):
            for s, count in enumerate(counts_role):
                dev_role[s] = total * int(count) // strat_count
    return dev_reps


@testutils.apply(zip(generate_games(True)))
def devreps_approx_test(game):
    approx = approx_dev_reps(game)
    exact = exact_dev_reps(game)
    diff = (approx - exact) / (exact + TINY)
    assert np.all(np.abs(diff) < EPS), \
        "dev reps were not close enough ({:f})".format(diff)


@testutils.apply([
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (1, 2, 2),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (2, 2, 2),
    (2, [1, 2], 2),
    (2, 2, [1, 2]),
    (2, [1, 2], [1, 2]),
    (2, [3, 4], [2, 3]),
])
# Test that all functions work on an EmptyGame
def empty_game_function_test(roles, players, strategies):
    game = randgames.empty_role_symmetric_game(roles, players, strategies)
    game.players  # Has players
    game.strategies  # Has strategies
    game.size  # Has size

    # Test that all profiles returns the correct number of things
    num_all_profs = 0
    for _ in game.all_profiles():
        num_all_profs += 1
    assert num_all_profs == game.size, \
        "size or all profile generation is wrong"

    # Assert as_ methods work
    assert isinstance(game.as_profile(game._mask), profile.Profile), \
        "as_profile did not return a profile"
    assert isinstance(game.as_profile(game.as_profile(game._mask)),
                      profile.Profile), \
        "as_profile twice did not return a profile"
    assert isinstance(game.as_mixture(game._mask), profile.Mixture), \
        "as_mixture did not return a mixture"
    assert isinstance(game.as_mixture(game.as_mixture(game._mask)),
                      profile.Mixture), \
        "as_mixture twice did not return a mixture"
    assert isinstance(game.as_array(next(game.all_profiles())), np.ndarray), \
        "as_array did not return an array"
    assert isinstance(game.as_array(game.as_array(next(game.all_profiles()))),
                      np.ndarray), \
        "as_array twice did not return an array"

    # Assert that mixture calculations do the right thing
    mix = game.uniform_mixture(as_array=True)
    assert np.allclose(mix.sum(1), 1), "uniform mixture wasn't a mixture"
    masked = np.ma.masked_array(mix, mix == 0)
    if max(len(s) for s in game.strategies.values()) == 1:
        assert (mix == 1).all(), "uniform mixtures wasn't uniform"
    else:
        assert np.allclose(np.diff(masked, 1), 0), \
            "uniform mixture wasn't uniform"

    mix = game.uniform_mixture(as_array=False)
    assert all(abs(sum(strats.values()) - 1) < EPS
               for strats in mix.values()), \
        "uniform mixture wasn't a mixture"
    assert all(max(abs(next(iter(strats.values())) - p)
                   for p in strats.values()) < EPS
               for strats in mix.values()), \
        "uniform mixture wasn't uniform"

    mix = game.random_mixture(as_array=True)
    assert np.allclose(mix.sum(1), 1), "random mixture wasn't a mixture"

    mix = game.random_mixture(as_array=False)
    assert all(abs(sum(strats.values()) - 1) < EPS
               for strats in mix.values()), \
        "random mixture wasn't a mixture"

    bias = 0.6
    mixes = game.biased_mixtures(as_array=True, bias=bias)
    saw_bias = np.zeros_like(game._mask, dtype=bool)
    count = 0
    for mix in mixes:
        count += 1
        saw_bias |= mix == bias

    num_strats = game._mask.sum(1)
    assert np.prod(num_strats[num_strats > 1] + 1) == count, \
        'Didn\'t generate the proper number of mixtures'
    assert np.all(
        saw_bias  # observed a bias
        | (~game._mask)  # couldn't have observed one
        | (game._mask.sum(1) == 1)[:, None]  # Only one strat so no bias
    ), 'Didn\'t bias every strategy'

    mixes = game.biased_mixtures(as_array=False, bias=bias)
    for _ in mixes:
        # TODO Do more than just call this method
        pass

    mixes = game.pure_mixtures(as_array=True)
    for mix in mixes:
        assert np.allclose(mix.sum(1), 1), "pure mixtures weren't mixtures"
        assert ((mix == 1).sum(1) == 1).all(), \
            "not all roles in pure mixture had an assignment"

    mixes = game.pure_mixtures(as_array=False)
    for mix in mixes:
        assert all(len(strats.values()) == 1 for strats in mix.values()), \
            "pure mixtures weren't pure"
        assert all(next(iter(strats.values())) == 1
                   for strats in mix.values()), \
            "pure mixtures weren't mixtures"

    # Test that various methods can be called
    game.to_json()
    str(game)
    repr(game)


@testutils.apply(zip(generate_games()))
# Test that game functions work
def game_function_test(game):
    # Check that min payoffs are actually minimum
    min_payoffs = game.min_payoffs()
    assert all(all(all(min_payoffs[role] - EPS < p for p in pay.values())
                   for role, pay in payoff.items())
               for _, payoff in game.payoffs()), \
        "not all payoffs less than min payoffs"

    # Test profile methods
    prof_count = 0
    for prof in game.data_profiles():
        prof_count += 1
        role = next(iter(random.sample(list(prof), 1)))
        strategy = next(iter(random.sample(list(prof[role]), 1)))

        game.get_payoff(prof, role, strategy)  # Works
        game.get_payoffs(prof)  # Works
        game[prof]  # Works
        game.get_payoffs(prof, as_array=True)  # Works

        assert prof in game, "profile from game not in game"

    missing_prof = prof.add(role, strategy)
    assert game.get_payoff(missing_prof, role, strategy, ()) == (), \
        "default payoff didn't return default"

    assert prof_count == game.size, \
        "game not complete or data_profiles missing data"

    # Test expected payoff
    mix = game.random_mixture()
    assert not any(map(math.isnan, game.get_expected_payoff(mix).values())), \
        "some dict expected payoffs were nan"
    assert not np.isnan(game.get_expected_payoff(mix, as_array=True)).any(), \
        "some array expected payoffs were nan"
    assert not any(any(map(math.isnan, strats.values()))
                   for strats in game.expected_values(mix).values()), \
        "some dict expected values were nan"
    assert not np.isnan(game.expected_values(mix, as_array=True)).any(), \
        "some array expected values were nan"

    # Max social welfare
    game.get_max_social_welfare()  # Works
    game.get_max_social_welfare(as_array=True)  # Works
    for role in game.strategies:
        game.get_max_social_welfare(role=role)  # Works
        game.get_max_social_welfare(role=role, as_array=True)  # Works

    # Test that various methods can be called
    game.to_json()
    str(game)
    repr(game)
    iter(game)


@testutils.apply([
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (1, 2, 2),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (2, 2, 2),
    (2, [1, 2], 2),
    (2, 2, [1, 2]),
    (2, [1, 2], [1, 2]),
    (2, [3, 4], [2, 3]),
])
# Test that a Game with no data can still be created
def empty_full_game_test(roles, players, strategies):
    empty_game = randgames.empty_role_symmetric_game(roles, players,
                                                     strategies)
    max_strats = max(len(s) for s in empty_game.strategies.values())
    counts = np.empty([0, roles, max_strats], dtype=int)
    values = np.empty([0, roles, max_strats])
    game = rsgame.Game(empty_game.players, empty_game.strategies, counts,
                       values)

    # Check that min payoffs are actually minimum
    game.min_payoffs()
    assert len(list(game.payoffs())) == 0, \
        "returned payoffs"

    assert len(list(game.data_profiles())) == 0, \
        "returned payoffs"

    # Test expected payoff
    mix = game.random_mixture()
    assert all(map(math.isnan, game.get_expected_payoff(mix).values())), \
        "not all expected payoffs were nan"
    assert all(all(map(math.isnan, strats.values()))
               for strats in game.expected_values(mix).values()), \
        "not all expected values were nan"

    # Max social welfare
    assert (np.nan, None) == game.get_max_social_welfare(), \
        "Didn't return an empty welfare"

    # Test that various methods can be called
    game.to_json()
    str(game)
    repr(game)


@testutils.apply(zip(generate_games(), itertools.cycle([1, 2, 5, 10])))
# Test that game functions work
def sample_game_function_test(game, samples):
    game = randgames.add_noise(game, samples)

    for prof in game.data_profiles():
        game.get_sample_payoffs(prof)  # Works
        game.get_sample_payoffs(prof, as_array=True)  # Works

    assert len(list(game.sample_payoffs())) == game.size, \
        "sample payoffs not the right size"
    assert len(list(game.sample_payoffs(as_array=True))) == game.size, \
        "sample payoffs not the right size"

    # Test that various methods can be called
    game.num_samples()
    game.remean()
    game.resample()
    game.resample(1)

    game.to_json()
    str(game)
    repr(game)


@testutils.apply([
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (1, 2, 2),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (2, 2, 2),
    (2, [1, 2], 2),
    (2, 2, [1, 2]),
    (2, [1, 2], [1, 2]),
    (2, [3, 4], [2, 3]),
])
# Test that a Game with no data can still be created
def empty_sample_game_test(roles, players, strategies):
    empty_game = randgames.empty_role_symmetric_game(roles, players,
                                                     strategies)
    max_strats = max(len(s) for s in empty_game.strategies.values())
    counts = np.empty([0, roles, max_strats], dtype=int)
    sample_values = []
    game = rsgame.SampleGame(empty_game.players, empty_game.strategies, counts,
                             sample_values)

    assert len(list(game.sample_payoffs())) == 0, \
        "some sample payoffs in empty game"

    # Test that various methods can be called
    game.num_samples()
    game.remean()
    game.resample()
    game.resample(1)

    game.to_json()
    str(game)
    repr(game)


@testutils.apply(zip([(), None]))
# Test that null payoff warnings are detected
def null_payoff_test(null_type):
    players = {'role': 1}
    strategies = {'role': ['strat']}
    prof_data = [{'role': [('strat', 1, null_type)]}]

    with warnings.catch_warnings(record=True) as w:
        rsgame.Game.from_payoff_format(players, strategies, prof_data)

        assert len(w) == 1, "raised more than one warning"
        w = w[0].message  # warning object
        assert isinstance(w, UserWarning), \
            "warning wasn't a user warning"
        assert str(w).startswith('Encountered null payoff data in profile:'), \
            "did not raise the proper warning message"


# Test for warn on copy
def copy_test():
    players = {'role': 1}
    strategies = {'role': ['strat']}
    prof_data = (p for p in [{'role': [('strat', 1, [5])]}])

    with warnings.catch_warnings(record=True) as w:
        rsgame.Game.from_payoff_format(players, strategies, prof_data)

        assert len(w) == 1, "raised more than one warning"
        w = w[0].message  # warning object
        assert isinstance(w, UserWarning), \
            "warning wasn't a user warning"
        assert str(w) == ('Copying profile data, this usually indicates '
                          'something went wrong'), \
            "did not raise the proper warning message ({})".format(w)


# Test that sample game from matrix creates a game
@testutils.apply([
    (1, 1, 1),
    (1, 1, 3),
    (1, 2, 1),
    (1, 2, 5),
    (2, 1, 1),
    (2, 1, 4),
    (2, 2, 1),
    (2, 2, 2),
    (3, 4, 1),
    (3, 4, 2),
], repeat=20)
def sample_game_from_matrix_test(players, strategies, samples):
    matrix = np.random.random([strategies] * players + [players, samples])
    strats = {str(r): [str(s) for s in range(strategies)]
              for r in range(players)}
    game = rsgame.SampleGame.from_matrix(strats, matrix)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == players, \
        "didn't generate correct number of players"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert all(len(s) == strategies for s in game.strategies.values()), \
        "didn't generate correct number of strategies {:d} vs {}".format(
            strategies, [len(s) for s in game.strategies.values()])
    assert len(game.num_samples()) == 1, \
        "profiles had a different number of samples"
    assert next(iter(game.num_samples())) == samples, \
        "profiles didn't have correct number of samples"


# Test sample game with different number of samples
def different_samples_test():
    players = {'a': 1, 'b': 1}
    strategies = {'a': ['c'], 'b': ['d', 'e']}
    prof_data = [
        {'a': [('c', 1, [5])],
         'b': [('d', 1, [2])]},
        {'a': [('c', 1, [5, 6])],
         'b': [('e', 1, [2, 3])]},
    ]

    game = rsgame.SampleGame.from_payoff_format(players, strategies, prof_data)

    assert {1, 2} == game.num_samples(), \
        "didn't get both sample sizes"
    str(game)
    repr(game)


# Test that if game is too large an exception is thrown instead of going into
# an infinite loop
@tools.raises(Exception)
def too_big_for_dev_reps_test():
    randgames.symmetric_game(2000, 2)
