import itertools
import math
import os
import random
import warnings
from collections import abc

import numpy as np
import scipy.misc as spm
from nose import tools

from gameanalysis import gamegen
from gameanalysis import profile
from gameanalysis import rsgame
from gameanalysis import utils
from test import testutils

TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps

# The lambda: 0 means that the payoffs are all zero, since they don't matter
SMALL_GAMES = [
    gamegen.symmetric_game(2, 2),
    gamegen.symmetric_game(2, 5),
    gamegen.symmetric_game(5, 2),
    gamegen.symmetric_game(5, 5),
    gamegen.independent_game(2, 2),
    gamegen.independent_game(2, 5),
    gamegen.independent_game(5, 2),
    gamegen.independent_game(5, 5),
    gamegen.role_symmetric_game(2, [1, 2], [2, 1]),
    gamegen.role_symmetric_game(2, 2, 2),
    gamegen.role_symmetric_game(2, 2, 5),
    gamegen.role_symmetric_game(2, 5, 2),
    gamegen.role_symmetric_game(2, 5, 5),
    gamegen.symmetric_game(170, 2),  # approximate devreps
    gamegen.symmetric_game(180, 2),  # actual devreps
]


def generate_games(allow_big=False):
    """Returns a generator for game testing"""
    for game in SMALL_GAMES:
        yield game

    if allow_big and os.getenv('BIG_TESTS') == 'ON':  # Big Games
        yield gamegen.symmetric_game(1000, 2)
        yield gamegen.symmetric_game(5, 40)
        yield gamegen.symmetric_game(3, 160)
        yield gamegen.symmetric_game(50, 2)
        yield gamegen.symmetric_game(20, 5)
        yield gamegen.symmetric_game(90, 5)
        yield gamegen.role_symmetric_game(2, 2, 40)
        yield gamegen.symmetric_game(12, 12)


def exact_dev_reps(game):
    """Uses python ints to compute dev reps. Much slower"""
    counts = game.profiles(as_array=True)
    dev_reps = np.empty_like(counts, dtype=object)
    player_counts = list(game.players.values())
    fcount = [math.factorial(x) for x in player_counts]
    for dev_prof, count_prof in zip(dev_reps, counts):
        total = utils.prod(fc // utils.prod(math.factorial(x) for x in cs)
                           for fc, cs
                           in zip(fcount, game.role_split(count_prof)))
        for dev_role, counts_role, player_count \
                in zip(game.role_split(dev_prof), game.role_split(count_prof),
                       player_counts):
            for s, count in enumerate(counts_role):
                dev_role[s] = total * int(count) // player_count
    return dev_reps


@testutils.apply(zip(generate_games(True)))
def devreps_approx_test(game):
    approx = game._dev_reps
    exact = exact_dev_reps(game)
    diff = ((approx - exact) / (exact + TINY)).astype(float)
    assert np.allclose(diff, 0), \
        "dev reps were not close enough ({})".format(diff)


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
    game = gamegen.empty_role_symmetric_game(roles, players, strategies)
    assert game.players is not None, "players was None"
    assert game.strategies is not None, "strategies was None"
    assert game.size is not None, "size was None"

    # Test that all profiles returns the correct number of things
    num_all_profs = 0
    for _ in game.all_profiles():
        num_all_profs += 1
    assert num_all_profs == game.size, \
        "size of all profile generation is wrong"
    all_profs = game.all_profiles(as_array=True)
    assert len(all_profs) == game.size, \
        "size of all profile generation is wrong"

    # Assert as_ methods work
    assert isinstance(game.as_profile(all_profs[0]), profile.Profile), \
        "as_profile did not return a profile"
    assert isinstance(game.as_profile(game.as_profile(all_profs[0])),
                      profile.Profile), \
        "as_profile twice did not return a profile"

    mix = game.random_mixtures(1, as_array=True)[0]
    assert isinstance(game.as_mixture(mix), profile.Mixture), \
        "as_mixture did not return a mixture"
    assert isinstance(game.as_mixture(game.as_mixture(mix)),
                      profile.Mixture), \
        "as_mixture twice did not return a mixture"
    prof = game.as_profile(all_profs[0])
    assert isinstance(game.as_profile(prof, as_array=True), np.ndarray), \
        "as_array did not return an array"
    assert isinstance(game.as_profile(game.as_profile(prof, True), True),
                      np.ndarray), \
        "as_array twice did not return an array"

    # Assert that mixture calculations do the right thing
    # Uniform
    mix = game.uniform_mixture(as_array=True)
    assert np.allclose(game.role_reduce(mix), 1), \
        "uniform mixture wasn't a mixture"
    if max(len(s) for s in game.strategies.values()) == 1:
        assert (mix == 1).all(), "uniform mixtures wasn't uniform"
    else:
        diff = np.diff(mix)
        changes = game.astrategies[:-1].cumsum() - 1
        diff[changes] = 0
        assert np.allclose(diff, 0), \
            "uniform mixture wasn't uniform"
        one_strats = (game.astrategies.cumsum() - 1)[game.astrategies == 1]
        assert np.allclose(mix[one_strats], 1), \
            "uniform mixture wasn't uniform"

    mix = game.uniform_mixture(as_array=False)
    assert all(abs(sum(strats.values()) - 1) < EPS
               for strats in mix.values()), \
        "uniform mixture wasn't a mixture"
    assert all(max(abs(next(iter(strats.values())) - p)
                   for p in strats.values()) < EPS
               for strats in mix.values()), \
        "uniform mixture wasn't uniform"

    # Random
    mixes = game.random_mixtures(20, as_array=True)
    assert np.allclose(game.role_reduce(mixes, axis=1), 1), \
        "random mixtures weren't mixtures"

    mix = next(game.random_mixtures(as_array=False))
    assert all(abs(sum(strats.values()) - 1) < EPS
               for strats in mix.values()), \
        "random mixture wasn't a mixture"

    # Biased
    bias = 0.6
    mixes = game.biased_mixtures(bias, as_array=True)
    assert np.prod(game.astrategies[game.astrategies > 1]) == \
        mixes.shape[0], \
        "Didn't generate the proper number of biased mixtures"
    saw_bias = (mixes == bias).any(0)
    saw_all_biases = (game.role_reduce(saw_bias, ufunc=np.logical_and) |
                      (game.astrategies == 1)).all()
    assert saw_all_biases, "Didn't bias every strategy"
    assert np.allclose(game.role_reduce(mixes, axis=1), 1), \
        "biased mixtures weren't mixtures"

    mixes = game.biased_mixtures(bias, as_array=False)
    for mix in mixes:
        assert isinstance(mix, abc.Mapping)

    # Role Biased
    mixes = game.role_biased_mixtures(bias, as_array=True)
    assert game.astrategies[game.astrategies > 1].sum() == mixes.shape[0], \
        "Didn't generate the proper number of role biased mixtures"
    saw_bias = (mixes == bias).any(0)
    saw_all_biases = (game.role_reduce(saw_bias, ufunc=np.logical_and)
                      | (game.astrategies == 1)).all()
    assert saw_all_biases, "Didn't bias every strategy"
    assert np.allclose(game.role_reduce(mixes, axis=1), 1), \
        "biased mixtures weren't mixtures"

    mixes = game.role_biased_mixtures(bias, as_array=False)
    for mix in mixes:
        assert isinstance(mix, abc.Mapping)

    # Grid
    points = 3
    mixes = game.grid_mixtures(points, as_array=True)
    expected_num = utils.prod(spm.comb(len(s), points - 1,
                                       repetition=True, exact=True)
                              for s in game.strategies.values())
    assert expected_num == mixes.shape[0], \
        "didn't create the right number of grid mixtures"
    assert np.allclose(game.role_reduce(mixes, 1), 1), \
        "grid mixtures weren't mixtures"

    # Pure
    mixes = game.pure_mixtures(as_array=True)
    for mix in mixes:
        assert np.allclose(game.role_reduce(mix), 1), \
            "pure mixtures weren't mixtures"
        assert (game.role_reduce(mix == 1) == 1).all(), \
            "not all roles in pure mixture had an assignment"

    mixes = game.pure_mixtures(as_array=False)
    for mix in mixes:
        assert all(len(strats.values()) == 1 for strats in mix.values()), \
            "pure mixtures weren't pure"
        assert all(next(iter(strats.values())) == 1
                   for strats in mix.values()), \
            "pure mixtures weren't mixtures"

    # Test that various methods can be called
    assert game.to_json() is not None, "game json was None"
    assert str(game) is not None, "game str was None"
    assert repr(game) is not None, "game repr was None"


@testutils.apply(zip(generate_games()))
# Test that game functions work
def game_function_test(game):
    # Check that min payoffs are actually minimum
    min_payoffs = game.min_payoffs()
    assert all(all(all(min_payoffs[role] - EPS < p for p in pay.values())
                   for role, pay in payoff.items())
               for payoff in game.payoffs()), \
        "not all payoffs less than min payoffs"

    # Test profile methods
    prof_count = 0
    for prof in game.profiles():
        prof_count += 1
        role = next(iter(random.sample(list(prof), 1)))
        strategy = next(iter(random.sample(list(prof[role]), 1)))

        game.get_payoff(prof, role, strategy)  # Works
        game.get_payoffs(prof)  # Works
        game[prof]  # Works
        game.get_payoffs(prof, as_array=True)  # Works

        assert prof in game, "profile from game not in game"

    assert prof_count == game.size, \
        "game not complete or profiles missing data"

    # Test expected payoff
    mix = next(game.random_mixtures())
    assert not any(map(math.isnan, game.get_expected_payoff(mix).values())), \
        "some dict expected payoffs were nan"
    assert not np.isnan(game.get_expected_payoff(mix, as_array=True)).any(), \
        "some array expected payoffs were nan"
    assert not any(any(map(math.isnan, strats.values()))
                   for strats in game.deviation_payoffs(mix).values()), \
        "some dict expected values were nan"
    assert not np.isnan(game.deviation_payoffs(mix, as_array=True)).any(), \
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
    empty_game = gamegen.empty_role_symmetric_game(roles, players,
                                                   strategies)
    counts = np.empty([0, empty_game.num_role_strats], dtype=int)
    values = np.empty([0, empty_game.num_role_strats])
    game = rsgame.Game(empty_game.players, empty_game.strategies, counts,
                       values)

    # Check that min payoffs are actually minimum
    game.min_payoffs()
    assert len(list(game.payoffs())) == 0, \
        "returned payoffs"

    assert len(list(game.profiles())) == 0, \
        "returned payoffs"

    # Test expected payoff
    mix = next(game.random_mixtures())
    assert all(map(math.isnan, game.get_expected_payoff(mix).values())), \
        "not all expected payoffs were nan"
    assert all(all(map(math.isnan, strats.values()))
               for strats in game.deviation_payoffs(mix).values()), \
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
    game = gamegen.add_noise(game, samples)

    for prof in game.profiles():
        game.get_sample_payoffs(prof)  # Works
        game.get_sample_payoffs(prof, as_array=True)  # Works

    assert len(list(game.sample_profile_payoffs())) == game.size, \
        "sample payoffs not the right size"
    assert len(list(game.sample_profile_payoffs(as_array=True))) == game.size, \
        "sample payoffs not the right size"

    # Test that various methods can be called
    assert {samples} == game.num_samples()
    game.resample()
    game.resample(1)
    game.remean()

    assert game.to_json() is not None
    assert str(game) is not None
    assert repr(game) is not None


def sample_game_resample_test():
    game = gamegen.role_symmetric_game(3, [1, 2, 3], [4, 3, 2])
    game = gamegen.add_noise(game, 20)

    payoffs = game.payoffs(as_array=True).copy()
    min_values = game.min_payoffs(as_array=True).copy()

    game.resample()

    assert np.any(payoffs != game.payoffs(as_array=True)), \
        "resampling didn't change payoffs"
    assert np.any(min_values != game.min_payoffs(as_array=True)), \
        "resampling didn't change min values by role"

    game.remean()

    assert np.allclose(payoffs, game.payoffs(as_array=True)), \
        "remeaning didn't reset payoffs properly"
    assert np.allclose(min_values, game.min_payoffs(as_array=True)), \
        "remeaning didn't reset minimum payoffs properly"


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
    empty_game = gamegen.empty_role_symmetric_game(roles, players,
                                                   strategies)
    counts = np.empty([0, empty_game.num_role_strats], dtype=int)
    sample_values = []
    game = rsgame.SampleGame(empty_game.players, empty_game.strategies, counts,
                             sample_values)

    assert len(list(game.sample_profile_payoffs())) == 0, \
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
    game = gamegen.symmetric_game(2000, 2)
    game.dev_reps()


def deviation_payoffs_jacobian_test():
    game = gamegen.rock_paper_scissors()
    eqm = np.array([1/3] * 3)
    dp, dpj = game.deviation_payoffs(eqm, jacobian=True)
    assert np.allclose(dp, 0), \
        "expected value at eq should be 0, {}".format(dp)
    expected_jac = np.array([[0., 1., -1.],
                             [-1., 0., 1.],
                             [1., -1., 0.]])
    assert np.allclose(dpj, expected_jac), \
        "jacobian was not expected {} instead of {}".format(dpj, expected_jac)


def trim_mixture_array_support_test():
    game = gamegen.empty_role_symmetric_game(1, 2, 3)
    mix = np.array([0.7, 0.3, 0])
    not_trimmed = game.trim_mixture_array_support(mix, 0.1)
    assert np.allclose(mix, not_trimmed), \
        "array got trimmed when it shouldn't"
    trimmed = game.trim_mixture_array_support(mix, 0.4)
    assert np.allclose([1, 0, 0], trimmed), \
        "array didn't get trimmed when it should"


def as_profile_test():
    game = gamegen.empty_role_symmetric_game(1, 2, 3)
    profile = next(game.all_profiles(as_array=False))
    aprofile = game.all_profiles(as_array=True)[0]

    assert profile == game.as_profile(profile, as_array=None)
    assert profile == game.as_profile(profile, as_array=False)
    assert np.all(aprofile == game.as_profile(profile, as_array=True))

    assert np.all(aprofile == game.as_profile(aprofile, as_array=None))
    assert np.all(aprofile == game.as_profile(aprofile, as_array=True))
    assert profile == game.as_profile(aprofile, as_array=False)


def as_mixture_test():
    game = gamegen.empty_role_symmetric_game(1, 2, 3)
    mixture = game.uniform_mixture(as_array=False)
    amixture = game.uniform_mixture(as_array=True)

    assert mixture == game.as_mixture(mixture, as_array=None)
    assert mixture == game.as_mixture(mixture, as_array=False)
    assert np.all(amixture == game.as_mixture(mixture, as_array=True))

    assert np.all(amixture == game.as_mixture(amixture, as_array=None))
    assert np.all(amixture == game.as_mixture(amixture, as_array=True))
    assert mixture == game.as_mixture(amixture, as_array=False)


@tools.raises(ValueError)
def from_game_failure_test():
    rsgame.Game.from_game(None)
