import itertools
import math

import numpy as np
import numpy.random as rand
import pytest
import scipy.misc as spm
import scipy.special as sps

from gameanalysis import gamegen
from gameanalysis import reduction
from gameanalysis import rsgame
from gameanalysis import utils
from test import testutils

TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps


def exact_dev_reps(game):
    """Uses python ints to compute dev reps. Much slower"""
    counts = game.profiles
    dev_reps = np.empty_like(counts, float)
    fcount = [math.factorial(x) for x in game.num_players]
    for dev_prof, count_prof in zip(dev_reps, counts):
        total = utils.prod(fc // utils.prod(math.factorial(x) for x in cs)
                           for fc, cs
                           in zip(fcount, game.role_split(count_prof)))
        for dev_role, counts_role, player_count \
                in zip(game.role_split(dev_prof), game.role_split(count_prof),
                       game.num_players):
            for s, count in enumerate(counts_role):
                rep = total * int(count) // player_count
                dev_role[s] = math.log(rep) if rep > 0 else -np.inf
    return dev_reps


@testutils.apply(testutils.game_sizes(allow_big=True))
def test_devreps_approx(players, strategies):
    base = rsgame.BaseGame(players, strategies)
    profiles = base.all_profiles()
    payoffs = np.zeros(profiles.shape, float)
    game = rsgame.Game(base, profiles, payoffs)
    approx = game._dev_reps
    exact = exact_dev_reps(game)
    # The equals checks for -inf == -inf
    assert np.all(np.isclose(approx, exact) | (exact == approx))


# Test that all functions work on an BaseGame
@testutils.apply(testutils.game_sizes())
def test_base_game_function(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    assert game.num_players is not None, "num players was None"
    assert game.num_strategies is not None, "num strategies was None"

    # Test copy constructor
    game2 = rsgame.BaseGame(game)
    assert np.all(game.num_players == game2.num_players)
    assert np.all(game.num_strategies == game2.num_strategies)

    # Test role index
    expected = game.role_repeat(np.arange(game.num_roles))
    actual = game.role_index[np.arange(game.num_role_strats)]
    assert np.all(expected == actual)

    # Test that all profiles returns the correct number of things
    all_profs = game.all_profiles()
    assert all_profs.shape[0] == game.num_all_profiles, \
        "size of all profile generation is wrong"

    # Assert that mixture calculations do the right thing
    # Uniform
    mix = game.uniform_mixture()
    assert np.allclose(game.role_reduce(mix), 1), \
        "uniform mixture wasn't a mixture"
    if game.num_strategies.max() == 1:
        assert (mix == 1).all(), "uniform mixtures wasn't uniform"
    else:
        diff = np.diff(mix)
        changes = game.num_strategies[:-1].cumsum() - 1
        diff[changes] = 0
        assert np.allclose(diff, 0), \
            "uniform mixture wasn't uniform"
        one_strats = (game.num_strategies.cumsum() - 1)[game.num_strategies
                                                        == 1]
        assert np.allclose(mix[one_strats], 1), \
            "uniform mixture wasn't uniform"

    # Random
    mixes = game.random_mixtures(20)
    assert np.allclose(game.role_reduce(mixes, axis=1), 1), \
        "random mixtures weren't mixtures"

    # Biased
    bias = 0.6
    mixes = game.biased_mixtures(bias)
    assert np.prod(game.num_strategies[game.num_strategies > 1]) == \
        mixes.shape[0], \
        "Didn't generate the proper number of biased mixtures"
    saw_bias = (mixes == bias).any(0)
    saw_all_biases = (game.role_reduce(saw_bias, ufunc=np.logical_and) |
                      (game.num_strategies == 1)).all()
    assert saw_all_biases, "Didn't bias every strategy"
    assert np.allclose(game.role_reduce(mixes, axis=1), 1), \
        "biased mixtures weren't mixtures"

    # Role Biased
    mixes = game.role_biased_mixtures(bias)
    assert game.num_strategies[game.num_strategies > 1].sum() == mixes.shape[0], \
        "Didn't generate the proper number of role biased mixtures"
    saw_bias = (mixes == bias).any(0)
    saw_all_biases = (game.role_reduce(saw_bias, ufunc=np.logical_and)
                      | (game.num_strategies == 1)).all()
    assert saw_all_biases, "Didn't bias every strategy"
    assert np.allclose(game.role_reduce(mixes, axis=1), 1), \
        "biased mixtures weren't mixtures"

    # Grid
    points = 3
    mixes = game.grid_mixtures(points)
    expected_num = utils.prod(spm.comb(s, points - 1, repetition=True,
                                       exact=True)
                              for s in game.num_strategies)
    assert expected_num == mixes.shape[0], \
        "didn't create the right number of grid mixtures"
    assert np.allclose(game.role_reduce(mixes, 1), 1), \
        "grid mixtures weren't mixtures"

    # Pure
    mixes = game.pure_mixtures()
    assert np.allclose(game.role_reduce(mixes), 1), \
        "pure mixtures weren't mixtures"
    assert np.all(game.role_reduce(np.isclose(mixes, 1)) == 1), \
        "not all roles in pure mixture had an assignment"

    profs = game.pure_profiles()
    assert np.all(game.role_reduce(profs > 0) == 1), \
        "pure profiles weren't pure"

    # Test that various methods can be called
    assert repr(game) is not None, "game repr was None"


@testutils.apply(testutils.game_sizes())
def test_max_prob_prof(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    profiles = game.all_profiles()
    log_prob = (np.sum(sps.gammaln(game.num_players + 1)) -
                np.sum(sps.gammaln(profiles + 1), 1))
    for mix in game.random_mixtures(100):
        probs = np.sum(np.log(mix + TINY) * profiles, 1) + log_prob
        mask = np.max(probs) - EPS < probs
        max_prob_profs = profiles[mask]
        actual = game.max_prob_prof(mix)
        assert np.all(np.in1d(game.profile_id(actual),
                              game.profile_id(max_prob_profs)))


def test_base_game_invalid_constructor():
    with pytest.raises(ValueError):
        rsgame.BaseGame(None, None, None)


def test_base_game_min_payoffs():
    with pytest.raises(NotImplementedError):
        rsgame.BaseGame(1, 1).min_payoffs()


def test_base_game_max_payoffs():
    with pytest.raises(NotImplementedError):
        rsgame.BaseGame(1, 1).max_payoffs()


def test_base_game_deviation_payoffs():
    base = rsgame.BaseGame(1, 1)
    mix = base.uniform_mixture()
    with pytest.raises(NotImplementedError):
        base.deviation_payoffs(mix)


def test_verify_mixture_profile():
    game = rsgame.BaseGame([2, 3], [3, 2])
    assert game.verify_profile([1, 1, 0, 3, 0])
    assert not game.verify_profile([1, 0, 0, 3, 0])
    assert game.verify_mixture([0.2, 0.3, 0.5, 0.6, 0.4])
    assert not game.verify_mixture([0.2, 0.3, 0.4, 0.5, 0.6])

    mix = game.uniform_mixture()
    random_profs = game.random_profiles(mix, 20)
    assert np.all(game.verify_profile(random_profs))

    random_profs = game.random_deviator_profiles(mix, 20)
    assert np.all(game.verify_profile(random_profs))


@testutils.apply(testutils.game_sizes())
def test_simplex_project(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    for non_mixture in rand.uniform(-1, 1, (100, game.num_role_strats)):
        new_mix = game.simplex_project(non_mixture)
        assert game.verify_mixture(new_mix), \
            "simplex project did not create a valid mixture"


def test_symmetric():
    assert rsgame.BaseGame(3, 4).is_symmetric()
    assert not rsgame.BaseGame([2, 2], 3).is_symmetric()


# Test that game functions work
@testutils.apply(testutils.game_sizes())
def test_game_function(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)

    # Test copy
    game2 = rsgame.Game(game)
    assert not np.may_share_memory(game.profiles, game2.profiles)
    assert not np.may_share_memory(game.payoffs, game2.payoffs)
    assert np.all(game.profiles == game2.profiles)
    assert np.all(game.payoffs == game2.payoffs)

    game3 = rsgame.Game(rsgame.BaseGame(game))
    assert game3.is_empty()

    mask = game.profiles > 0

    # Check that min payoffs are actually minimum
    min_payoffs = game.role_repeat(game.min_payoffs())
    assert np.all((game.payoffs >= min_payoffs)[mask]), \
        "not all payoffs less than min payoffs"
    max_payoffs = game.role_repeat(game.max_payoffs())
    assert np.all((game.payoffs <= max_payoffs)[mask]), \
        "not all payoffs greater than max payoffs"

    # Test profile methods
    for prof in game.profiles:
        game.get_payoffs(prof)  # Works
        assert prof in game, "profile from game not in game"

    # Test expected payoff
    mix = game.random_mixtures()[0]
    assert not np.isnan(game.get_expected_payoffs(mix)).any(), \
        "some array expected payoffs were nan"
    assert not np.isnan(game.deviation_payoffs(mix)).any(), \
        "some array expected values were nan"

    # Max social welfare
    assert game.get_max_social_welfare() is not None
    for role_index in range(game.num_roles):
        assert game.get_max_social_welfare(role_index) is not None

    # Test that various methods can be called
    assert repr(game) is not None


# Test that a Game with no data can still be created
@testutils.apply(testutils.game_sizes())
def test_empty_full_game(players, strategies):
    game = rsgame.Game(players, strategies)

    # Check that min payoffs can be called
    assert np.isnan(game.min_payoffs()).all()
    assert np.isnan(game.max_payoffs()).all()
    assert game.payoffs.shape[0] == 0
    assert game.profiles.shape[0] == 0
    assert game.is_empty()
    assert game.is_constant_sum()

    # Test expected payoff
    mix = game.random_mixtures()[0]
    assert np.isnan(game.get_expected_payoffs(mix)).all(), \
        "not all expected payoffs were nan"
    assert np.isnan(game.deviation_payoffs(mix)).all(), \
        "not all expected values were nan"
    pays, jac = game.deviation_payoffs(mix, jacobian=True)
    assert np.isnan(game.deviation_payoffs(mix, jacobian=True)[1]).all(), \
        "not all expected values were nan"

    # Max social welfare
    assert (np.nan, None) == game.get_max_social_welfare(), \
        "Didn't return an empty welfare"

    # Default payoff
    for prof in game.all_profiles():
        assert np.isnan(game.get_payoffs(prof, np.nan)[prof > 0]).all()

    # Test that various methods can be called
    assert repr(game) is not None


def test_deviation_mixture_support():
    base = rsgame.BaseGame([2, 2], 3)
    profiles1 = [
        [2, 0, 0, 2, 0, 0],
        [1, 1, 0, 2, 0, 0],
        [0, 2, 0, 2, 0, 0],
    ]
    payoffs1 = [
        [1, 0, 0, 2, 0, 0],
        [3, 4, 0, 5, 0, 0],
        [0, 6, 0, 7, 0, 0],
    ]
    profiles2 = [
        [2, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 2, 0, 1, 1, 0],
    ]
    payoffs2 = [
        [8, 0, 0, 9, 10, 0],
        [11, 12, 0, 13, 14, 0],
        [0, 15, 0, 16, 17, 0],
    ]
    game1 = rsgame.Game(base, profiles1, payoffs1)
    game2 = rsgame.Game(base, profiles2, payoffs2)
    game3 = rsgame.Game(base, profiles1 + profiles2, payoffs1 + payoffs2)
    mix1 = [0.5, 0.5, 0, 0.3, 0.7, 0]
    mix2 = [0.5, 0.5, 0, 1, 0, 0]

    dev_payoffs = game1.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game1.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs, [2, 5, np.nan, 4.75, np.nan, np.nan],
                       equal_nan=True)
    dev_payoffs = game2.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game2.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs,
                       [np.nan, np.nan, np.nan, np.nan, 13.75, np.nan],
                       equal_nan=True)
    dev_payoffs = game3.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game3.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs, [2, 5, np.nan, 4.75, 13.75, np.nan],
                       equal_nan=True)


def test_game_invalid_constructor():
    with pytest.raises(ValueError):
        rsgame.Game(None, None, None, None, None)


def test_constant_sum():
    game = gamegen.two_player_zero_sum_game(2)
    assert game.is_constant_sum()
    payoffs = game.payoffs.copy()
    payoffs[game.profiles > 0] += 1
    game2 = rsgame.Game(game, game.profiles, payoffs)
    assert game2.is_constant_sum()
    profiles = [
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
    ]
    payoffs = [
        [1, 0, 2, 0],
        [3, 0, 0, 4],
        [0, 5, 6, 0],
        [0, 7, 0, 8],
    ]
    game3 = rsgame.Game(game, profiles, payoffs)
    assert not game3.is_constant_sum()


# Test that sample game functions work
@testutils.apply(zip(testutils.game_sizes(), itertools.cycle([1, 2, 5, 10])))
def test_sample_game_function(game_size, samples):
    base = gamegen.role_symmetric_game(*game_size)
    game = gamegen.add_noise(base, 1, samples)

    # Test constructors
    game2 = rsgame.SampleGame(game)
    assert not np.may_share_memory(game.profiles, game2.profiles)
    assert not np.may_share_memory(game.payoffs, game2.payoffs)
    assert not any(np.may_share_memory(sp, sp2) for sp, sp2
                   in zip(game.sample_payoffs, game2.sample_payoffs))
    assert np.all(game.profiles == game2.profiles)
    assert np.all(game.payoffs == game2.payoffs)
    assert all(np.all(sp == sp2) for sp, sp2
               in zip(game.sample_payoffs, game2.sample_payoffs))

    game3 = rsgame.SampleGame(base)
    assert not np.may_share_memory(base.profiles, game3.profiles)
    assert not np.may_share_memory(base.payoffs, game3.payoffs)
    assert np.all(base.profiles == game3.profiles)
    assert np.all(base.payoffs == game3.payoffs)
    assert np.all(game3.num_samples == 1)

    game4 = rsgame.SampleGame(rsgame.BaseGame(*game_size))
    assert game4.is_empty()

    game5 = rsgame.SampleGame(*game_size)
    assert game5.is_empty()

    game5 = rsgame.SampleGame(game.num_players, game.num_strategies,
                              game.profiles, game.sample_payoffs)

    # Test that various methods can be called
    assert (np.all(1 <= game.num_samples) and
            np.all(game.num_samples <= samples))
    game.resample()
    game.resample(1)
    game.remean()

    assert repr(game) is not None


def test_sample_game_resample():
    game = gamegen.role_symmetric_game([1, 2, 3], [4, 3, 2])
    game = gamegen.add_noise(game, 1, 20)

    payoffs = game.payoffs.copy()
    min_values = game.min_payoffs().copy()
    max_values = game.max_payoffs().copy()

    game.resample()

    # This isn't guaranteed to be true, but they're highly unlikely
    assert np.any(payoffs != game.payoffs), \
        "resampling didn't change payoffs"

    game.remean()

    assert np.allclose(payoffs, game.payoffs), \
        "remeaning didn't reset payoffs properly"
    assert np.allclose(min_values, game.min_payoffs()), \
        "remeaning didn't reset minimum payoffs properly"
    assert np.allclose(max_values, game.max_payoffs()), \
        "remeaning didn't reset minimum payoffs properly"


# Test that a Game with no data can still be created
@testutils.apply(testutils.game_sizes())
def test_empty_sample_game(players, strategies):
    base = rsgame.BaseGame(players, strategies)
    profiles = np.empty([0, base.num_role_strats], dtype=int)
    game = rsgame.SampleGame(base, profiles, [])

    # Test that various methods can be called
    assert game.num_samples is not None
    game.remean()
    game.resample()
    game.resample(1)

    repr(game)


def test_sample_game_invalid_constructor():
    with pytest.raises(ValueError):
        rsgame.SampleGame(None, None, None, None, None)


# Test that sample game from matrix creates a game
@testutils.apply([
    (1, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 1, 2),
    (2, 2, 2),
    (3, 2, 4),
], repeat=20)
def test_sample_game_from_matrix(players, strategies, samples):
    matrix = np.random.random([strategies] * players + [players, samples])
    game = rsgame.SampleGame(matrix)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == players, \
        "didn't generate correct number of players"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(game.num_samples == [samples]), \
        "profiles didn't have correct number of samples"


# Test sample game with different number of samples
def test_different_samples():
    base = rsgame.BaseGame(1, [1, 2])
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6], [0, 0], [2, 3]],
        ],
    ]

    game = rsgame.SampleGame(base, profiles, payoffs)

    assert np.all([1, 2] == game.num_samples), \
        "didn't get both sample sizes"
    assert repr(game) is not None


def test_deviation_payoffs_jacobian():
    game = gamegen.rock_paper_scissors()
    eqm = np.array([1/3] * 3)
    dp, dpj = game.deviation_payoffs(eqm, jacobian=True)
    assert np.allclose(dp, 0)
    expected_jac = np.array([[0., -1., 1.],
                             [1., 0., -1.],
                             [-1., 1., 0.]])
    assert np.allclose(dpj, expected_jac)


def test_trim_mixture_support():
    game = rsgame.BaseGame(2, 3)
    mix = np.array([0.7, 0.3, 0])
    not_trimmed = game.trim_mixture_support(mix, 0.1)
    assert np.allclose(mix, not_trimmed), \
        "array got trimmed when it shouldn't"
    trimmed = game.trim_mixture_support(mix, 0.4)
    assert np.allclose([1, 0, 0], trimmed), \
        "array didn't get trimmed when it should {}".format(
            trimmed)


@testutils.apply(testutils.game_sizes())
def test_profile_count(players, strategies):
    game = rsgame.BaseGame(players, strategies)

    num_profiles = game.all_profiles().shape[0]
    assert num_profiles == game.num_all_profiles, \
        "num_all_profiles didn't return the correct number"

    num_payoffs = np.sum(game.all_profiles() > 0)
    assert num_payoffs == game.num_all_payoffs, \
        "num_all_payoffs didn't return the correct number"

    red = reduction.DeviationPreserving(
        game.num_strategies, game.num_players ** 2, game.num_players)

    num_dpr_profiles = red.expand_profiles(game.all_profiles()).shape[0]
    assert num_dpr_profiles == game.num_all_dpr_profiles, \
        ("num_all_dpr_profiles did not return the correct number {} "
         "instead of {}").format(game.num_all_dpr_profiles,
                                 num_dpr_profiles)
