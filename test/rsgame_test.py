import itertools
import math

import numpy as np
import numpy.random as rand
import pytest
import scipy.misc as spm
import scipy.special as sps

from gameanalysis import gamegen
from gameanalysis import gameio
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


@pytest.mark.parametrize('players,strategies', testutils.big_games)
def test_devreps_approx(players, strategies):
    base = rsgame.basegame(players, strategies)
    profiles = base.all_profiles()
    payoffs = np.zeros(profiles.shape, float)
    game = rsgame.game_copy(base, profiles, payoffs)
    approx = game._dev_reps
    exact = exact_dev_reps(game)
    # The equals checks for -inf == -inf
    assert np.all(np.isclose(approx, exact) | (exact == approx))


# Test that all functions work on an BaseGame
@pytest.mark.parametrize('players,strategies', testutils.games)
def test_basegame_function(players, strategies):
    game = rsgame.basegame(players, strategies)
    assert game.num_players is not None, "num players was None"
    assert game.num_strategies is not None, "num strategies was None"

    # Test copy constructor
    game2 = rsgame.basegame_copy(game)
    assert game == game2
    assert np.all(game.num_players == game2.num_players)
    assert np.all(game.num_strategies == game2.num_strategies)

    # Test role indices
    expected = game.role_repeat(np.arange(game.num_roles))
    actual = game.role_indices[np.arange(game.num_role_strats)]
    assert np.all(expected == actual)

    # Test that all profiles returns the correct number of things
    all_profs = game.all_profiles()
    assert all_profs.shape[0] == game.num_all_profiles, \
        "size of all profile generation is wrong"

    # Test that all subgames returns the correct number
    all_subs = game.all_subgames()
    assert all_subs.shape[0] == game.num_all_subgames
    assert game.role_reduce(all_subs, ufunc=np.bitwise_or).all()
    uniques = np.unique(utils.axis_to_elem(all_subs))
    assert uniques.size == all_subs.shape[0]

    pure_subs = game.pure_subgames()
    assert pure_subs.shape[0] == game.num_pure_subgames
    assert game.role_reduce(pure_subs, ufunc=np.bitwise_or).all()
    uniques = np.unique(utils.axis_to_elem(pure_subs))
    assert uniques.size == pure_subs.shape[0]

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

    # Random Subgames
    assert game.verify_subgame(game.random_subgames())
    assert game.verify_subgame(game.random_subgames(20)).all()

    # Random
    assert np.allclose(game.role_reduce(game.random_mixtures()), 1)
    mixes = game.random_mixtures(20)
    assert np.allclose(game.role_reduce(mixes, axis=1), 1), \
        "random mixtures weren't mixtures"

    # Random Sparse
    assert np.allclose(game.role_reduce(game.random_sparse_mixtures()), 1)
    mixes = game.random_sparse_mixtures(20)
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
    assert (game.num_strategies[game.num_strategies > 1].sum()
            == mixes.shape[0]), \
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
    assert repr(game) is not None
    assert hash(game) is not None


@pytest.mark.parametrize('players,strategies', testutils.games)
def test_max_prob_prof(players, strategies):
    game = rsgame.basegame(players, strategies)
    profiles = game.all_profiles()
    log_prob = (np.sum(sps.gammaln(game.num_players + 1)) -
                np.sum(sps.gammaln(profiles + 1), 1))
    for mix in game.random_mixtures(100):
        probs = np.sum(np.log(mix + TINY) * profiles, 1) + log_prob
        mask = np.max(probs) - EPS < probs
        max_prob_profs = profiles[mask]
        actual = game.max_prob_prof(mix)
        assert np.all(np.in1d(utils.axis_to_elem(actual),
                              utils.axis_to_elem(max_prob_profs)))


def test_basegame_min_payoffs():
    with pytest.raises(NotImplementedError):
        rsgame.basegame(1, 1).min_payoffs()


def test_basegame_max_payoffs():
    with pytest.raises(NotImplementedError):
        rsgame.basegame(1, 1).max_payoffs()


def test_basegame_deviation_payoffs():
    base = rsgame.basegame(1, 1)
    mix = base.uniform_mixture()
    with pytest.raises(NotImplementedError):
        base.deviation_payoffs(mix)


def test_verify_mixture_profile():
    game = rsgame.basegame([2, 3], [3, 2])
    assert game.verify_profile([1, 1, 0, 3, 0])
    assert not game.verify_profile([1, 0, 0, 3, 0])
    assert game.verify_mixture([0.2, 0.3, 0.5, 0.6, 0.4])
    assert not game.verify_mixture([0.2, 0.3, 0.4, 0.5, 0.6])

    assert game.verify_profile(game.random_profiles())
    assert game.verify_profile(game.random_profiles(20)).all()

    mix = game.uniform_mixture()
    assert game.verify_profile(game.random_deviator_profiles(mix)).all()
    assert game.verify_profile(game.random_deviator_profiles(mix, 20)).all()


@pytest.mark.parametrize('players,strategies', testutils.games)
def test_simplex_project(players, strategies):
    game = rsgame.basegame(players, strategies)
    for non_mixture in rand.uniform(-1, 1, (100, game.num_role_strats)):
        new_mix = game.simplex_project(non_mixture)
        assert game.verify_mixture(new_mix), \
            "simplex project did not create a valid mixture"


def test_symmetric():
    assert rsgame.basegame(3, 4).is_symmetric()
    assert not rsgame.basegame([2, 2], 3).is_symmetric()


# Test that game functions work
@pytest.mark.parametrize('players,strategies', testutils.games)
def test_game_function(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)

    # Test copy
    game2 = rsgame.game_copy(game)
    assert not np.may_share_memory(game.payoffs, game2.payoffs)
    assert np.all(game.profiles == game2.profiles)
    assert np.all(game.payoffs == game2.payoffs)

    game3 = rsgame.game_copy(rsgame.basegame_copy(game))
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
    mix = game.random_mixtures()

    dev1 = game.deviation_payoffs(mix)
    dev2, dev_jac = game.deviation_payoffs(mix, jacobian=True)
    assert not np.isnan(dev1).any()
    assert not np.isnan(dev_jac).any()
    assert np.allclose(dev1, dev2)

    pay1 = game.get_expected_payoffs(mix)
    pay2 = game.get_expected_payoffs(mix, deviations=dev1)
    pay3, jac1 = game.get_expected_payoffs(mix, jacobian=True)
    pay4, jac2 = game.get_expected_payoffs(
        mix, deviations=(dev1, dev_jac), jacobian=True)
    assert not np.isnan(pay1).any()
    assert (np.allclose(pay1, pay2) and np.allclose(pay1, pay3) and
            np.allclose(pay1, pay4))
    assert not np.isnan(jac1).any()
    assert np.allclose(jac1, jac2)

    # Test that various methods can be called
    assert repr(game) is not None
    assert hash(game) is not None


# Test that a Game with no data can still be created
@pytest.mark.parametrize('players,strategies', testutils.games)
def test_empty_full_game(players, strategies):
    game = rsgame.game(players, strategies)

    # Check that min payoffs can be called
    assert np.isnan(game.min_payoffs()).all()
    assert np.isnan(game.max_payoffs()).all()
    assert game.payoffs.shape[0] == 0
    assert game.profiles.shape[0] == 0
    assert game.is_empty()
    assert game.is_constant_sum()

    # Test expected payoff
    mix = game.random_mixtures()
    assert np.isnan(game.get_expected_payoffs(mix)).all(), \
        "not all expected payoffs were nan"
    assert np.isnan(game.deviation_payoffs(mix)).all(), \
        "not all expected values were nan"
    pays, jac = game.deviation_payoffs(mix, jacobian=True)
    assert np.isnan(game.deviation_payoffs(mix, jacobian=True)[1]).all(), \
        "not all expected values were nan"

    # Default payoff
    for prof in game.all_profiles():
        pay = game.get_payoffs(prof)
        assert np.isnan(pay[prof > 0]).all()
        assert np.all(pay[prof == 0] == 0)

    # Test that various methods can be called
    assert repr(game) is not None


def test_deviation_mixture_support():
    base = rsgame.basegame([2, 2], 3)
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
    game1 = rsgame.game_copy(base, profiles1, payoffs1)
    game2 = rsgame.game_copy(base, profiles2, payoffs2)
    game3 = rsgame.game_copy(base, profiles1 + profiles2, payoffs1 + payoffs2)
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


def test_constant_sum():
    game = gamegen.two_player_zero_sum_game(2)
    assert game.is_constant_sum()
    payoffs = game.payoffs.copy()
    payoffs[game.profiles > 0] += 1
    game2 = rsgame.game_copy(game, game.profiles, payoffs)
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
    game3 = rsgame.game_copy(game, profiles, payoffs)
    assert not game3.is_constant_sum()


# Test that sample game functions work
@pytest.mark.parametrize('game_size,samples',
                         zip(testutils.games, itertools.cycle([1, 2, 5, 10])))
def test_samplegame_function(game_size, samples):
    base = gamegen.role_symmetric_game(*game_size)
    game = gamegen.add_noise(base, 1, samples)

    # Test constructors
    game2 = rsgame.samplegame_copy(game)
    assert not np.may_share_memory(game.payoffs, game2.payoffs)
    assert not any(np.may_share_memory(sp, sp2) for sp, sp2
                   in zip(game.sample_payoffs, game2.sample_payoffs))
    assert np.all(game.profiles == game2.profiles)
    assert np.all(game.payoffs == game2.payoffs)
    assert all(np.all(sp == sp2) for sp, sp2
               in zip(game.sample_payoffs, game2.sample_payoffs))

    game3 = rsgame.samplegame_copy(base)
    assert not np.may_share_memory(base.payoffs, game3.payoffs)
    assert np.all(base.profiles == game3.profiles)
    assert np.all(base.payoffs == game3.payoffs)
    assert np.all(game3.num_samples == 1)

    game4 = rsgame.samplegame_copy(rsgame.basegame(*game_size))
    assert game4.is_empty()

    game5 = rsgame.samplegame(*game_size)
    assert game5.is_empty()

    game5 = rsgame.samplegame(game.num_players, game.num_strategies,
                              game.profiles, game.sample_payoffs)

    # Test that various methods can be called
    assert (np.all(1 <= game.num_samples) and
            np.all(game.num_samples <= samples))
    game.resample()
    game.resample(1)
    game.remean()

    assert repr(game) is not None
    assert hash(game) is not None


def test_samplegame_resample():
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
@pytest.mark.parametrize('players,strategies', testutils.games)
def test_empty_samplegame(players, strategies):
    game = rsgame.samplegame(players, strategies)

    # Test that various methods can be called
    assert game.num_samples is not None
    game.remean()
    game.resample()
    game.resample(1)

    assert repr(game) is not None

    assert not game.get_sample_payoffs(game.random_profiles()).size


# Test that sample game from matrix creates a game
@pytest.mark.parametrize('players,strategies,samples', [
    (1, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 1, 2),
    (2, 2, 2),
    (3, 2, 4),
] * 20)
def test_samplegame_from_matrix(players, strategies, samples):
    matrix = np.random.random([strategies] * players + [players, samples])
    game = rsgame.samplegame_matrix(matrix)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == players, \
        "didn't generate correct number of players"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(game.num_samples == [samples]), \
        "profiles didn't have correct number of samples"


# Test sample game with different number of samples
def test_different_samples():
    base = rsgame.basegame(1, [1, 2])
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

    game = rsgame.samplegame_copy(base, profiles, payoffs)

    assert np.all([1, 2] == game.num_samples), \
        "didn't get both sample sizes"
    assert repr(game) is not None


def test_deviation_payoffs_jacobian():
    game = gamegen.rock_paper_scissors()
    eqm = np.array([1 / 3] * 3)
    dp, dpj = game.deviation_payoffs(eqm, jacobian=True)
    assert np.allclose(dp, 0)
    expected_jac = np.array([[0., -1., 1.],
                             [1., 0., -1.],
                             [-1., 1., 0.]])
    assert np.allclose(dpj, expected_jac)


def test_trim_mixture_support():
    game = rsgame.basegame(2, 3)
    mix = np.array([0.7, 0.3, 0])
    not_trimmed = game.trim_mixture_support(mix, 0.1)
    assert np.allclose(mix, not_trimmed), \
        "array got trimmed when it shouldn't"
    trimmed = game.trim_mixture_support(mix, 0.4)
    assert np.allclose([1, 0, 0], trimmed), \
        "array didn't get trimmed when it should {}".format(
            trimmed)


@pytest.mark.parametrize('players,strategies', testutils.games)
def test_profile_count(players, strategies):
    game = rsgame.basegame(players, strategies)

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


def test_big_game_functions():
    """Test that everything works when game_size > int max"""
    base = rsgame.basegame([100, 100], [30, 30])
    game = gamegen.add_profiles(base, 1000)
    assert game.num_all_profiles > np.iinfo(int).max
    assert game.num_all_dpr_profiles > np.iinfo(int).max
    assert np.all(game.profile_id(game.profiles) >= 0)


def test_nan_mask_for_dev_payoffs():
    profiles = [[3, 0, 0, 0],
                [2, 1, 0, 0],
                [2, 0, 1, 0]]
    payoffs = [[1, 0, 0, 0],
               [np.nan, 2, 0, 0],
               [5, 0, np.nan, 0]]
    game = rsgame.game([3], [4], profiles, payoffs)
    devs = game.deviation_payoffs([1, 0, 0, 0])
    assert np.allclose(devs, [1, 2, np.nan, np.nan], equal_nan=True)

    devs = game.deviation_payoffs([1, 0, 0, 0], assume_complete=True)
    assert np.allclose(devs, [1, 2, np.nan, 0], equal_nan=True)


def test_nan_payoffs_for_dev_payoffs():
    profiles = [[3, 0, 3, 0],
                [2, 1, 3, 0],
                [3, 0, 2, 1]]
    payoffs = [[1, 0, 2, 0],
               [np.nan, 3, np.nan, 0],
               [np.nan, 0, np.nan, 4]]
    game = rsgame.game([3, 3], [2, 2], profiles, payoffs)
    devs = game.deviation_payoffs([1, 0, 1, 0])
    assert np.allclose(devs, [1, 3, 2, 4])


def test_expected_payoffs_jac():
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[1, 0],
               [3, 3],
               [0, 1]]
    game = rsgame.game(2, 2, profiles, payoffs)
    ep, ep_jac = game.get_expected_payoffs([.5, .5], jacobian=True)
    ep_jac -= ep_jac.sum() / 2  # project on simplex
    assert np.allclose(ep, 2)
    assert np.allclose(ep_jac, 0), \
        "maximum surplus should have 0 jacobian"


@pytest.mark.parametrize('p', [2, 5, 10, 100])
def test_deviation_nans(p):
    profiles = [[p,     0, 0, 0, 1],
                [p - 1, 1, 0, 0, 1],
                [p - 1, 0, 1, 0, 1],
                [p - 1, 0, 0, 1, 1]]
    payoffs = [[1,      0, 0, 0, 2],
               [np.nan, 3, 0, 0, np.nan],
               [np.nan, 0, 4, 0, np.nan],
               [np.nan, 0, 0, 5, np.nan]]
    game = rsgame.game([p, 1], [4, 1], profiles, payoffs)
    mix = np.array([1, 0, 0, 0, 1])
    pays = game.deviation_payoffs(mix)
    assert not np.isnan(pays).any()


@pytest.mark.parametrize('p', [2, 5, 10, 100])
@pytest.mark.parametrize('q', [2, 5, 10, 100])
def test_deviation_nans_2(p, q):
    profiles = [[p,     0, 0, 0, q,     0],
                [p - 1, 1, 0, 0, q,     0],
                [p - 1, 0, 1, 0, q,     0],
                [p - 1, 0, 0, 1, q,     0],
                [p,     0, 0, 0, q - 1, 1]]
    payoffs = [[1,      0, 0, 0, 2,      0],
               [np.nan, 3, 0, 0, np.nan, 0],
               [np.nan, 0, 4, 0, np.nan, 0],
               [np.nan, 0, 0, 5, np.nan, 0],
               [6,      0, 0, 0, np.nan, 7]]
    game = rsgame.game([p, q], [4, 2], profiles, payoffs)
    mix = np.array([1, 0, 0, 0, 1, 0])
    pays = game.deviation_payoffs(mix)
    assert not np.isnan(pays).any()


@pytest.mark.parametrize('players,strategies', testutils.games)
def test_json_copy_basegame(players, strategies):
    game1 = rsgame.basegame(players, strategies)
    serial = gamegen.serializer(game1)
    game2, _ = gameio.read_basegame(serial.to_basegame_json(game1))
    assert game1 == game2


@pytest.mark.parametrize('players,strategies', testutils.games)
def test_json_copy_game(players, strategies):
    game1 = gamegen.role_symmetric_game(players, strategies)
    serial = gamegen.serializer(game1)
    game2, _ = gameio.read_game(serial.to_game_json(game1))
    assert game1 == game2
    assert np.all(game1.profiles == game2.profiles)
    assert np.allclose(game1.payoffs, game2.payoffs)


@pytest.mark.parametrize('game_size,samples',
                         zip(testutils.games, itertools.cycle([1, 2, 5, 10])))
def test_json_copy_samplegame(game_size, samples):
    base = gamegen.role_symmetric_game(*game_size)
    game1 = gamegen.add_noise(base, 1, samples)
    serial = gamegen.serializer(game1)
    game2, _ = gameio.read_samplegame(serial.to_samplegame_json(game1))
    assert game1 == game2
    assert np.all(game1.profiles == game2.profiles)
    assert np.allclose(game1.payoffs, game2.payoffs)
    for spay1, spay2 in zip(game1.sample_payoffs, game2.sample_payoffs):
        assert np.allclose(spay1, spay2)
