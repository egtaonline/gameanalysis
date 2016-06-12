import itertools
import os

import numpy as np
import numpy.random as rand

from gameanalysis import gamegen
from gameanalysis import reduction
from gameanalysis import rsgame
from test import testutils


@testutils.apply(itertools.product(
    [0.6, 0.9, 1],
    [
        ([1], [1], [1]),
        ([2], [1], [2]),
        ([3], [1], [2]),
        ([1], [2], [1]),
        ([1, 2], [2, 1], [1, 2]),
        ([1, 4], [2, 1], [1, 2]),
        ([4, 9], [3, 2], [2, 3]),
    ]), repeat=10)
# Simple test of exact DPR
def test_dpr(keep_prob, game_desc):
    players, strategies, red_players = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    sgame = gamegen.add_noise(game, 1, 3)
    red = reduction.DeviationPreserving(strategies, players, red_players)

    # Try to reduce game
    red_game = red.reduce_game(game)
    red_sgame = red.reduce_game(sgame)

    # Assert that reducing all profiles covers reduced game
    reduced_full_profiles = red_game.profile_id(
        red.reduce_profiles(game.profiles))
    reduced_profiles = red_game.profile_id(red_game.profiles)
    assert np.setdiff1d(reduced_profiles, reduced_full_profiles).size == 0, \
        "reduced game contained profiles it shouldn't have"
    reduced_sample_profiles = red_game.profile_id(red_sgame.profiles)
    assert np.setdiff1d(reduced_sample_profiles,
                        reduced_full_profiles).size == 0, \
        "reduced sample game contained profiles it shouldn't have"
    assert np.setxor1d(reduced_sample_profiles,
                       reduced_profiles).size == 0, \
        "reduced sample game and reduced game had different profiles"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game
    full_profiles = game.profile_id(game.profiles)
    full_reduced_profiles = game.profile_id(
        red.expand_profiles(red_game.profiles))
    assert np.setdiff1d(full_reduced_profiles, full_profiles).size == 0, \
        "full game did not have data for all profiles required of reduced"
    full_reduced_sample_profiles = game.profile_id(
        red.expand_profiles(red_sgame.profiles))
    assert np.setdiff1d(full_reduced_sample_profiles,
                        full_profiles).size == 0, \
        ("full sample game did not have data for all profiles required of "
         "reduced")
    assert np.setxor1d(full_reduced_profiles,
                       full_reduced_sample_profiles).size == 0, \
        "sample game didn't produce identical results"


def test_empty_dpr_1():
    """Reduction is empty because profile is invalid"""
    profiles = [
        [2, 4],
    ]
    payoffs = [
        [1, 2],
    ]
    game = rsgame.Game(6, 2, profiles, payoffs)
    red = reduction.DeviationPreserving([2], [6], [2])
    red_game = red.reduce_game(game)
    assert red_game.is_empty()


def test_empty_dpr_2():
    """Reduction is empty because profile doesn't have all payoffs"""
    profiles = [
        [1, 3],
    ]
    payoffs = [
        [1, 2],
    ]
    game = rsgame.Game(4, 2, profiles, payoffs)
    red = reduction.DeviationPreserving([2], [4], [2])
    red_game = red.reduce_game(game)
    assert red_game.is_empty()


def test_empty_sample_dpr_1():
    """Reduction is empty because profile is invalid"""
    profiles = [
        [2, 4],
    ]
    payoffs = [
        [
            [[1], [2]],
        ],
    ]
    game = rsgame.SampleGame(6, 2, profiles, payoffs)
    red = reduction.DeviationPreserving([2], [6], [2])
    red_game = red.reduce_game(game)
    assert red_game.is_empty()


def test_empty_sample_dpr_2():
    """Reduction is empty because profile doesn't have all payoffs"""
    profiles = [
        [1, 3],
    ]
    payoffs = [
        [
            [[1], [2]],
        ],
    ]
    game = rsgame.SampleGame(4, 2, profiles, payoffs)
    red = reduction.DeviationPreserving([2], [4], [2])
    red_game = red.reduce_game(game)
    assert red_game.is_empty()


def test_dpr_repr():
    red = repr(reduction.DeviationPreserving([3], [4], [2]))
    repr_str = ("DeviationPreserving([3], [4], [2])")
    assert red == repr_str


@testutils.apply(itertools.product(
    [0.6, 0.9, 1],
    [
        ([1], [1]),
        ([2], [1]),
        ([3], [1]),
        ([1], [2]),
        ([1, 2], [2, 1]),
        ([1, 4], [2, 1]),
        ([4, 8], [3, 2]),
    ]), repeat=10)
# Simple test of Twins Reduction
def test_twins(keep_prob, game_desc):
    players, strategies = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    red = reduction.Twins(strategies, players)

    # Try to reduce game
    red_game = red.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    reduced_full_profiles = red_game.profile_id(
        red.reduce_profiles(game.profiles))
    reduced_profiles = red_game.profile_id(red_game.profiles)
    assert np.setdiff1d(reduced_profiles, reduced_full_profiles).size == 0, \
        "reduced game contained profiles it shouldn't have"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game
    full_profiles = game.profile_id(game.profiles)
    full_reduced_profiles = game.profile_id(
        red.expand_profiles(red_game.profiles))
    assert np.setdiff1d(full_reduced_profiles, full_profiles).size == 0, \
        "full game did not have data for all profiles required of reduced"


def test_twins_repr():
    red = repr(reduction.Twins([2], [4]))
    repr_str = "Twins([2], [4])"
    assert red == repr_str


@testutils.apply(itertools.product(
    [0.6, 0.9, 1],
    [
        ([1], [1], [1]),
        ([2], [1], [2]),
        ([4], [1], [2]),
        ([1], [2], [1]),
        ([1, 2], [2, 1], [1, 2]),
        ([1, 4], [2, 1], [1, 2]),
        ([4, 9], [3, 2], [2, 3]),
    ]), repeat=10)
# Simple test of Hierarchical
def test_hierarchical(keep_prob, game_desc):
    players, strategies, red_players = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    sgame = gamegen.add_noise(game, 1, 3)
    red = reduction.Hierarchical(strategies, players, red_players)

    # Try to reduce game
    red_game = red.reduce_game(game)
    red_sgame = red.reduce_game(sgame)

    # Assert that reducing all profiles covers reduced game
    reduced_full_profiles = red_game.profile_id(
        red.reduce_profiles(game.profiles))
    reduced_profiles = red_game.profile_id(red_game.profiles)
    assert np.setxor1d(reduced_profiles, reduced_full_profiles).size == 0, \
        "reduced game contained profiles it shouldn't have"
    reduced_sample_profiles = red_game.profile_id(red_sgame.profiles)
    assert np.setxor1d(reduced_sample_profiles,
                       reduced_full_profiles).size == 0, \
        "reduced game contained profiles it shouldn't have"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game
    full_profiles = game.profile_id(game.profiles)
    full_reduced_profiles = game.profile_id(
        red.expand_profiles(red_game.profiles))
    assert np.setdiff1d(full_reduced_profiles, full_profiles).size == 0, \
        "full game did not have data for all profiles required of reduced"
    full_reduced_sample_profiles = game.profile_id(
        red.expand_profiles(red_sgame.profiles))
    assert np.setdiff1d(full_reduced_sample_profiles,
                        full_profiles).size == 0, \
        "full game did not have data for all profiles required of reduced"
    assert np.setxor1d(full_reduced_profiles,
                       full_reduced_sample_profiles).size == 0, \
        "sample game didn't produce identical results"


def test_empty_sample_hierarchical():
    profiles = [
        [1, 3],
    ]
    payoffs = [
        [
            [[1], [2]],
        ],
    ]
    game = rsgame.SampleGame(4, 2, profiles, payoffs)
    red = reduction.Hierarchical([2], [4], [2])
    red_game = red.reduce_game(game)
    assert red_game.is_empty()


def test_hierarchical_repr():
    red = repr(reduction.Hierarchical([3], [4], [2]))
    repr_str = "Hierarchical([3], [4], [2])"
    assert red == repr_str


@testutils.apply(itertools.product(
    [0, 0.6, 1],
    [
        ([1], [1]),
        ([2], [1]),
        ([3], [1]),
        ([1], [2]),
        ([1, 2], [2, 1]),
        ([1, 4], [2, 1]),
        ([4, 9], [3, 2]),
    ]))
# Simple test of identity reduction
def test_identity(keep_prob, game_desc):
    players, strategies = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    red = reduction.Identity()

    # Try to reduce game
    red_game = red.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    reduced_full_profiles = red_game.profile_id(
        red.reduce_profiles(game.profiles))
    reduced_profiles = red_game.profile_id(red_game.profiles)
    assert np.setxor1d(reduced_full_profiles, reduced_profiles).size == 0, \
        "reduced game didn't match full game"

    full_profiles = game.profile_id(game.profiles)
    full_reduced_profiles = game.profile_id(
        red.expand_profiles(red_game.profiles))
    assert np.setxor1d(full_profiles, full_reduced_profiles).size == 0, \
        "full game did not match reduced game"


def test_identity_repr():
    red = reduction.Identity()
    assert repr(red) == "Identity()"


def test_rsym_expand_tie_breaking():
    """Test that standard expansion breaks ties appropriately"""

    def expand(prof, full, red):
        return reduction._expand_rsym_profiles(
            rsgame.BaseGame(1, len(prof)), np.asarray(prof)[None],
            np.array([full]), np.array([red]))

    full = expand([4, 2], 20, 6)
    assert np.all(full == [13, 7]), \
        "Didn't tie break on approximation first"
    full = expand([1, 3], 10, 4)
    assert np.all(full == [2, 8]), \
        "Didn't tie break on larger count if approximations same"
    full = expand([2, 2], 11, 4)
    assert np.all(full == [6, 5]), \
        "Didn't tie break on strategy name if all else same"


def test_rsym_reduce_tie_breaking():
    """Test that the standard reduction breaks ties appropriately"""

    def reduce_prof(prof, full, red):
        return reduction._reduce_rsym_profiles(
            rsgame.BaseGame(1, len(prof)), np.asarray(prof)[None],
            np.array([full]), np.array([red]))[0]

    red = reduce_prof([13, 7], 20, 6)
    assert np.all(red == [4, 2]), \
        "Didn't tie break on approximation first"
    # Expanded profile has no valid reduction
    red = reduce_prof([14, 6], 20, 6)
    assert red.size == 0, "Didn't properly find no reduced profile"
    red = reduce_prof([2, 8], 10, 4)
    assert np.all(red == [1, 3]), \
        "Didn't tie break on larger count if approximations same"
    red = reduce_prof([6, 5], 11, 4)
    assert np.all(red == [2, 2]), \
        "Didn't tie break on strategy name if all else same"


@testutils.apply(repeat=1000)
def test_random_identity_rsym():
    num_strats = rand.randint(2, 20)
    profile = rand.randint(20, size=num_strats)[None]
    reduced_players = profile.sum()
    while reduced_players < 2:
        profile[0, rand.randint(num_strats)] += 1  # pragma: no cover
        reduced_players += 1  # pragma: no cover
    full_players = rand.randint(reduced_players + 1, reduced_players ** 2)
    game = rsgame.BaseGame(reduced_players, num_strats)
    rp = np.array([reduced_players])
    fp = np.array([full_players])
    full_profile = reduction._expand_rsym_profiles(game, profile, fp, rp)
    red_profile, valid = reduction._reduce_rsym_profiles(game, full_profile,
                                                         fp, rp)
    assert np.all(valid), \
        "reduction wasn't valid, but it definitely was:\n{}\n{}".format(
            profile, full_profile)
    assert np.all(red_profile == profile), \
        "reduction was valid, but not identity:\n{}\n{}\n{}".format(
            profile, full_profile, red_profile)


def test_approximate_dpr_expansion():
    """Test expansion on approximate dpr"""
    game = rsgame.BaseGame([8, 11], 2)
    red = reduction.DeviationPreserving([2, 2], [8, 11], [3, 4])
    red_prof = [[1, 2, 2, 2]]
    full_profs, contributions = red.expand_profiles(red_prof, True)
    profs = game.profile_id(full_profs)
    expected = game.profile_id([
        [4, 4, 6, 5],
        [1, 7, 6, 5],
        [3, 5, 4, 7],
        [3, 5, 7, 4]])
    assert np.setxor1d(profs, expected).size == 0, \
        "generated different profiles than expected"
    expected_conts = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], bool)
    eord = np.argsort(expected)
    pord = np.argsort(profs)
    assert np.all(expected_conts[eord] == contributions[pord])


def test_expansion_contributions():
    """Test expansion on approximate dpr"""
    game = rsgame.BaseGame([4, 9], 2)
    red = reduction.DeviationPreserving([2, 2], [4, 9], [2, 3])
    red_profs = [
        [2, 0, 0, 3],
        [1, 1, 3, 0]]
    full_profs, contributions = red.expand_profiles(red_profs, True)
    profs = game.profile_id(full_profs)
    expected = game.profile_id([
        [4, 0, 0, 9],
        [1, 3, 9, 0],
        [3, 1, 9, 0],
        [2, 2, 9, 0]])
    assert np.setxor1d(profs, expected).size == 0, \
        "generated different profiles than expected"
    expected_conts = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]], bool)
    eord = np.argsort(expected)
    pord = np.argsort(profs)
    assert np.all(expected_conts[eord] == contributions[pord])


def test_approximate_dpr_reduce_game():
    """Test approximate dpr game reduction"""
    game = gamegen.role_symmetric_game([3, 4], 2)
    red = reduction.DeviationPreserving([2, 2], [3, 4], [2, 2])
    redgame = red.reduce_game(game)
    # Pure strategies are reduced properly
    assert (redgame.get_payoffs([2, 0, 0, 2])[0]
            == game.get_payoffs([3, 0, 0, 4])[0])
    # Mixed strategies are reduced properly
    assert (redgame.get_payoffs([1, 1, 1, 1])[0]
            == game.get_payoffs([1, 2, 2, 2])[0])
    assert (redgame.get_payoffs([1, 1, 1, 1])[1]
            == game.get_payoffs([2, 1, 2, 2])[1])
    assert (redgame.get_payoffs([1, 1, 1, 1])[2]
            == game.get_payoffs([2, 1, 1, 3])[2])
    assert (redgame.get_payoffs([1, 1, 1, 1])[3]
            == game.get_payoffs([2, 1, 3, 1])[3])


def test_sample_game_payoff():
    profiles = [
        [0, 4, 0, 9],
        [0, 4, 1, 8],
        [0, 4, 4, 5],
        [0, 4, 3, 6],
    ]
    payoffs = [
        [
            [[0] * 4, [1, 2, 3, 4], [0] * 4, [5, 6, 7, 8]],
        ],
        [
            [[0, 0], [0, 0], [9, 10], [0, 0]],
        ],
        [
            [[0] * 3, [0] * 3, [0] * 3, [11, 12, 13]],
        ],
        [
            [[0] * 5, [14, 15, 16, 17, 18], [0] * 5, [0] * 5],
        ],
    ]
    game = rsgame.SampleGame([4, 9], 2, profiles, payoffs)
    red = reduction.DeviationPreserving([2, 2], [4, 9], [2, 3])
    red_game = red.reduce_game(game)

    prof_map = dict(zip(
        red_game.profile_id(red_game.profiles),
        itertools.chain.from_iterable(red_game.sample_payoffs)))

    payoffs = prof_map[red_game.profile_id([0, 2, 0, 3])]
    actual = payoffs[1]
    expected = [1, 2, 3, 4]
    assert np.setxor1d(actual, expected).size == 0
    actual = payoffs[3]
    expected = [5, 6, 7, 8]
    assert np.setxor1d(actual, expected).size == 0

    payoffs = prof_map[red_game.profile_id([0, 2, 1, 2])]
    actual = payoffs[1]
    expected = [14, 15, 16, 17, 18]
    assert np.setxor1d(actual, expected).size == 3
    actual = payoffs[2]
    expected = [9, 10]
    assert np.setxor1d(actual, expected).size == 0
    actual = payoffs[3]
    expected = [11, 12, 13]
    assert np.setxor1d(actual, expected).size == 1


@testutils.apply([
    ([1], [1]),
    ([2], [1]),
    ([3], [1]),
    ([1], [2]),
    ([1, 2], [2, 1]),
    ([1, 4], [2, 1]),
    ([4, 9], [3, 2]),
] + ([
    ([3] * 3, [3] * 3),
    ([3, 4, 9], [4, 3, 2]),
] if os.getenv('BIG_TESTS') == 'ON' else []), repeat=20)
def test_random_approximate_dpr(players, strategies):
    """Test approximate dpr preserves completeness on random games"""
    game = gamegen.role_symmetric_game(players, strategies)
    red_counts = 2 + (rand.random(game.num_roles) * (game.num_players - 1))\
        .astype(int)
    red_counts[game.num_players == 1] = 1
    red = reduction.DeviationPreserving(strategies, players, red_counts)

    # Try to reduce game
    red_game = red.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    assert red_game.is_complete(), "DPR did not preserve completeness"
