import itertools

import numpy as np
import numpy.random as rand

from gameanalysis import gamegen
from gameanalysis import reduction
from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import utils

from test import testutils


@testutils.apply(testutils.game_sizes())
def test_pure_subgame(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    subgames = subgame.pure_subgames(game)
    expectation = game.num_strategies[None].repeat(game.num_roles, 0)
    np.fill_diagonal(expectation, 1)
    expectation = game.role_repeat(expectation.prod(1))
    assert np.all(subgames.sum(0) == expectation)


def test_subgame():
    game = rsgame.BaseGame([3, 4], [3, 2])
    subg = np.asarray([1, 0, 1, 0, 1], bool)
    devs = subgame.deviation_profiles(game, subg)
    assert devs.shape[0] == 7, \
        "didn't generate the right number of deviating profiles"
    adds = subgame.additional_strategy_profiles(game, subg, 1).shape[0]
    assert adds == 6, \
        "didn't generate the right number of additional profiles"
    subg2 = subg.copy()
    subg2[1] = True
    assert (subgame.subgame(game, subg2).num_all_profiles ==
            adds + subgame.subgame(game, subg).num_all_profiles), \
        "additional profiles didn't return the proper amount"

    serial = gamegen.game_serializer(game)
    sub_serial = subgame.subserializer(serial, subg)
    assert (subgame.subgame(game, subg).num_role_strats ==
            sub_serial.num_role_strats)


@testutils.apply(testutils.game_sizes('small'))
def test_maximal_subgames(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    subs = subgame.maximal_subgames(game)
    assert subs.shape[0] == 1, \
        "found more than maximal subgame in a complete game"
    assert subs.all(), \
        "found subgame wasn't the full one"


@testutils.apply(itertools.product(testutils.game_sizes('small'),
                                   [0.9, 0.6, 0.4]))
def test_missing_data_maximal_subgames(game_desc, prob):
    base = rsgame.BaseGame(*game_desc)
    game = gamegen.add_profiles(base, prob)
    subs = subgame.maximal_subgames(game)

    if subs.size:
        maximal = np.all(subs <= subs[:, None], -1)
        np.fill_diagonal(maximal, False)
        assert not maximal.any(), \
            "One maximal subgame dominated another"

    for sub in subs:
        subprofs = subgame.translate(subgame.subgame(base, sub).all_profiles(),
                                     sub)
        assert all(p in game for p in subprofs), \
            "Maximal subgame didn't have all profiles"
        for dev in np.nonzero(~sub)[0]:
            devprofs = subgame.additional_strategy_profiles(
                game, sub, dev)
            assert not all(p in game for p in devprofs), \
                "Maximal subgame could be bigger {} {}".format(
                    dev, sub)


@testutils.apply(testutils.game_sizes('big'), repeat=20)
def test_deviation_profile_count(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    sup = (rand.random(game.num_roles) * game.num_strategies).astype(int) + 1
    inds = np.concatenate([rand.choice(s, x) + o for s, x, o
                           in zip(game.num_strategies, sup, game.role_starts)])
    mask = np.zeros(game.num_role_strats, bool)
    mask[inds] = True

    devs = subgame.deviation_profiles(game, mask)
    assert devs.shape[0] == subgame.num_deviation_profiles(game, mask), \
        "num_deviation_profiles didn't return correct number"
    assert np.sum(devs > 0) == subgame.num_deviation_payoffs(game, mask), \
        "num_deviation_profiles didn't return correct number"
    assert np.all(np.sum(devs * ~mask, 1) == 1)

    count = 0
    for r_ind in range(game.num_roles):
        r_devs = subgame.deviation_profiles(game, mask, r_ind)
        assert np.all(np.sum(r_devs * ~mask, 1) == 1)
        count += r_devs.shape[0]
    assert count == subgame.num_deviation_profiles(game, mask)

    red = reduction.DeviationPreserving(
        game.num_strategies, game.num_players ** 2, game.num_players)
    dpr_devs = red.expand_profiles(subgame.deviation_profiles(
        game, mask)).shape[0]
    num = subgame.num_dpr_deviation_profiles(game, mask)
    assert dpr_devs == num, \
        "num_dpr_deviation_profiles didn't return correct number"


@testutils.apply(testutils.game_sizes(), repeat=20)
def test_subgame_preserves_completeness(players, strategies):
    """Test that subgame function preserves completeness"""
    game = gamegen.role_symmetric_game(players, strategies)
    assert game.is_complete(), "gamegen didn't create complete game"

    mask = game.random_profiles(game.uniform_mixture())[0] > 0

    sub_game = subgame.subgame(game, mask)
    assert sub_game.is_complete(), "subgame didn't preserve game completeness"

    sgame = gamegen.add_noise(game, 1, 3)
    sub_sgame = subgame.subgame(sgame, mask)
    assert sub_sgame.is_complete(), \
        "subgame didn't preserve sample game completeness"


def test_translate():
    prof = np.arange(6) + 1
    mask = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1], bool)
    expected = [1, 0, 0, 2, 3, 0, 4, 5, 0, 6]
    assert np.all(expected == subgame.translate(prof, mask))


def test_num_subgames():
    game = rsgame.BaseGame([3, 4], [4, 3])
    actual = subgame.num_pure_subgames(game)
    expected = subgame.pure_subgames(game).shape[0]
    assert actual == 12 == expected

    actual = subgame.num_all_subgames(game)
    assert actual == 105


@testutils.apply([
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
    ([3, 4], [4, 3]),
    ([1, 2, 3], [3, 1, 2]),
])
def test_all_subgames(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    all_subgames = subgame.all_subgames(game)
    assert game.role_reduce(all_subgames, ufunc=np.logical_or).all(), \
        "Not all subgames were valid"

    distinct = np.unique(utils.axis_to_elem(all_subgames)).size
    assert distinct == all_subgames.shape[0]

    ids = subgame.subgame_id(game, all_subgames)
    distinct_ids = np.unique(ids).size
    assert distinct_ids == all_subgames.shape[0]

    all_subgames2 = subgame.subgame_from_id(game, ids)
    assert np.all(all_subgames == all_subgames2)


@testutils.apply(testutils.game_sizes())
def test_random_subgames(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    rand_subgames = subgame.random_subgames(game, 30)
    assert rand_subgames.shape[0] == 30
    assert game.role_reduce(rand_subgames, ufunc=np.logical_or).all(), \
        "Not all subgames were valid"

    rand_subgames2 = subgame.subgame_from_id(
        game, subgame.subgame_id(game, rand_subgames))
    assert np.all(rand_subgames == rand_subgames2)


def test_maximal_subgames_partial_profiles():
    """Test that maximal subgames properly handles partial profiles"""
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[1, 0],
               [np.nan, 2],
               [0, 3]]
    game = rsgame.Game([2], [2], profiles, payoffs)
    subs = subgame.maximal_subgames(game)
    expected = utils.axis_to_elem(np.array([
        [True, False],
        [False, True]]))
    assert np.setxor1d(utils.axis_to_elem(subs), expected).size == 0, \
        "Didn't produce both pure subgames"
