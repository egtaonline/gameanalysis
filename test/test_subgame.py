import numpy as np
import pytest

from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import utils
from gameanalysis.reduction import deviation_preserving as dpr
from test import testutils


def test_subgame():
    game = rsgame.emptygame([3, 4], [3, 2])
    subg = np.asarray([1, 0, 1, 0, 1], bool)
    devs = subgame.deviation_profiles(game, subg)
    assert devs.shape[0] == 7, \
        "didn't generate the right number of deviating profiles"
    adds = subgame.additional_strategy_profiles(game, subg, 1).shape[0]
    assert adds == 6, \
        "didn't generate the right number of additional profiles"
    subg2 = subg.copy()
    subg2[1] = True
    assert (game.subgame(subg2).num_all_profiles ==
            adds + game.subgame(subg).num_all_profiles), \
        "additional profiles didn't return the proper amount"


@pytest.mark.parametrize('players,strategies', testutils.games)
def test_maximal_subgames(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    subs = subgame.maximal_subgames(game)
    assert subs.shape[0] == 1, \
        "found more than maximal subgame in a complete game"
    assert subs.all(), \
        "found subgame wasn't the full one"


@pytest.mark.parametrize('players,strategies', testutils.games)
@pytest.mark.parametrize('prob', [0.9, 0.6, 0.4])
def test_missing_data_maximal_subgames(players, strategies, prob):
    base = rsgame.emptygame(players, strategies)
    game = gamegen.add_profiles(base, prob)
    subs = subgame.maximal_subgames(game)

    if subs.size:
        maximal = np.all(subs <= subs[:, None], -1)
        np.fill_diagonal(maximal, False)
        assert not maximal.any(), \
            "One maximal subgame dominated another"

    for sub in subs:
        subprofs = subgame.translate(base.subgame(sub).all_profiles(), sub)
        assert all(p in game for p in subprofs), \
            "Maximal subgame didn't have all profiles"
        for dev in np.nonzero(~sub)[0]:
            devprofs = subgame.additional_strategy_profiles(
                game, sub, dev)
            assert not all(p in game for p in devprofs), \
                "Maximal subgame could be bigger {} {}".format(
                    dev, sub)  # pragma: no cover


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', testutils.games)
def test_deviation_profile_count(players, strategies, _):
    game = rsgame.emptygame(players, strategies)
    mask = game.random_subgames()

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

    full_game = rsgame.emptygame(
        game.num_role_players ** 2, game.num_role_strats)
    dpr_devs = dpr.expand_profiles(
        full_game, subgame.deviation_profiles(game, mask)).shape[0]
    num = subgame.num_dpr_deviation_profiles(game, mask)
    assert dpr_devs == num, \
        "num_dpr_deviation_profiles didn't return correct number"


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', testutils.games)
def test_subgame_preserves_completeness(players, strategies, _):
    """Test that subgame function preserves completeness"""
    game = gamegen.role_symmetric_game(players, strategies)
    assert game.is_complete(), "gamegen didn't create complete game"

    mask = game.random_subgames()
    sub_game = game.subgame(mask)
    assert sub_game.is_complete(), "subgame didn't preserve game completeness"

    sgame = gamegen.add_noise(game, 1, 3)
    sub_sgame = sgame.subgame(mask)
    assert sub_sgame.is_complete(), \
        "subgame didn't preserve sample game completeness"


def test_translate():
    prof = np.arange(6) + 1
    mask = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1], bool)
    expected = [1, 0, 0, 2, 3, 0, 4, 5, 0, 6]
    assert np.all(expected == subgame.translate(prof, mask))


def test_maximal_subgames_partial_profiles():
    """Test that maximal subgames properly handles partial profiles"""
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[1, 0],
               [np.nan, 2],
               [0, 3]]
    game = paygame.game([2], [2], profiles, payoffs)
    subs = subgame.maximal_subgames(game)
    expected = utils.axis_to_elem(np.array([
        [True, False],
        [False, True]]))
    assert np.setxor1d(utils.axis_to_elem(subs), expected).size == 0, \
        "Didn't produce both pure subgames"


@pytest.mark.parametrize('players,strategies', testutils.games)
def test_subgame_to_from_id(players, strategies):
    """Test that subgame function preserves completeness"""
    game = rsgame.emptygame(players, strategies)
    subgs = game.all_subgames()
    subgs2 = subgame.from_id(game, subgame.to_id(game, subgs))
    assert np.all(subgs == subgs2)
