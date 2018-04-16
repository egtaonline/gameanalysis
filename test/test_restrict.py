"""Test restrict"""
import numpy as np
import pytest

from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from test import utils as testutils # pylint: disable=wrong-import-order


def test_restriction():
    """Test basic restriction"""
    game = rsgame.empty([3, 4], [3, 2])
    rest = np.asarray([1, 0, 1, 0, 1], bool)
    devs = restrict.deviation_profiles(game, rest)
    assert devs.shape[0] == 7, \
        "didn't generate the right number of deviating profiles"
    adds = restrict.additional_strategy_profiles(game, rest, 1).shape[0]
    assert adds == 6, \
        "didn't generate the right number of additional profiles"
    rest2 = rest.copy()
    rest2[1] = True
    assert (game.restrict(rest2).num_all_profiles ==
            adds + game.restrict(rest).num_all_profiles), \
        "additional profiles didn't return the proper amount"


@pytest.mark.parametrize('players,strategies', testutils.GAMES)
def test_maximal_restrictions(players, strategies):
    """Test maximal restrictions"""
    game = gamegen.game(players, strategies)
    rests = restrict.maximal_restrictions(game)
    assert rests.shape[0] == 1, \
        'found more than maximal restriction in a complete game'
    assert rests.all(), \
        "found restriction wasn't the full one"


@pytest.mark.parametrize('players,strategies', testutils.GAMES)
@pytest.mark.parametrize('prob', [0.9, 0.6, 0.4])
def test_missing_data_maximal_restrictions(players, strategies, prob):
    """Test missing data"""
    game = gamegen.game(players, strategies, prob)
    rests = restrict.maximal_restrictions(game)

    if rests.size:
        maximal = np.all(rests <= rests[:, None], -1)
        np.fill_diagonal(maximal, False)
        assert not maximal.any(), \
            'One maximal restriction dominated another'

    for rest in rests:
        rgame = rsgame.empty_copy(game).restrict(rest)
        restprofs = restrict.translate(rgame.all_profiles(), rest)
        assert all(p in game for p in restprofs), \
            "Maximal restriction didn't have all profiles"
        for dev in np.nonzero(~rest)[0]:
            devprofs = restrict.additional_strategy_profiles(
                game, rest, dev)
            assert not all(p in game for p in devprofs), (  # pragma: no branch
                'Maximal restriction could be bigger {} {}'.format(
                    dev, rest))


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', testutils.GAMES)
def test_random_deviation_profile_count(players, strategies, _):
    """Test dev profile count"""
    game = rsgame.empty(players, strategies)
    rest = game.random_restriction()

    devs = restrict.deviation_profiles(game, rest)
    assert devs.shape[0] == restrict.num_deviation_profiles(game, rest), \
        "num_deviation_profiles didn't return correct number"
    assert np.sum(devs > 0) == restrict.num_deviation_payoffs(game, rest), \
        "num_deviation_profiles didn't return correct number"
    assert np.all(np.sum(devs * ~rest, 1) == 1)

    count = 0
    for r_ind in range(game.num_roles):
        r_devs = restrict.deviation_profiles(game, rest, r_ind)
        assert np.all(np.sum(r_devs * ~rest, 1) == 1)
        count += r_devs.shape[0]
    assert count == restrict.num_deviation_profiles(game, rest)


def test_dpr_deviation_count():
    """Test dpr dev count"""
    game = rsgame.empty(3, 2)
    num_devs = restrict.num_dpr_deviation_profiles(
        game, [True, False])
    assert num_devs == 2

    game = rsgame.empty([1, 3], 2)
    num_devs = restrict.num_dpr_deviation_profiles(
        game, [True, True, True, False])
    assert num_devs == 6

    game = rsgame.empty(1, [3, 1])
    num_devs = restrict.num_dpr_deviation_profiles(
        game, [True, True, False, True])
    assert num_devs == 1

    game = rsgame.empty([3, 2, 1], [1, 2, 3])
    num_devs = restrict.num_dpr_deviation_profiles(
        game, [True, True, False, True, False, True])
    assert num_devs == 7


def test_big_game_counts():
    """Test that everything works when game_size > int max"""
    game = rsgame.empty([100, 100], [30, 30])
    num_devs = restrict.num_dpr_deviation_profiles(
        game, [False] + [True] * 58 + [False])
    assert num_devs > np.iinfo(int).max


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', testutils.GAMES)
def test_random_restriction_preserves_completeness(players, strategies, _):
    """Test that restriction function preserves completeness"""
    game = gamegen.game(players, strategies)
    assert game.is_complete(), "gamegen didn't create complete game"

    rest = game.random_restriction()
    rgame = game.restrict(rest)
    assert rgame.is_complete(), \
        "restriction didn't preserve game completeness"

    sgame = gamegen.gen_noise(game)
    redsgame = sgame.restrict(rest)
    assert redsgame.is_complete(), \
        "restriction didn't preserve sample game completeness"


def test_translate():
    """Test translate"""
    prof = np.arange(6) + 1
    rest = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1], bool)
    expected = [1, 0, 0, 2, 3, 0, 4, 5, 0, 6]
    assert np.all(expected == restrict.translate(prof, rest))


def test_maximal_restrictions_partial_profiles():
    """Test that maximal restrictions properly handles partial profiles"""
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[1, 0],
               [np.nan, 2],
               [0, 3]]
    game = paygame.game([2], [2], profiles, payoffs)
    rests = restrict.maximal_restrictions(game)
    expected = utils.axis_to_elem(np.array([
        [True, False],
        [False, True]]))
    assert np.setxor1d(utils.axis_to_elem(rests), expected).size == 0, \
        "Didn't produce both pure restrictions"


@pytest.mark.parametrize('players,strategies', testutils.GAMES)
def test_restriction_to_from_id(players, strategies):
    """Test that restriction function preserves completeness"""
    game = rsgame.empty(players, strategies)
    rests = game.all_restrictions()
    rests2 = restrict.from_id(game, restrict.to_id(game, rests))
    assert np.all(rests == rests2)
