import numpy as np
import pytest

from gameanalysis import aggfn
from gameanalysis import agggen
from gameanalysis import nash
from gameanalysis import rsgame


def verify_aggfn(game):
    payoff_game = game.to_rsgame()
    assert game.is_complete()

    # Check accuracy of min and max payoffs
    assert np.all(
        (payoff_game.payoffs >= game.min_strat_payoffs() - 1e-6) |
        (payoff_game.profiles == 0))
    assert np.all(
        (payoff_game.payoffs <= game.max_strat_payoffs() + 1e-6) |
        (payoff_game.profiles == 0))

    # Check that we get the same deviations if we construct the full game
    # game deviation payoff jacobian is inaccurate for sparse mixtures, so we
    # can't use it as ground truth
    for mix in game.random_mixtures(20):
        dev, jac = game.deviation_payoffs(mix, jacobian=True)
        tdev, tjac = payoff_game.deviation_payoffs(mix, assume_complete=True,
                                                   jacobian=True)
        assert np.allclose(dev, tdev)
        assert np.allclose(jac, tjac)

    # Check that sparse mixtures don't result in nans since we don't have
    # ground truth
    for mix in game.random_sparse_mixtures(20):
        dev, jac = game.deviation_payoffs(mix, jacobian=True)
        assert not np.isnan(dev).any()
        assert not np.isnan(jac).any()


@pytest.mark.parametrize('players,strategies,functions', [
    (2 * [1], 1, 1),
    (2 * [1], 2, 2),
    (2 * [2], 1, 2),
    (2 * [2], 2, 2),
    (5 * [1], 2, 3),
    (2 * [1], 5, 3),
    (2 * [2], 5, 4),
    ([1, 2], 2, 5),
    ([1, 2], [2, 1], 4),
    (2, [1, 2], 4),
    ([3, 4], [2, 3], 6),
])
def test_random_sum_game(players, strategies, functions):
    """Test that deviation payoff formulation is accurate"""
    game = agggen.random_aggfn(players, strategies, functions)
    verify_aggfn(game)


@pytest.mark.parametrize('players,strategies,functions', [
    (2 * [1], 1, 1),
    (2 * [1], 2, 2),
    (2 * [2], 1, 2),
    (2 * [2], 2, 2),
    (5 * [1], 2, 3),
    (2 * [1], 5, 3),
    (2 * [2], 5, 4),
    ([1, 2], 2, 5),
    ([1, 2], [2, 1], 4),
    (2, [1, 2], 4),
    ([3, 4], [2, 3], 6),
])
def test_random_role_game(players, strategies, functions):
    """Test that deviation payoff formulation is accurate"""
    game = agggen.random_aggfn(players, strategies, functions, by_role=True)
    verify_aggfn(game)


@pytest.mark.parametrize('players,strategies,functions', [
    ([5, 5], 2, 3),
    ([2, 2], 5, 3),
])
@pytest.mark.parametrize('by_role', [False, True])
def test_nash_finding(players, strategies, functions, by_role):
    game = agggen.random_aggfn(players, strategies, functions, by_role=by_role)
    eqa = nash.mixed_nash(game)
    assert eqa.size > 0, "didn't find any equilibria"


def test_alternate_constructors():
    game = aggfn.aggfn(2, 2, [[1, 2]], [[True], [True]], [[0, 1, 2]])
    game2 = aggfn.aggfn_copy(game, [[1, 2]], [[True], [True]], [[0, 1, 2]])
    assert game == game2


def test_from_function():
    base = rsgame.basegame([2, 2], 2)
    game = aggfn.aggfn_funcs([2, 2], 2, [[1, 2, 3, 4]],
                             [[True], [True], [True], [True]], [lambda x: x])
    assert len(game._function_table.shape) == 2
    game = aggfn.aggfn_funcs_copy(
        base, [[1, 2, 3, 4]], [[True], [True], [True], [True]], [lambda x: x])
    assert len(game._function_table.shape) == 2
    game = aggfn.aggfn_funcs(
        [2, 2], 2, [[1, 2, 3, 4]], [[True], [True], [True], [True]],
        [lambda x, y: x + y])
    assert len(game._function_table.shape) == 3
    game = aggfn.aggfn_funcs_copy(
        base, [[1, 2, 3, 4]], [[True], [True], [True], [True]],
        [lambda x, y: x + y])
    assert len(game._function_table.shape) == 3


@pytest.mark.parametrize('by_role', [False, True])
def test_serializer(by_role):
    game = agggen.random_aggfn([5, 4], [4, 3], 3, by_role=by_role)
    serial = agggen.serializer(game)
    assert repr(serial) is not None

    jgame = serial.to_agfngame_json(game)
    game2 = serial.from_agfngame_json(jgame)
    assert game == game2
    game3, serial3 = aggfn.read_agfngame(jgame)
    assert serial == serial3
    assert game == game3


def test_aggfn_repr():
    game = agggen.random_aggfn(5, 4, 3)
    expected = 'AgfnGame([5], [4], 3)'
    assert repr(game) == expected

    game = agggen.random_aggfn([5, 4], [4, 3], 3)
    expected = 'AgfnGame([5 4], [4 3], 3)'
    assert repr(game) == expected
