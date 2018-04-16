"""Test aggfn"""
import json

import numpy as np
import pytest

from gameanalysis import aggfn
from gameanalysis import paygame
from gameanalysis import rsgame


_GAME = aggfn.aggfn(
    [2, 1], [2, 3],
    [[-1, 0, 1, 2, 3],
     [0, 1, 0, 1, 0],
     [1, 1, 1, 1, 1],
     [0, 0, 1, 1, 1]],
    [[True, True, True, True],
     [False, True, True, False],
     [True, False, True, True],
     [False, False, False, False],
     [True, False, False, True]],
    [[[0, 1], [1, 3], [2, 0]],
     [[0, 0], [3, 2], [2, 4]],
     [[9, 7], [4, 1], [1, 4]],
     [[3, 6], [0, -1], [3, 4]]])


def rand(players, strategies, functions):
    """Create random game"""
    base = rsgame.empty(players, strategies)
    action_weights = np.random.normal(0, 1, (functions, base.num_strats))
    function_inputs = np.random.random((base.num_strats, functions)) < .5
    for func in function_inputs.T:
        func[np.random.choice(base.num_strats, 2, False)] = [False, True]
    function_table = np.random.normal(
        0, 1, (functions,) + tuple(base.num_role_players + 1))
    offsets = np.random.normal(0, 1, base.num_strats)
    return aggfn.aggfn_replace(base, action_weights, function_inputs,
                               function_table, offsets)


def test_restrict_function_removal():
    """Test that functions are removed when restricted"""
    game = aggfn.aggfn(
        [2, 1], [2, 3],
        [[-1, 0, 1, 2, 3],
         [0, 0, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1]],
        [[True, True, True, True],
         [False, True, True, False],
         [True, False, True, True],
         [False, False, False, False],
         [True, False, False, True]],
        [[[0, 0], [1, 2], [2, 3]],
         [[0, 0], [3, 2], [2, 1]],
         [[9, 8], [4, 5], [1, 0]],
         [[3, 2], [0, 1], [0, 3]]])
    rgame = game.restrict([True, True, True, False, True])
    expected = aggfn.aggfn_names(
        ['r0', 'r1'], [2, 1], [['s0', 's1'], ['s2', 's4']],
        [[-1, 0, 1, 3],
         [1, 1, 1, 1],
         [0, 0, 1, 1]],
        [[True, True, True],
         [False, True, False],
         [True, True, True],
         [True, False, True]],
        [[[0, 0], [1, 2], [2, 3]],
         [[9, 8], [4, 5], [1, 0]],
         [[3, 2], [0, 1], [0, 3]]])
    assert rgame == expected


def test_props():
    """Test game properties"""
    assert _GAME.num_functions == 4
    assert _GAME.num_profiles == 9
    assert _GAME.is_complete()
    assert not _GAME.is_empty()
    assert repr(_GAME) == 'AgfnGame([2 1], [2 3], 4)'


def test_min_max():
    """Test min and max payoffs"""
    mins = [-2, 1, 0, 0, 0]
    assert np.all(mins == _GAME.min_strat_payoffs())
    maxs = [9, 13, 18, 25, 24]
    assert np.all(maxs == _GAME.max_strat_payoffs())


def test_get_payoffs():
    """Test get payoffs"""
    payoffs = [0, 3, 0, 5, 0]
    assert np.allclose(payoffs, _GAME.get_payoffs([1, 1, 0, 1, 0]))
    payoffs = [0, 6, 11, 0, 0]
    assert np.allclose(payoffs, _GAME.get_payoffs([0, 2, 1, 0, 0]))
    payoffs = [0, 3, 0, 6, 0]
    assert np.allclose(payoffs, _GAME.get_payoffs([0, 2, 0, 1, 0]))


def test_dev_pays():
    """Test deviation payoffs"""
    mix = [0.5, 0.5, 0.5, 0.5, 0]
    payoffs = [1, 4.5, 7.75, 6.5, 8.25]
    assert np.allclose(payoffs, _GAME.deviation_payoffs(mix))


def test_payoff_vals():
    """Test payoff values"""
    profiles = [[2, 0, 1, 0, 0],
                [2, 0, 0, 1, 0],
                [2, 0, 0, 0, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 2, 1, 0, 0],
                [0, 2, 0, 1, 0],
                [0, 2, 0, 0, 1]]
    payoffs = [[4, 0, 8, 0, 0],
               [-1, 0, 0, 10, 0],
               [1, 0, 0, 0, 5],
               [1, 6, 6, 0, 0],
               [0, 3, 0, 5, 0],
               [-2, 3, 0, 0, 9],
               [0, 6, 11, 0, 0],
               [0, 3, 0, 6, 0],
               [0, 3, 0, 0, 10]]
    copy = paygame.game_replace(_GAME, profiles, payoffs)
    assert paygame.game_copy(_GAME) == copy


def test_function_perm():
    """Test function permutations preserve equality"""
    functabp = [[[9, 7], [4, 1], [1, 4]],
                [[0, 0], [3, 2], [2, 4]],
                [[0, 1], [1, 3], [2, 0]],
                [[3, 6], [0, -1], [3, 4]]]
    funcinpsp = [[True, True, True, True],
                 [True, True, False, False],
                 [True, False, True, True],
                 [False, False, False, False],
                 [False, False, True, True]]
    actwp = [[1, 1, 1, 1, 1],
             [0, 1, 0, 1, 0],
             [-1, 0, 1, 2, 3],
             [0, 0, 1, 1, 1]]
    perm = aggfn.aggfn(_GAME.num_role_players, _GAME.num_role_strats, actwp,
                       funcinpsp, functabp)
    assert perm == _GAME


def test_restrict():
    """Test restrictions"""
    rest = [False, True, True, False, True]
    sfuncinps = [[False, True, True, False],
                 [True, False, True, True],
                 [True, False, False, True]]
    sactw = [[0, 1, 3],
             [1, 0, 0],
             [1, 1, 1],
             [0, 1, 1]]
    sgame = aggfn.aggfn_names(['r0', 'r1'], [2, 1], [['s1'], ['s2', 's4']],
                              sactw, sfuncinps, _GAME.function_table)
    assert sgame == _GAME.restrict(rest)


def verify_aggfn(game):
    """Verify that aggfn matches the expanded version"""
    payoff_game = paygame.game_copy(game)
    assert not game.is_empty()
    assert game.is_complete()

    ngame = game.normalize()
    assert np.all(
        np.isclose(ngame.max_role_payoffs(), 1) |
        np.isclose(ngame.max_role_payoffs(), 0))
    assert np.allclose(ngame.min_role_payoffs(), 0)

    # Check accuracy of min and max payoffs
    assert np.all(
        (payoff_game.payoffs() >= game.min_strat_payoffs() - 1e-6) |
        (payoff_game.profiles() == 0))
    assert np.all(
        (payoff_game.payoffs() <= game.max_strat_payoffs() + 1e-6) |
        (payoff_game.profiles() == 0))

    # Test that get payoffs works for multiple dimensions
    profiles = game.random_profiles(20).reshape((4, 5, -1))
    payoffs = game.get_payoffs(profiles)
    true_payoffs = payoff_game.get_payoffs(profiles)
    assert np.allclose(payoffs, true_payoffs)

    # Check that we get the same deviations if we construct the full game
    # game deviation payoff jacobian is inaccurate for sparse mixtures, so we
    # can't use it as ground truth
    for mix in game.random_mixtures(20):
        idev, ijac = game.deviation_payoffs(mix, jacobian=True)
        tdev, tjac = payoff_game.deviation_payoffs(mix, jacobian=True)
        assert np.allclose(idev, tdev)
        tjac -= np.repeat(np.add.reduceat(tjac, game.role_starts, 1) /
                          game.num_role_strats, game.num_role_strats, 1)
        ijac -= np.repeat(np.add.reduceat(ijac, game.role_starts, 1) /
                          game.num_role_strats, game.num_role_strats, 1)
        assert np.allclose(ijac, tjac)

    # Check that sparse mixtures produce correct deviations
    # TODO For some reason, the jacobians don't match with the jacobians for
    # the standard payoff game when the mixture is sparse. My hunch is that the
    # aggfn version has payoff effects clse to zero that aren't captured by
    # simply recording the payoffs. However, this doesn't make a whole lot of
    # sense.
    for mix in game.random_sparse_mixtures(20):
        dev = game.deviation_payoffs(mix)
        tdev = payoff_game.deviation_payoffs(mix)
        assert np.allclose(dev, tdev)

    # Check that it serializes properly
    jgame = json.dumps(game.to_json())
    copy = aggfn.aggfn_json(json.loads(jgame))
    assert game == copy
    # As does it's normalized version
    jgame = json.dumps(ngame.to_json())
    copy = aggfn.aggfn_json(json.loads(jgame))
    assert ngame == copy


@pytest.mark.parametrize('players,strategies,functions', [
    (2 * [1], 2, 2),
    (2 * [2], 2, 2),
    (5 * [1], 2, 1),
    (5 * [1], 2, 3),
    (2 * [1], 5, 3),
    (2 * [2], 5, 4),
    ([1, 2], 2, 5),
    ([1, 2], [2, 1], 4),
    (2, [1, 2], 4),
    ([3, 4], [2, 3], 6),
])
def test_random_game(players, strategies, functions):
    """Test that deviation payoff formulation is accurate"""
    verify_aggfn(rand(players, strategies, functions))


def test_alternate_constructors():
    """Test other constructors"""
    game = aggfn.aggfn(2, 2, [[1, 2]], [[True], [False]], [[0, 1, 2]])
    game2 = aggfn.aggfn_replace(game, [[1, 2]], [[True], [False]], [[0, 1, 2]])
    assert game == game2


def test_from_function():
    """Test from function"""
    game = aggfn.aggfn_funcs(2, 2, [[1, 2]], [[True], [False]], [lambda x: x])
    assert len(game.function_table.shape) == 2
    game = aggfn.aggfn_funcs(
        [2, 2], 2, [[1, 2, 3, 4]], [[True], [False], [False], [True]],
        [lambda x, y: x + y])
    assert len(game.function_table.shape) == 3


def test_from_function_call():
    """Test from function with __call__ attribute"""
    class Func1(object): # pylint: disable=too-few-public-methods
        """Function class"""
        def __call__(self, arg):
            return arg

    game = aggfn.aggfn_funcs(2, 2, [[1, 2]],
                             [[True], [False]], [Func1()])
    assert len(game.function_table.shape) == 2

    class Func2(object): # pylint: disable=too-few-public-methods
        """Function class"""
        def __call__(self, arg1, arg2):
            return arg1 + arg2

    game = aggfn.aggfn_funcs([2, 2], 2, [[1, 2, 3, 4]],
                             [[True], [False], [False], [True]], [Func2()])
    assert len(game.function_table.shape) == 3


def test_repr():
    """Test aggfn repr"""
    expected = 'AgfnGame([5], [4], 3)'
    assert repr(rand(5, 4, 3)) == expected

    expected = 'AgfnGame([5 4], [4 3], 3)'
    assert repr(rand([5, 4], [4, 3], 3)) == expected


def test_eq():
    """Test equality"""
    copy = aggfn.aggfn_replace(
        _GAME, _GAME.action_weights, _GAME.function_inputs,
        _GAME.function_table, np.ones(_GAME.num_strats))
    assert copy != _GAME


def test_add():
    """Test game addition"""
    game_a = aggfn.aggfn(
        [2, 1], [2, 3],
        [[-1, 0, 1, 2, 3]],
        [[True], [False], [True], [False], [True]],
        [[[0, 1], [1, 3], [2, 0]]])
    game_b = aggfn.aggfn(
        [2, 1], [2, 3],
        [[0, 1, 3, 1, 0]],
        [[False], [True], [True], [False], [False]],
        [[[1, 2], [3, 4], [5, 0]]])
    game_add = aggfn.aggfn(
        [2, 1], [2, 3],
        [[-1, 0, 1, 2, 3],
         [0, 1, 3, 1, 0]],
        [[True, False],
         [False, True],
         [True, True],
         [False, False],
         [True, False]],
        [[[0, 1], [1, 3], [2, 0]],
         [[1, 2], [3, 4], [5, 0]]])
    assert game_a + game_b == game_add
    assert game_b + game_a == game_add


def test_json():
    """Test json serialization"""
    jstr = json.dumps(_GAME.to_json())
    assert _GAME == aggfn.aggfn_json(json.loads(jstr))
