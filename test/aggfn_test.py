import numpy as np
import pytest

from gameanalysis import aggfn
from gameanalysis import agggen
from gameanalysis import nash
from gameanalysis import paygame


_sum_game = aggfn.aggfn(
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
    [[0, 1, 2, 3],
     [0, 3, 2, 1],
     [9, 4, 1, 0],
     [3, 0, 0, 3]])

_role_game = aggfn.aggfn(
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


def test_sum_aggfn_prop():
    game = _sum_game
    assert game.num_functions == 4
    assert game.num_profiles == 9
    assert game.is_complete()
    assert not game.is_empty()
    assert repr(game) == 'SumAgfnGame([2 1], [2 3], 4)'


def test_sum_aggfn_min_max():
    game = _sum_game
    mins = [-3, 0, 0, 0, 0]
    assert np.all(mins == game.min_strat_payoffs())
    maxs = [9, 12, 15, 21, 21]
    assert np.all(maxs == game.max_strat_payoffs())


def test_sum_aggfn_payoffs():
    game = _sum_game
    payoffs = [0, 3, 0, 5, 0]
    assert np.allclose(payoffs, game.get_payoffs([1, 1, 0, 1, 0]))
    payoffs = [0, 2, 1, 0, 0]
    assert np.allclose(payoffs, game.get_payoffs([0, 2, 1, 0, 0]))
    payoffs = [0, 3, 0, 6, 0]
    assert np.allclose(payoffs, game.get_payoffs([0, 2, 0, 1, 0]))


def test_sum_aggfn_dev_pays():
    game = _sum_game
    mix = [0.5, 0.5, 0.5, 0.5, 0]
    payoffs = [-1.5, 2.5, 2.75, 5.75, 7.75]
    assert np.allclose(payoffs, game.deviation_payoffs(mix))


def test_sum_aggfn_payoff_vals():
    game = _sum_game
    profiles = [[2, 0, 1, 0, 0],
                [2, 0, 0, 1, 0],
                [2, 0, 0, 0, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
                [0, 2, 1, 0, 0],
                [0, 2, 0, 1, 0],
                [0, 2, 0, 0, 1]]
    payoffs = [[-3, 0, 6, 0, 0],
               [-1, 0, 0, 7, 0],
               [-2, 0, 0, 0, 13],
               [-2, 2, 2, 0, 0],
               [0, 3, 0, 5, 0],
               [-1, 3, 0, 0, 7],
               [0, 2, 1, 0, 0],
               [0, 3, 0, 6, 0],
               [0, 3, 0, 0, 4]]
    copy = paygame.game_replace(game, profiles, payoffs)
    assert paygame.game_copy(game) == copy


def test_sum_aggfn_function_permutation():
    game = _sum_game
    functabp = [[9, 4, 1, 0],
                [0, 3, 2, 1],
                [0, 1, 2, 3],
                [3, 0, 0, 3]]
    funcinpsp = [[True, True, True, True],
                 [True, True, False, False],
                 [True, False, True, True],
                 [False, False, False, False],
                 [False, False, True, True]]
    actwp = [[1, 1, 1, 1, 1],
             [0, 1, 0, 1, 0],
             [-1, 0, 1, 2, 3],
             [0, 0, 1, 1, 1]]
    perm = aggfn.aggfn(game.num_role_players, game.num_role_strats, actwp,
                       funcinpsp, functabp)
    assert perm == game


def test_sum_aggfn_subgame():
    game = _sum_game
    mask = [True, False, True, True, False]
    sfuncinps = [[True, True, True, True],
                 [True, False, True, True],
                 [False, False, False, False]]
    sactw = [[-1, 1, 2],
             [0, 0, 1],
             [1, 1, 1],
             [0, 1, 1]]
    sgame = aggfn.aggfn_names(['r0', 'r1'], [2, 1], [['s0'], ['s2', 's3']],
                              game.function_names, sactw, sfuncinps,
                              game._function_table)
    assert sgame == game.subgame(mask)


def test_subgame_function_removal():
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
        [[0, 1, 2, 3],
         [0, 3, 2, 1],
         [9, 4, 1, 0],
         [3, 0, 0, 3]])
    sub = game.subgame([True, True, True, False, True])
    expected = aggfn.aggfn_names(
        ['r0', 'r1'], [2, 1], [['s0', 's1'], ['s2', 's4']],
        ['f0', 'f2', 'f3'],
        [[-1, 0, 1, 3],
         [1, 1, 1, 1],
         [0, 0, 1, 1]],
        [[True, True, True],
         [False, True, False],
         [True, True, True],
         [True, False, True]],
        [[0, 1, 2, 3],
         [9, 4, 1, 0],
         [3, 0, 0, 3]])
    assert sub == expected


def test_sum_aggfn_to_json():
    game = _sum_game
    jgame = {
        'players': {'r0': 2, 'r1': 1},
        'strategies': {'r0': ['s0', 's1'], 'r1': ['s2', 's3', 's4']},
        'action_weights': {
            'r0': {'s0': {'f0': -1, 'f2': 1},
                   's1': {'f1': 1, 'f2': 1}},
            'r1': {'s2': {'f0': 1, 'f2': 1, 'f3': 1},
                   's3': {'f0': 2, 'f1': 1, 'f2': 1, 'f3': 1},
                   's4': {'f0': 3, 'f2': 1, 'f3': 1}}},
        'function_inputs': {
            'f0': {'r0': ['s0'], 'r1': ['s2', 's4']},
            'f1': {'r0': ['s0', 's1']},
            'f2': {'r0': ['s0', 's1'], 'r1': ['s2']},
            'f3': {'r0': ['s0'], 'r1': ['s2', 's4']}},
        'function_tables': {
            'f0': [0, 1, 2, 3],
            'f1': [0, 3, 2, 1],
            'f2': [9, 4, 1, 0],
            'f3': [3, 0, 0, 3]},
        'type': 'aggfn.1'}
    assert game.to_json() == jgame
    assert aggfn.aggfn_json(jgame) == game


def test_role_aggfn_props():
    game = _role_game
    assert game.num_functions == 4
    assert game.num_profiles == 9
    assert game.is_complete()
    assert not game.is_empty()
    assert repr(game) == 'RoleAgfnGame([2 1], [2 3], 4)'


def test_role_aggfn_min_max():
    game = _role_game
    mins = [-2, 1, 0, 0, 0]
    assert np.all(mins == game.min_strat_payoffs())
    maxs = [9, 13, 18, 25, 24]
    assert np.all(maxs == game.max_strat_payoffs())


def test_role_aggfn_get_payoffs():
    game = _role_game
    payoffs = [0, 3, 0, 5, 0]
    assert np.allclose(payoffs, game.get_payoffs([1, 1, 0, 1, 0]))
    payoffs = [0, 6, 11, 0, 0]
    assert np.allclose(payoffs, game.get_payoffs([0, 2, 1, 0, 0]))
    payoffs = [0, 3, 0, 6, 0]
    assert np.allclose(payoffs, game.get_payoffs([0, 2, 0, 1, 0]))


def test_role_aggfn_dev_pays():
    game = _role_game
    mix = [0.5, 0.5, 0.5, 0.5, 0]
    payoffs = [1, 4.5, 7.75, 6.5, 8.25]
    assert np.allclose(payoffs, game.deviation_payoffs(mix))


def test_role_aggfn_payoff_vals():
    game = _role_game
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
    copy = paygame.game_replace(game, profiles, payoffs)
    assert paygame.game_copy(game) == copy


def test_role_aggfn_function_perm():
    game = _role_game
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
    perm = aggfn.aggfn(game.num_role_players, game.num_role_strats, actwp,
                       funcinpsp, functabp)
    assert perm == game


def test_role_aggfn_subgame():
    game = _role_game
    mask = [False, True, True, False, True]
    sfuncinps = [[False, True, True, False],
                 [True, False, True, True],
                 [True, False, False, True]]
    sactw = [[0, 1, 3],
             [1, 0, 0],
             [1, 1, 1],
             [0, 1, 1]]
    sgame = aggfn.aggfn_names(['r0', 'r1'], [2, 1], [['s1'], ['s2', 's4']],
                              game.function_names, sactw, sfuncinps,
                              game._function_table)
    assert sgame == game.subgame(mask)


def test_role_aggfn_to_json():
    game = _role_game
    jgame = {
        'players': {'r0': 2, 'r1': 1},
        'strategies': {'r0': ['s0', 's1'], 'r1': ['s2', 's3', 's4']},
        'action_weights': {
            'r0': {'s0': {'f0': -1, 'f2': 1},
                   's1': {'f1': 1, 'f2': 1}},
            'r1': {'s2': {'f0': 1, 'f2': 1, 'f3': 1},
                   's3': {'f0': 2, 'f1': 1, 'f2': 1, 'f3': 1},
                   's4': {'f0': 3, 'f2': 1, 'f3': 1}}},
        'function_inputs': {
            'f0': {'r0': ['s0'], 'r1': ['s2', 's4']},
            'f1': {'r0': ['s0', 's1']},
            'f2': {'r0': ['s0', 's1'], 'r1': ['s2']},
            'f3': {'r0': ['s0'], 'r1': ['s2', 's4']}},
        'function_tables': {
            'f0': [[0, 1], [1, 3], [2, 0]],
            'f1': [[0, 0], [3, 2], [2, 4]],
            'f2': [[9, 7], [4, 1], [1, 4]],
            'f3': [[3, 6], [0, -1], [3, 4]]},
        'type': 'aggfn.1'}
    assert game.to_json() == jgame
    assert aggfn.aggfn_json(jgame) == game


def test_normalize_const_sum_aggfn():
    game = aggfn.aggfn(
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
        [[0, 1, 2, 3],
         [0, 3, 2, 1],
         [9, 4, 1, 0],
         [2, 2, 2, 2]]).normalize()
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.allclose(game.max_role_payoffs(), 1)


def test_normalize_const_role_aggfn():
    game = aggfn.aggfn(
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
         [[2, 2], [2, 2], [2, 2]]]).normalize()
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.allclose(game.max_role_payoffs(), 1)


def test_normalize_always_sum_aggfn():
    game = aggfn.aggfn(
        [2, 1], [2, 3],
        [[-1, 0, 1, 2, 3],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1]],
        [[True, True, True, True],
         [False, True, True, True],
         [True, False, True, True],
         [False, False, False, True],
         [True, False, False, True]],
        [[0, 1, 2, 3],
         [0, 3, 2, 1],
         [9, 4, 1, 0],
         [3, 0, 0, 3]]).normalize()
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.allclose(game.max_role_payoffs(), 1)


def test_normalize_never_sum_aggfn():
    game = aggfn.aggfn(
        [2, 1], [2, 3],
        [[-1, 0, 1, 2, 3],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1]],
        [[True, True, True, False],
         [False, True, True, False],
         [True, False, True, False],
         [False, False, False, False],
         [True, False, False, False]],
        [[0, 1, 2, 3],
         [0, 3, 2, 1],
         [9, 4, 1, 0],
         [3, 0, 0, 3]]).normalize()
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.allclose(game.max_role_payoffs(), 1)


def test_normalize_always_role_aggfn():
    game = aggfn.aggfn(
        [2, 1], [2, 3],
        [[-1, 0, 1, 2, 3],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1]],
        [[True, True, True, True],
         [False, True, True, True],
         [True, False, True, True],
         [False, False, False, True],
         [True, False, False, True]],
        [[[0, 1], [1, 3], [2, 0]],
         [[0, 0], [3, 2], [2, 4]],
         [[9, 7], [4, 1], [1, 4]],
         [[3, 6], [0, -1], [3, 4]]]).normalize()
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.allclose(game.max_role_payoffs(), 1)


def test_normalize_never_role_aggfn():
    game = aggfn.aggfn(
        [2, 1], [2, 3],
        [[-1, 0, 1, 2, 3],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 1, 1]],
        [[True, True, True, False],
         [False, True, True, False],
         [True, False, True, False],
         [False, False, False, False],
         [True, False, False, False]],
        [[[0, 1], [1, 3], [2, 0]],
         [[0, 0], [3, 2], [2, 4]],
         [[9, 7], [4, 1], [1, 4]],
         [[3, 6], [0, -1], [3, 4]]]).normalize()
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.allclose(game.max_role_payoffs(), 1)


def verify_aggfn(game):
    """Verify that aggfn matches the expanded version"""
    payoff_game = paygame.game_copy(game)
    assert not game.is_empty()
    assert game.is_complete()

    ngame = game.normalize()
    assert np.all(
        np.isclose(ngame.max_role_payoffs() - ngame.min_role_payoffs(), 1) |
        (ngame.num_role_strats == 1))
    assert np.allclose(ngame.min_role_payoffs(), 0)
    assert np.all(
        np.isclose(ngame.max_role_payoffs(), 1) |
        (ngame.num_role_strats == 1))

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
        dev = game.deviation_payoffs(mix)
        tdev = payoff_game.deviation_payoffs(mix)
        assert np.allclose(dev, tdev)

        dev, jac = game.deviation_payoffs(mix, jacobian=True)
        tdev, tjac = payoff_game.deviation_payoffs(mix, jacobian=True)
        assert np.allclose(dev, tdev)
        assert np.allclose(jac, tjac)

    # Check that sparse mixtures don't result in nans since we don't have
    # ground truth
    for mix in game.random_sparse_mixtures(20):
        dev, jac = game.deviation_payoffs(mix, jacobian=True)
        assert not np.isnan(dev).any()
        assert not np.isnan(jac).any()


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
def test_random_sum_game(players, strategies, functions):
    """Test that deviation payoff formulation is accurate"""
    game = agggen.random_aggfn(players, strategies, functions, by_role=False)
    verify_aggfn(game)


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
def test_random_role_game(players, strategies, functions):
    """Test that deviation payoff formulation is accurate"""
    game = agggen.random_aggfn(players, strategies, functions, by_role=True)
    verify_aggfn(game)


@pytest.mark.parametrize('players,strategies,functions', [
    ([1, 2], 2, 3),
])
@pytest.mark.parametrize('by_role', [False, True])
def test_nash_finding(players, strategies, functions, by_role):
    # This distribution tends to have `easier` nash equilibria, and we only
    # want to test that it functionally works.
    dist = agggen.random_poly_dist([0.1, 0.8, 0.1])
    game = agggen.random_aggfn(
        players, strategies, functions, func_dist=dist, by_role=by_role)
    eqa = nash.mixed_nash(game)
    assert eqa.size > 0, "didn't find any equilibria"


def test_alternate_constructors():
    game = aggfn.aggfn(2, 2, [[1, 2]], [[True], [True]], [[0, 1, 2]])
    game2 = aggfn.aggfn_replace(game, [[1, 2]], [[True], [True]], [[0, 1, 2]])
    assert game == game2


def test_from_function():
    game = aggfn.aggfn_funcs([2, 2], 2, [[1, 2, 3, 4]],
                             [[True], [True], [True], [True]], [lambda x: x])
    assert len(game._function_table.shape) == 2
    game = aggfn.aggfn_funcs(
        [2, 2], 2, [[1, 2, 3, 4]], [[True], [True], [True], [True]],
        [lambda x, y: x + y])
    assert len(game._function_table.shape) == 3


@pytest.mark.parametrize('by_role', [False, True])
def test_serialization(by_role):
    game = agggen.random_aggfn([5, 4], [4, 3], 3, by_role=by_role)
    jgame = game.to_json()
    copy = aggfn.aggfn_json(jgame)
    assert game == copy


def test_aggfn_repr():
    game = agggen.random_aggfn(5, 4, 3)
    expected = 'RoleAgfnGame([5], [4], 3)'
    assert repr(game) == expected

    game = agggen.random_aggfn([5, 4], [4, 3], 3)
    expected = 'SumAgfnGame([5 4], [4 3], 3)'
    assert repr(game) == expected


def test_function_index():
    game = _sum_game
    for i in range(game.num_functions):
        assert game.function_index('f{:d}'.format(i)) == i
