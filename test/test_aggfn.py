import json

import numpy as np
import pytest

from gameanalysis import aggfn
from gameanalysis import paygame
from gameanalysis import rsgame


_game = aggfn.aggfn(
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
    base = rsgame.emptygame(players, strategies)
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
        ['f0', 'f2', 'f3'],
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
    assert _game.num_functions == 4
    assert _game.num_profiles == 9
    assert _game.is_complete()
    assert not _game.is_empty()
    assert repr(_game) == 'AgfnGame([2 1], [2 3], 4)'


def test_min_max():
    mins = [-2, 1, 0, 0, 0]
    assert np.all(mins == _game.min_strat_payoffs())
    maxs = [9, 13, 18, 25, 24]
    assert np.all(maxs == _game.max_strat_payoffs())


def test_get_payoffs():
    payoffs = [0, 3, 0, 5, 0]
    assert np.allclose(payoffs, _game.get_payoffs([1, 1, 0, 1, 0]))
    payoffs = [0, 6, 11, 0, 0]
    assert np.allclose(payoffs, _game.get_payoffs([0, 2, 1, 0, 0]))
    payoffs = [0, 3, 0, 6, 0]
    assert np.allclose(payoffs, _game.get_payoffs([0, 2, 0, 1, 0]))


def test_dev_pays():
    mix = [0.5, 0.5, 0.5, 0.5, 0]
    payoffs = [1, 4.5, 7.75, 6.5, 8.25]
    assert np.allclose(payoffs, _game.deviation_payoffs(mix))


def test_payoff_vals():
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
    copy = paygame.game_replace(_game, profiles, payoffs)
    assert paygame.game_copy(_game) == copy


def test_function_perm():
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
    perm = aggfn.aggfn(_game.num_role_players, _game.num_role_strats, actwp,
                       funcinpsp, functabp)
    assert perm == _game


def test_restrict():
    rest = [False, True, True, False, True]
    sfuncinps = [[False, True, True, False],
                 [True, False, True, True],
                 [True, False, False, True]]
    sactw = [[0, 1, 3],
             [1, 0, 0],
             [1, 1, 1],
             [0, 1, 1]]
    sgame = aggfn.aggfn_names(['r0', 'r1'], [2, 1], [['s1'], ['s2', 's4']],
                              _game.function_names, sactw, sfuncinps,
                              _game.function_table)
    assert sgame == _game.restrict(rest)


def test_to_from_json():
    jgame = {
        'players': {'r0': 2, 'r1': 1},
        'strategies': {'r0': ['s0', 's1'], 'r1': ['s2', 's3', 's4']},
        'action_weights': {
            'f0': {'r0': {'s0': -1},
                   'r1': {'s2': 1, 's3': 2, 's4': 3}},
            'f1': {'r0': {'s1': 1},
                   'r1': {'s3': 1}},
            'f2': {'r0': {'s0': 1, 's1': 1},
                   'r1': {'s2': 1, 's3': 1, 's4': 1}},
            'f3': {'r1': {'s2': 1, 's3': 1, 's4': 1}},
        },
        'function_inputs': {
            'f0': {'r0': ['s0'], 'r1': ['s2', 's4']},
            'f1': {'r0': ['s0', 's1']},
            'f2': {'r0': ['s0', 's1'], 'r1': ['s2']},
            'f3': {'r0': ['s0'], 'r1': ['s2', 's4']}},
        'function_tables': {
            'f0': [
                {'r0': 0, 'r1': 1, 'value': 1},
                {'r0': 1, 'r1': 0, 'value': 1},
                {'r0': 1, 'r1': 1, 'value': 3},
                {'r0': 2, 'r1': 0, 'value': 2},
            ],
            'f1': [
                {'r0': 1, 'r1': 0, 'value': 3},
                {'r0': 1, 'r1': 1, 'value': 2},
                {'r0': 2, 'r1': 0, 'value': 2},
                {'r0': 2, 'r1': 1, 'value': 4},
            ],
            'f2': [
                {'r0': 0, 'r1': 0, 'value': 9},
                {'r0': 0, 'r1': 1, 'value': 7},
                {'r0': 1, 'r1': 0, 'value': 4},
                {'r0': 1, 'r1': 1, 'value': 1},
                {'r0': 2, 'r1': 0, 'value': 1},
                {'r0': 2, 'r1': 1, 'value': 4},
            ],
            'f3': [
                {'r0': 0, 'r1': 0, 'value': 3},
                {'r0': 0, 'r1': 1, 'value': 6},
                {'r0': 1, 'r1': 1, 'value': -1},
                {'r0': 2, 'r1': 0, 'value': 3},
                {'r0': 2, 'r1': 1, 'value': 4},
            ],
        },
        'type': 'aggfn.2'}
    assert _game.to_json() == jgame
    assert json.loads(json.dumps(_game.to_json())) == jgame
    assert aggfn.aggfn_json(jgame) == _game


def test_from_json_v1():
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
    assert aggfn.aggfn_json(jgame) == _game


def test_from_json_vunnk():
    jgame = {
        'players': {'r0': 1},
        'strategies': {'r0': ['s0']},
        'function_tables': ['f0'],
        'function_inputs': {'f0': {}},
        'type': 'aggfn.unk'}
    with pytest.raises(AssertionError):
        aggfn.aggfn_json(jgame)


def test_from_json_sum():
    game = aggfn.aggfn(
        [1, 1], [2, 2],
        [[-1, 0, 1, 2]],
        [[True], [False], [False], [True]],
        [[[0, 1], [1, 3]]])
    jgame = {
        'players': {'r0': 1, 'r1': 1},
        'strategies': {'r0': ['s0', 's1'], 'r1': ['s2', 's3']},
        'action_weights': {
            'r0': {'s0': {'f0': -1}},
            'r1': {'s2': {'f0': 1},
                   's3': {'f0': 2}}},
        'function_inputs': {
            'f0': {'r0': ['s0'], 'r1': ['s3']}},
        'function_tables': {
            'f0': [0, 1, 3]},
        'type': 'aggfn.1'}
    assert aggfn.aggfn_json(jgame) == game


def test_from_json_const():
    game = aggfn.aggfn(
        [1, 1], [2, 2],
        [[-1, 0, 1, 2]],
        [[True], [False], [False], [True]],
        [[[0, 1], [1, 3]]],
        [4, 2, 1, 6])
    jgame = {
        'players': {'r0': 1, 'r1': 1},
        'strategies': {'r0': ['s0', 's1'], 'r1': ['s2', 's3']},
        'action_weights': {
            'r0': {'s0': {'f0': -1, 'f3': 4},
                   's1': {'f1': 2}},
            'r1': {'s2': {'f0': 1, 'f1': 1},
                   's3': {'f0': 2, 'f2': 3}}},
        'function_inputs': {
            'f0': {'r0': ['s0'], 'r1': ['s3']},
            'f1': {'r0': ['s1'], 'r1': ['s2']},
            'f2': {},
            'f3': {'r0': ['s0', 's1'], 'r1': ['s2', 's3']}},
        'function_tables': {
            'f0': [0, 1, 3],
            'f1': [1, 1, 1],
            'f2': [2, 3, 4],
            'f3': [3, 2, 1]},
        'type': 'aggfn.1'}
    assert aggfn.aggfn_json(jgame) == game


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
    game = aggfn.aggfn(2, 2, [[1, 2]], [[True], [False]], [[0, 1, 2]])
    game2 = aggfn.aggfn_replace(game, [[1, 2]], [[True], [False]], [[0, 1, 2]])
    assert game == game2


def test_from_function():
    game = aggfn.aggfn_funcs(2, 2, [[1, 2]], [[True], [False]], [lambda x: x])
    assert len(game.function_table.shape) == 2
    game = aggfn.aggfn_funcs(
        [2, 2], 2, [[1, 2, 3, 4]], [[True], [False], [False], [True]],
        [lambda x, y: x + y])
    assert len(game.function_table.shape) == 3


def test_from_function_call():

    class func:
        def __call__(self, x):
            return x

    game = aggfn.aggfn_funcs(2, 2, [[1, 2]],
                             [[True], [False]], [func()])
    assert len(game.function_table.shape) == 2

    class func:
        def __call__(self, x, y):
            return x + y

    game = aggfn.aggfn_funcs([2, 2], 2, [[1, 2, 3, 4]],
                             [[True], [False], [False], [True]], [func()])
    assert len(game.function_table.shape) == 3


def test_repr():
    expected = 'AgfnGame([5], [4], 3)'
    assert repr(rand(5, 4, 3)) == expected

    expected = 'AgfnGame([5 4], [4 3], 3)'
    assert repr(rand([5, 4], [4, 3], 3)) == expected


def test_function_index():
    for i in range(_game.num_functions):
        assert _game.function_index('f{:d}'.format(i)) == i


def test_eq():
    copy = aggfn.aggfn_replace(
        _game, _game.action_weights, _game.function_inputs,
        _game.function_table, np.ones(_game.num_strats))
    assert copy != _game
