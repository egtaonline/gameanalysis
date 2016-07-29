import json

import numpy as np

from gameanalysis import dominance
from gameanalysis import gamegen
from gameanalysis import gameio
from gameanalysis import rsgame


def test_weakly_dominated():
    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 0],
        [2, 1],
        [0, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.weakly_dominated(game)
    assert np.all(wd == [False, True])

    profiles = [
        [2, 0],
        [0, 2]
    ]
    payoffs = [
        [2, 0],
        [0, 2],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.weakly_dominated(game)
    assert np.all(wd == [False, False])

    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 0],
        [2, 1],
        [0, 2],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.weakly_dominated(game)
    assert np.all(wd == [False, True])

    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 0],
        [2, 2],
        [0, 2],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.weakly_dominated(game)
    assert np.all(wd == [True, True])


def test_weakly_dominated_conditional():
    profiles = [
        [0, 2],
        [1, 1],
    ]
    payoffs = [
        [0, 1],
        [1, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.weakly_dominated(game)
    assert np.all(wd == [True, False])
    wd = dominance.weakly_dominated(game, False)
    assert np.all(wd == [True, True])


def test_strictly_dominated():
    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 0],
        [2, 1],
        [0, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.strictly_dominated(game)
    assert np.all(wd == [False, True])

    profiles = [
        [2, 0],
        [0, 2]
    ]
    payoffs = [
        [2, 0],
        [0, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.strictly_dominated(game)
    assert np.all(wd == [False, False])

    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 0],
        [2, 2],
        [0, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.strictly_dominated(game)
    assert np.all(wd == [False, False])


def test_strictly_dominated_conditional():
    profiles = [
        [0, 2],
        [1, 1],
    ]
    payoffs = [
        [0, 1],
        [2, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.strictly_dominated(game)
    assert np.all(wd == [False, False])
    wd = dominance.strictly_dominated(game, False)
    assert np.all(wd == [False, True])

    profiles = [
        [2, 0],
        [1, 1],
    ]
    payoffs = [
        [2, 0],
        [2, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.strictly_dominated(game)
    assert np.all(wd == [False, True])

    profiles = [
        [2, 0],
        [1, 1],
    ]
    payoffs = [
        [2, 0],
        [2, 2],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.strictly_dominated(game)
    assert np.all(wd == [False, False])

    profiles = [
        [2, 0],
        [0, 2]
    ]
    payoffs = [
        [2, 0],
        [0, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.strictly_dominated(game, False)
    assert np.all(wd == [False, False])


def test_never_best_response():
    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [1, 0],
        [2, 2],
        [0, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.never_best_response(game, False)
    assert np.all(wd == [False, False])

    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 0],
        [2, 2],
        [0, 2],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.never_best_response(game, False)
    assert np.all(wd == [False, False])

    profiles = [
        [2, 0],
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [1, 0],
        [2, 2],
        [0, 3],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.never_best_response(game, False)
    assert np.all(wd == [True, False])


def test_never_best_response_conditional():
    profiles = [
        [2, 0],
        [0, 2],
    ]
    payoffs = [
        [1, 0],
        [0, 1],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.never_best_response(game, True)
    assert np.all(wd == [False, False])

    profiles = [
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 2],
        [0, 3],
    ]
    game = rsgame.Game(2, 2, profiles, payoffs)
    wd = dominance.never_best_response(game, True)
    assert np.all(wd == [True, False])


def test_travellers_dilemma():
    game = gamegen.travellers_dilemma(max_value=6)
    mask = dominance.iterated_elimination(game, 'weakdom')
    assert np.all(mask == [True] + [False] * 4)


def test_known_fail_case():
    with open('test/hard_nash_game_1.json') as f:
        game, _ = gameio.read_game(json.load(f))
    dominance.iterated_elimination(game, 'neverbr')
