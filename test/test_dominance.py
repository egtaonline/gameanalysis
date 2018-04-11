"""Test dominance"""
from os import path

import numpy as np

from gameanalysis import dominance
from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import paygame


def test_weakly_dominated():
    """Test weak domination"""
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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.weakly_dominated(game)
    assert np.all(dom == [False, True])

    profiles = [
        [2, 0],
        [0, 2]
    ]
    payoffs = [
        [2, 0],
        [0, 2],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.weakly_dominated(game)
    assert np.all(dom == [False, False])

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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.weakly_dominated(game)
    assert np.all(dom == [False, True])

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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.weakly_dominated(game)
    assert np.all(dom == [True, True])


def test_weakly_dominated_conditional():
    """Test weak domination conditional"""
    profiles = [
        [0, 2],
        [1, 1],
    ]
    payoffs = [
        [0, 1],
        [1, 1],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.weakly_dominated(game)
    assert np.all(dom == [True, False])
    dom = dominance.weakly_dominated(game, conditional=False)
    assert np.all(dom == [True, True])


def test_strictly_dominated():
    """Test strict dominance"""
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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.strictly_dominated(game)
    assert np.all(dom == [False, True])

    profiles = [
        [2, 0],
        [0, 2]
    ]
    payoffs = [
        [2, 0],
        [0, 1],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.strictly_dominated(game)
    assert np.all(dom == [False, False])

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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.strictly_dominated(game)
    assert np.all(dom == [False, False])


def test_strictly_dominated_conditional():
    """Test strict domination conditional"""
    profiles = [
        [0, 2],
        [1, 1],
    ]
    payoffs = [
        [0, 1],
        [2, 1],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.strictly_dominated(game)
    assert np.all(dom == [False, False])
    dom = dominance.strictly_dominated(game, conditional=False)
    assert np.all(dom == [False, True])

    profiles = [
        [2, 0],
        [1, 1],
    ]
    payoffs = [
        [2, 0],
        [2, 1],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.strictly_dominated(game)
    assert np.all(dom == [False, True])

    profiles = [
        [2, 0],
        [1, 1],
    ]
    payoffs = [
        [2, 0],
        [2, 2],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.strictly_dominated(game)
    assert np.all(dom == [False, False])

    profiles = [
        [2, 0],
        [0, 2]
    ]
    payoffs = [
        [2, 0],
        [0, 1],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.strictly_dominated(game, conditional=False)
    assert np.all(dom == [False, False])


def test_never_best_response():
    """Test never best response"""
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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.never_best_response(game, conditional=False)
    assert np.all(dom == [False, False])

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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.never_best_response(game, conditional=False)
    assert np.all(dom == [False, False])

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
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.never_best_response(game, conditional=False)
    assert np.all(dom == [True, False])


def test_never_best_response_conditional():
    """Test never best response conditional"""
    profiles = [
        [2, 0],
        [0, 2],
    ]
    payoffs = [
        [1, 0],
        [0, 1],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.never_best_response(game, conditional=True)
    assert np.all(dom == [False, False])

    profiles = [
        [1, 1],
        [0, 2],
    ]
    payoffs = [
        [2, 2],
        [0, 3],
    ]
    game = paygame.game(2, 2, profiles, payoffs)
    dom = dominance.never_best_response(game, conditional=True)
    assert np.all(dom == [True, False])


def test_travellers_dilemma():
    """Test iterated elimination on travelers dilemma"""
    game = gamegen.travellers_dilemma(max_value=6)
    mask = dominance.iterated_elimination(game, 'weakdom')
    assert np.all(mask == [True] + [False] * 4)


def test_known_fail_case():
    """Test iterated elimination on hard game"""
    with open(path.join('example_games', 'hard_nash.json')) as fil:
        game = gamereader.load(fil)
    dominance.iterated_elimination(game, 'neverbr')
