import json

import pytest

from gameanalysis import aggfn
from gameanalysis import agggen


_sizes = [
    (2 * [1], 2, 2),
    (2 * [2], 2, 2),
    (5 * [1], 2, 3),
    (2 * [1], 5, 3),
    (2 * [2], 5, 4),
    ([1, 2], 2, 5),
    ([1, 2], [2, 1], 4),
    (2, [1, 2], 4),
    ([3, 4], [2, 3], 6),
]
_probs = [
    (0.2, 0.2),
    (0.5, 0.5),
    (0.9, 0.9),
    (0.3, 0.8),
    (0.8, 0.3),
]


def verify(game):
    jgame = json.dumps(game.to_json())
    copy = aggfn.aggfn_json(json.loads(jgame))
    assert game == copy


@pytest.mark.parametrize('players,strategies,functions', _sizes)
@pytest.mark.parametrize('inp,weight', _probs)
def test_normal_game(players, strategies, functions, inp, weight):
    verify(agggen.normal_aggfn(players, strategies, functions))
    verify(agggen.normal_aggfn(players, strategies, functions, input_prob=inp,
                               weight_prob=weight))


@pytest.mark.parametrize('players,strategies,functions', _sizes)
@pytest.mark.parametrize('inp,weight', _probs)
@pytest.mark.parametrize('deg', [1, 2, 4, [0.1, 0.5, 0.4], [0, 0, 1]])
def test_random_poly_game(players, strategies, functions, inp, weight, deg):
    verify(agggen.poly_aggfn(players, strategies, functions))
    verify(agggen.poly_aggfn(players, strategies, functions, input_prob=inp,
                             weight_prob=weight, degree=deg))


@pytest.mark.parametrize('players,strategies,functions', _sizes)
@pytest.mark.parametrize('inp,weight', _probs)
@pytest.mark.parametrize('period', [0.5, 1, 2, 10])
def test_random_sine_game(players, strategies, functions, inp, weight, period):
    verify(agggen.sine_aggfn(players, strategies, functions))
    verify(agggen.sine_aggfn(players, strategies, functions, input_prob=inp,
                             weight_prob=weight, period=period))


@pytest.mark.parametrize('players,facilities,required', [
    (2, 2, 1),
    (2, 3, 2),
    (3, 4, 2),
    (4, 3, 2),
    (5, 6, 4),
])
@pytest.mark.parametrize('deg', [1, 2, 4])
def test_congestion_game(players, facilities, required, deg):
    game = agggen.congestion(players, facilities, required)
    verify(game)
    # Check that function serial has the right format
    assert all(sum(c == '_' for c in strat) == required - 1
               for strat in game.strat_names[0])

    game = agggen.congestion(players, facilities, required, degree=deg)
    verify(game)
    # Check that function serial has the right format
    assert all(sum(c == '_' for c in strat) == required - 1
               for strat in game.strat_names[0])


@pytest.mark.parametrize('players,strategies', [
    (2, 2),
    (2, 2),
    (2, 3),
    (3, 4),
    (4, 3),
    (5, 6),
])
@pytest.mark.parametrize('prob', [0, 0.1, 0.5, 0.9, 1])
def test_local_effect_game(players, strategies, prob):
    """Test that deviation payoff formulation is accurate"""
    verify(agggen.local_effect(players, strategies))
    verify(agggen.local_effect(players, strategies, edge_prob=prob))
