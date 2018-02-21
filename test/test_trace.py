import numpy as np
import pytest

from gameanalysis import agggen
from gameanalysis import mergegame
from gameanalysis import nash
from gameanalysis import paygame
from gameanalysis import regret
from gameanalysis import trace
from test import utils


# This tests the edge case for non-differentiability of the equilibrium is
# sound and recovered.
def test_trace_equilibria():
    profs = [[2, 0],
             [1, 1],
             [0, 2]]
    pays1 = [[1, 0],
             [1, 0],
             [0, 0]]
    game1 = paygame.game(2, 2, profs, pays1)
    pays2 = [[0, 0],
             [0, 1],
             [0, 1]]
    game2 = paygame.game(2, 2, profs, pays2)

    ts, mixes = trace.trace_equilibria(game1, game2, 0, [1, 0])
    assert np.isclose(ts[0], 0)
    assert np.isclose(ts[-1], 0.5, atol=1e-4)
    assert np.allclose(mixes, [1, 0])
    ts, mixes = trace.trace_equilibria(game1, game2, 1, [0, 1])
    assert np.isclose(ts[0], 0.5, atol=1e-4)
    assert np.isclose(ts[-1], 1)
    assert np.allclose(mixes, [0, 1])


@pytest.mark.parametrize('players,strats', utils.games)
def test_random_trace_equilibria(players, strats):
    game1 = agggen.normal_aggfn(players, strats, 6)
    game2 = agggen.normal_aggfn(players, strats, 6)

    eqa = game1.trim_mixture_support(nash.mixed_nash(game1), thresh=1e-5)
    for eqm in eqa:
        # leeway for support trimming
        thresh = regret.mixture_regret(game1, eqm) + 2
        ts, mixes = trace.trace_equilibria(game1, game2, 0, eqm)
        for t, mix in zip(ts, mixes):
            reg = regret.mixture_regret(mergegame.merge(game1, game2, t), mix)
            assert reg < thresh

    eqa = game2.trim_mixture_support(nash.mixed_nash(game2), thresh=1e-5)
    for eqm in eqa:
        # leeway for support trimming
        thresh = regret.mixture_regret(game2, eqm) + 1
        ts, mixes = trace.trace_equilibria(game1, game2, 1, eqm)
        for t, mix in zip(ts, mixes):
            reg = regret.mixture_regret(mergegame.merge(game1, game2, t), mix)
            assert reg < thresh
