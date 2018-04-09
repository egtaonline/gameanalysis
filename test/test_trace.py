import numpy as np
import pytest

from gameanalysis import agggen
from gameanalysis import nash
from gameanalysis import paygame
from gameanalysis import regret
from gameanalysis import rsgame
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
    game0 = paygame.game(2, 2, profs, pays1)
    pays2 = [[0, 0],
             [0, 1],
             [0, 1]]
    game1 = paygame.game(2, 2, profs, pays2)

    ts, mixes = trace.trace_equilibria(game0, game1, 0, [1, 0])
    assert np.isclose(ts[0], 0)
    assert np.isclose(ts[-1], 0.5, atol=1.1e-3)
    assert np.allclose(mixes, [1, 0])
    ts, mixes = trace.trace_equilibria(game0, game1, 1, [0, 1])
    assert np.isclose(ts[0], 0.5, atol=1.1e-3)
    assert np.isclose(ts[-1], 1)
    assert np.allclose(mixes, [0, 1])


@pytest.mark.parametrize('players,strats', utils.games)
def test_random_trace_equilibria(players, strats):
    game0 = agggen.normal_aggfn(players, strats, 6)
    game1 = agggen.normal_aggfn(players, strats, 6)

    eqa = game0.trim_mixture_support(nash.mixed_nash(
        game0, regret_thresh=1e-4))
    for eqm in eqa:
        if 1e-3 < regret.mixture_regret(game0, eqm):
            # trimmed equilibrium had too high of regret...
            continue  # pragma: no cover
        ts, mixes = trace.trace_equilibria(game0, game1, 0, eqm)
        for t, mix in zip(ts, mixes):
            reg = regret.mixture_regret(rsgame.mix(game0, game1, t), mix)
            assert reg <= 1.1e-3

    eqa = game1.trim_mixture_support(nash.mixed_nash(
        game1, regret_thresh=1e-4))
    for eqm in eqa:
        if 1e-3 < regret.mixture_regret(game1, eqm):
            # trimmed equilibrium had too high of regret...
            continue  # pragma: no cover
        ts, mixes = trace.trace_equilibria(game0, game1, 1, eqm)
        for t, mix in zip(ts, mixes):
            reg = regret.mixture_regret(rsgame.mix(game0, game1, t), mix)
            assert reg <= 1.1e-3


@pytest.mark.parametrize('players,strats', utils.games)
def test_random_trace_interpolate(players, strats):
    game0 = agggen.normal_aggfn(players, strats, 6)
    game1 = agggen.normal_aggfn(players, strats, 6)

    t = np.random.random()
    eqa = game0.trim_mixture_support(nash.mixed_nash(
        rsgame.mix(game0, game1, t),
        regret_thresh=1e-4))
    for eqm in eqa:
        if 1e-3 < regret.mixture_regret(rsgame.mix(game0, game1, t), eqm):
            # trimmed equilibrium had too high of regret...
            continue  # pragma: no cover

        # Test that interp reovers missing equilibria
        ts, mixes = trace.trace_equilibria(game0, game1, t, eqm)
        start, interp, end = np.sort(np.random.choice(
            ts.size, 3, replace=False))
        interp_mix = trace.trace_interpolate(
            game0, game1, [ts[start], ts[end]], [mixes[start], mixes[end]],
            ts[interp])
        assert np.allclose(interp_mix, mixes[interp], rtol=1e-3, atol=1e-5)

        # Test interp at first
        mix = trace.trace_interpolate(
            game0, game1, ts, mixes, ts[0])
        assert np.allclose(mix, mixes[0])

        # Test interp at last
        mix = trace.trace_interpolate(
            game0, game1, ts, mixes, ts[-1])
        assert np.allclose(mix, mixes[-1])

        # Test random t
        t_interp = np.random.uniform(ts[0], ts[-1])
        mix = trace.trace_interpolate(
            game0, game1, ts, mixes, t_interp)
        assert regret.mixture_regret(rsgame.mix(
            game0, game1, t_interp), mix) <= 1.1e-3
