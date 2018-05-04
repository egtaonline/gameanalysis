"""Test trace"""
import numpy as np
import pytest

from gameanalysis import gamegen
from gameanalysis import nash
from gameanalysis import paygame
from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import trace
from test import utils # pylint: disable=wrong-import-order


# This tests the edge case for non-differentiability of the equilibrium is
# sound and recovered.
def test_trace_equilibria():
    """Test trace known game equilibrium"""
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

    probs, mixes = trace.trace_equilibrium(game0, game1, 0, [1, 0], 1)
    assert np.isclose(probs[0], 0)
    assert np.isclose(probs[-1], 0.5, atol=1.1e-3)
    assert np.allclose(mixes, [1, 0])
    probs, mixes = trace.trace_equilibrium(game0, game1, 1, [0, 1], 0)
    assert np.isclose(probs[0], 1)
    assert np.isclose(probs[-1], 0.5, atol=1.1e-3)
    assert np.allclose(mixes, [0, 1])


@pytest.mark.parametrize('players,strats', utils.GAMES)
def test_random_trace_equilibria(players, strats):
    """Test random equilibrium trace"""
    game0 = gamegen.normal_aggfn(players, strats, 6)
    game1 = gamegen.normal_aggfn(players, strats, 6)

    eqa = game0.trim_mixture_support(nash.mixed_nash(
        game0, regret_thresh=1e-4))
    for eqm in eqa:
        if regret.mixture_regret(game0, eqm) > 1e-3:
            # trimmed equilibrium had too high of regret...
            continue  # pragma: no cover
        probs, mixes = trace.trace_equilibrium(game0, game1, 0, eqm, 1)
        for prob, mix in zip(probs, mixes):
            reg = regret.mixture_regret(rsgame.mix(game0, game1, prob), mix)
            assert reg <= 1.1e-3

    eqa = game1.trim_mixture_support(nash.mixed_nash(
        game1, regret_thresh=1e-4))
    for eqm in eqa:
        if regret.mixture_regret(game1, eqm) > 1e-3:
            # trimmed equilibrium had too high of regret...
            continue  # pragma: no cover
        probs, mixes = trace.trace_equilibrium(game0, game1, 1, eqm, 0)
        for prob, mix in zip(probs, mixes):
            reg = regret.mixture_regret(rsgame.mix(game0, game1, prob), mix)
            assert reg <= 1.1e-3


MIX = gamegen.sym_2p2s_known_eq(.5)
DOM1 = paygame.game(
    2, 2, [[2, 0], [1, 1], [0, 2]], [[.1, 0], [.1, 0], [0, 0]])
DOM2 = paygame.game(
    2, 2, [[2, 0], [1, 1], [0, 2]], [[0, 0], [0, .1], [0, .1]])
OTHERS = [
    (gamegen.normal_aggfn(play, strt, 6), gamegen.normal_aggfn(play, strt, 6))
    for play, strt in utils.GAMES]


@pytest.mark.parametrize('game0,game1', [
    (MIX, DOM1),
    (DOM1, MIX),
    (MIX, DOM2),
    (DOM2, MIX),
] + OTHERS)
@pytest.mark.parametrize('_', range(3))
def test_random_trace_interpolate(game0, game1, _): # pylint: disable=too-many-locals
    """Test random trace interpolation"""
    prob = np.random.random()
    eqa = game0.trim_mixture_support(nash.mixed_nash(
        rsgame.mix(game0, game1, prob),
        regret_thresh=1e-4))
    for eqm in eqa:
        if regret.mixture_regret(rsgame.mix(game0, game1, prob), eqm) > 1e-3:
            # trimmed equilibrium had too high of regret...
            continue  # pragma: no cover

        for target in [0, 1]:
            # Test that interpolate recovers missing equilibria
            probs, mixes = trace.trace_equilibrium(
                game0, game1, prob, eqm, target)
            if probs.size < 3:
                # not enough to test leave one out
                continue # pragma: no cover

            start, interp, end = np.sort(np.random.choice(
                probs.size, 3, replace=False))
            interp_mix, = trace.trace_interpolate(
                game0, game1, [probs[start], probs[end]],
                [mixes[start], mixes[end]], [probs[interp]])
            assert np.allclose(interp_mix, mixes[interp], rtol=1e-2, atol=1e-4)

            # Test interp at first
            mix, = trace.trace_interpolate(
                game0, game1, probs, mixes, [probs[0]])
            assert np.allclose(mix, mixes[0], rtol=1e-2, atol=1e-4)

            # Test interp at last
            mix, = trace.trace_interpolate(
                game0, game1, probs, mixes, [probs[-1]])
            assert np.allclose(mix, mixes[-1], rtol=1e-2, atol=1e-4)

            # Test random t
            p_interp = np.random.uniform(probs[0], probs[-1])
            mix, = trace.trace_interpolate(
                game0, game1, probs, mixes, [p_interp])
            assert regret.mixture_regret(rsgame.mix(
                game0, game1, p_interp), mix) <= 1.1e-3
