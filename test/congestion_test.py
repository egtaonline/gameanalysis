import numpy as np

from gameanalysis import nash
from gameanalysis import congestion

from test import testutils


@testutils.apply([
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
], repeat=20)
def test_deviation_payoffs(players, facilities, required):
    """Test that deviation payoff formulation is accurate"""
    cgame = congestion.CongestionGame(players, facilities, required)
    game = cgame.to_game()
    mixes = game.random_mixtures(20)

    for mix in mixes:
        dev, jac = cgame.deviation_payoffs(mix, jacobian=True)
        tdev, tjac = game.deviation_payoffs(mix, jacobian=True,
                                            assume_complete=True)
        assert np.allclose(dev, tdev)

        # We need to project the Jacobian onto the simplex gradient subspace
        jac -= jac.mean(-1, keepdims=True)
        tjac -= tjac.mean(-1, keepdims=True)
        assert np.allclose(jac, tjac)


@testutils.apply(repeat=20)
def test_jacobian_zeros():
    """Test that jacobian has appropriate zeros"""
    cgame = congestion.CongestionGame(3, 3, 1)
    _, jac = cgame.deviation_payoffs(cgame.random_mixtures()[0], jacobian=True)
    np.fill_diagonal(jac, 0)
    assert np.allclose(jac, 0), \
        "deviation jacobian wasn't diagonal"

    cgame = congestion.CongestionGame(5, 4, 2)
    _, jac = cgame.deviation_payoffs(cgame.random_mixtures()[0], jacobian=True)
    ns = cgame.num_strategies[0]
    opp_diag = np.arange(ns - 1, ns ** 2 - 1, ns - 1)
    assert np.allclose(jac.flat[opp_diag], 0), \
        ("jacobian with non interfering strategies didn't have appropriate "
         "zeros")


@testutils.apply([
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
], repeat=2)
def test_nash_finding(players, facilities, required):
    """Test that nash works on congestion games"""
    cgame = congestion.CongestionGame(players, facilities, required)
    eqa = nash.mixed_nash(cgame)
    assert eqa.size > 0, "didn't find any equilibria"


def test_serializer():
    """Test that serializer works"""
    cgame = congestion.CongestionGame(3, 3, 1)
    serial = cgame.gen_serializer()
    serial.to_prof_json(cgame.random_mixtures()[0])


def test_repr():
    """Test repr"""
    cgame = congestion.CongestionGame(3, 3, 1)
    assert repr(cgame) == "CongestionGame(3, 3, 1)"
