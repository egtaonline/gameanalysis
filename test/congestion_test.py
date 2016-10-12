import numpy as np
import pytest

from gameanalysis import congestion
from gameanalysis import gamegen
from gameanalysis import gameio
from gameanalysis import nash
from gameanalysis import rsgame


EPS = np.finfo(float).eps


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
] * 20)
def test_deviation_payoffs(players, facilities, required):
    """Test that deviation payoff formulation is accurate"""
    cgame = gamegen.congestion_game(players, required, facilities)
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


@pytest.mark.parametrize('_', range(20))
def test_jacobian_zeros(_):
    """Test that jacobian has appropriate zeros"""
    cgame = gamegen.congestion_game(3, 1, 3)
    _, jac = cgame.deviation_payoffs(cgame.random_mixtures()[0], jacobian=True)
    np.fill_diagonal(jac, 0)
    assert np.allclose(jac, 0), \
        "deviation jacobian wasn't diagonal"

    cgame = gamegen.congestion_game(5, 2, 4)
    _, jac = cgame.deviation_payoffs(cgame.random_mixtures()[0], jacobian=True)
    ns = cgame.num_strategies[0]
    opp_diag = np.arange(ns - 1, ns ** 2 - 1, ns - 1)
    assert np.allclose(jac.flat[opp_diag], 0), \
        ("jacobian with non interfering strategies didn't have appropriate "
         "zeros")


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
] * 2)
def test_nash_finding(players, facilities, required):
    """Test that nash works on congestion games"""
    cgame = gamegen.congestion_game(players, required, facilities)
    eqa = nash.mixed_nash(cgame)
    assert eqa.size > 0, "didn't find any equilibria"


def test_serializer():
    """Test that serializer works"""
    cgame = gamegen.congestion_game(3, 1, 3)
    serial = cgame.gen_serializer()
    serial.to_prof_json(cgame.random_mixtures()[0])


def test_repr():
    """Test repr"""
    cgame = gamegen.congestion_game(3, 1, 3)
    assert repr(cgame) == "CongestionGame(3, 1, 3)"


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
] * 20)
def test_manual_copy(players, facilities, required):
    """Test that manually copying a congestion game works"""
    cgame1 = gamegen.congestion_game(players, required, facilities)
    cgame2 = congestion.CongestionGame(players, required,
                                       cgame1.facility_coefs)
    mixes = cgame1.random_mixtures(20)
    for mix in mixes:
        dev1 = cgame1.deviation_payoffs(mix)
        dev2 = cgame2.deviation_payoffs(mix)
        assert np.allclose(dev1, dev2)


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
] * 20)
def test_json_copy(players, facilities, required):
    """Test that manually copying a congestion game works"""
    cgame1 = gamegen.congestion_game(players, required, facilities)
    cgame2, _ = congestion.read_congestion_game(cgame1.to_json())
    mixes = cgame1.random_mixtures(20)
    for mix in mixes:
        dev1 = cgame1.deviation_payoffs(mix)
        dev2 = cgame2.deviation_payoffs(mix)
        assert np.allclose(dev1, dev2)


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
] * 20)
def test_serial_json_copy(players, facilities, required):
    """Test that manually copying a congestion game works"""
    cgame1 = gamegen.congestion_game(players, required, facilities)
    serial = cgame1.gen_serializer()
    cgame2, _ = congestion.read_congestion_game(cgame1.to_json(serial))
    mixes = cgame1.random_mixtures(20)
    for mix in mixes:
        dev1 = cgame1.deviation_payoffs(mix)
        dev2 = cgame2.deviation_payoffs(mix)
        assert np.allclose(dev1, dev2)


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
] * 20)
def test_serial_to_base_game(players, facilities, required):
    """Test that reading a congestion game as a base game works"""
    cgame = gamegen.congestion_game(players, required, facilities)
    serial1 = cgame.gen_serializer()
    base1 = rsgame.BaseGame(cgame)
    base2, serial2 = gameio.read_base_game(cgame.to_json(serial1))
    assert base1 == base2
    assert serial1 == serial2


def test_to_str():
    cgame = gamegen.congestion_game(3, 2, 3)
    serial = cgame.gen_serializer()
    expected = ('CongestionGame:\n    Players: 3\n    Required Facilities: 2\n'
                '    Facilities: 0, 1, 2\n')
    assert cgame.to_str() == expected
    assert cgame.to_str(serial) == expected


def test_warning_on_bad_serial():
    cgame = gamegen.congestion_game(3, 1, 2)
    serial = gameio.GameSerializer(['all'], [['unserscore_fac', '0', '1']])
    with pytest.warns(UserWarning):
        cgame.to_json(serial)
    with pytest.warns(UserWarning):
        cgame.to_str(serial)


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (5, 6, 4),
] * 20)
def test_payoff_bounds(players, facilities, required):
    cgame = gamegen.congestion_game(3, 1, 2)
    minp = cgame.min_payoffs()[0] - EPS
    maxp = cgame.max_payoffs()[0] + EPS
    profiles = cgame.all_profiles()
    payoffs = np.sum(cgame.get_payoffs(profiles) * profiles, 1)
    assert np.all(minp <= payoffs)
    assert np.all(maxp >= payoffs)
