import itertools
import random

import numpy as np
import numpy.random as rand
import pytest

from gameanalysis import gamegen
from gameanalysis import matgame
from gameanalysis import rsgame
from gameanalysis import serialize
from gameanalysis import utils


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_min_max(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs)
    game = rsgame.game_copy(matg)

    assert np.allclose(matg.min_strat_payoffs(), game.min_strat_payoffs())
    assert np.allclose(matg.max_strat_payoffs(), game.max_strat_payoffs())


@pytest.mark.parametrize('strats', [
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_normalize(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs).normalize()

    assert np.allclose(matg.min_role_payoffs(), 0)
    assert np.allclose(matg.max_role_payoffs(), 1)


def test_compress_profiles():
    matg = matgame.matgame(rand.random((2, 3, 4, 3)))
    prof = [0, 1, 0, 1, 0, 1, 0, 0, 0]
    comp_prof = [1, 1, 0]
    assert np.all(comp_prof == matg.compress_profile(prof))
    assert np.all(prof == matg.uncompress_profile(comp_prof))

    prof = [0, 1, 1, 0, 0, 0, 0, 0, 1]
    comp_prof = [1, 0, 3]
    assert np.all(comp_prof == matg.compress_profile(prof))
    assert np.all(prof == matg.uncompress_profile(comp_prof))


def test_profiles_payoffs():
    matg = matgame.matgame([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    copy = rsgame.game_copy(matg)
    profs = [[1, 0, 1, 0],
             [1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 1, 0, 1]]
    pays = [[1, 0, 2, 0],
            [3, 0, 0, 4],
            [0, 5, 6, 0],
            [0, 7, 0, 8]]
    game = rsgame.game([1, 1], 2, profs, pays)
    assert copy == game


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_compress_profiles(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs)

    prof = matg.random_profiles()
    copy_prof = matg.uncompress_profile(matg.compress_profile(prof))
    assert np.all(prof == copy_prof)

    profs = matg.random_profiles(20)
    copy_profs = matg.uncompress_profile(matg.compress_profile(profs))
    assert np.all(profs == copy_profs)

    profs = matg.random_profiles(20).reshape((4, 5, -1))
    copy_profs = matg.uncompress_profile(matg.compress_profile(profs))
    assert np.all(profs == copy_profs)


def test_get_payoffs():
    payoffs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    matg = matgame.matgame(payoffs)

    expected = [1, 0, 2, 0]
    assert np.allclose(expected, matg.get_payoffs([1, 0, 1, 0]))

    expected = [3, 0, 0, 4]
    assert np.allclose(expected, matg.get_payoffs([1, 0, 0, 1]))

    expected = [0, 5, 6, 0]
    assert np.allclose(expected, matg.get_payoffs([0, 1, 1, 0]))

    expected = [0, 7, 0, 8]
    assert np.allclose(expected, matg.get_payoffs([0, 1, 0, 1]))


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_get_payoffs(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs)
    profiles = matg.random_profiles(20).reshape((4, 5, -1))
    payoffs = matg.get_payoffs(profiles)
    assert profiles.shape == payoffs.shape
    assert np.all((profiles > 0) | (payoffs == 0))


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_deviations(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs)
    game = rsgame.game_copy(matg)

    mix = matg.random_mixtures()
    matdev = matg.deviation_payoffs(mix)
    gamedev = game.deviation_payoffs(mix)
    assert np.allclose(matdev, gamedev)

    mix = matg.random_mixtures()
    matdev, matjac = matg.deviation_payoffs(mix, jacobian=True)
    gamedev, gamejac = game.deviation_payoffs(mix, jacobian=True)
    assert np.allclose(matdev, gamedev)
    assert np.allclose(matjac, gamejac)

    for mix in matg.random_mixtures(20):
        matdev = matg.deviation_payoffs(mix)
        gamedev = game.deviation_payoffs(mix)
        assert np.allclose(matdev, gamedev)


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_invariants(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs)

    assert not matg.is_empty()
    assert matg.is_complete()

    prof = matg.random_profiles()
    assert prof in matg
    for prof in matg.random_profiles(20):
        assert prof in matg


def test_is_constant_sum():
    payoffs = [[[2, -1], [0, 1]], [[5, -4], [0.5, 0.5]]]
    matg = matgame.matgame(payoffs)
    assert matg.is_constant_sum()

    payoffs = [[[2, -1], [0, 1]], [[5, -4], [0.5, 0]]]
    matg = matgame.matgame(payoffs)
    assert not matg.is_constant_sum()


def test_submatgame():
    matg = matgame.matgame(rand.random((2, 3, 4, 3)))
    mask = [True, True, True, False, True, False, False, True, True]
    smatg = matgame.matgame(matg.payoff_matrix[:, [0, 2]][:, :, 2:].copy())
    assert smatg == matg.subgame(mask)

    matg = matgame.matgame([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mask = [True, True, True, False]
    matg = matgame.matgame([[[1, 2]], [[5, 6]]])


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_matgame_hash_eq(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs)

    copy = matgame.matgame_copy(matg)
    assert hash(copy) == hash(matg)
    assert copy == matg

    game = rsgame.game_copy(matg)
    copy = matgame.matgame_copy(game)
    assert hash(copy) == hash(matg)
    assert copy == matg


def test_matgame_repr():
    matg = matgame.matgame(rand.random((2, 1)))
    assert repr(matg) == 'MatrixGame([2])'
    matg = matgame.matgame(rand.random((2, 3, 2)))
    assert repr(matg) == 'MatrixGame([2 3])'


@pytest.mark.parametrize('players,strats', [
    (1, [2, 2]),
    (2, 2),
    ([1, 2], 2),
    ([4, 3], [4, 3]),
    ([2, 3, 4], [4, 3, 2]),
])
def test_random_matgame_copy(players, strats):
    game = gamegen.role_symmetric_game(players, strats)
    matg = matgame.matgame_copy(game)
    inds = np.cumsum(game.num_role_players[:-1] * game.num_role_strats[:-1])

    mprofs = matg.random_profiles(20)
    mpays = matg.get_payoffs(mprofs)
    mpays = np.concatenate(
        [m.reshape(20, p, -1).mean(1).filled(0) for m, p
         in zip(np.split(np.ma.masked_array(mpays, mprofs == 0), inds, 1),
                game.num_role_players)], 1)

    profs = np.concatenate(
        [m.reshape(20, p, -1).sum(1) for m, p
         in zip(np.split(mprofs, inds, 1), game.num_role_players)], 1)
    pays = game.get_payoffs(profs)

    assert np.allclose(mpays, pays)


def test_serializer_copy():
    game = matgame.matgame(np.random.random((2, 3, 4, 3)))
    serial = matgame.matgameserializer_copy(gamegen.serializer(game))
    expected = ("MatGameSerializer(('r0', 'r1', 'r2'), (('s0', 's1'), "
                "('s0', 's1', 's2'), ('s0', 's1', 's2', 's3')))")
    assert repr(serial) == expected

    jgame = serial.to_json(game)
    copy = serial.from_json(jgame)
    assert game == copy
    copy, scopy = matgame.read_matgame(jgame)
    assert serial == scopy
    assert game == copy

    mask = [True, False, True, False, False, True, True, False, False]
    sserial = matgame.matgameserializer(
        ['r0', 'r1', 'r2'], [['s0'], ['s0'], ['s0', 's1']])
    assert sserial == serial.subserial(mask)


def test_serialize_copy_role_lengths():
    serial = serialize.gameserializer(
        ['a', 'b'], [['1', '2'], ['3', '4', '5']])

    mserial = matgame.matgameserializer_copy(serial)
    expected = matgame.matgameserializer(
        ['a', 'b'], [['1', '2'], ['3', '4', '5']])
    assert utils.is_sorted(mserial.role_names)
    assert mserial == expected

    mserial = matgame.matgameserializer_copy(serial, [1, 1])
    expected = matgame.matgameserializer(
        ['a', 'b'], [['1', '2'], ['3', '4', '5']])
    assert utils.is_sorted(mserial.role_names)
    assert mserial == expected

    mserial = matgame.matgameserializer_copy(serial, [2, 1])
    expected = matgame.matgameserializer(
        ['ap0', 'ap1', 'bp0'],
        [['1', '2'], ['1', '2'], ['3', '4', '5']])
    assert utils.is_sorted(mserial.role_names)
    assert mserial == expected


def test_serialize_copy_role_lengths_natural():
    serial = serialize.gameserializer(
        ['q', 'qq'], [['1', '2'], ['3', '4', '5']])
    mserial = matgame.matgameserializer_copy(serial, [2, 1])
    expected = matgame.matgameserializer(
        ['qp0', 'qp1', 'qqp0'],
        [['1', '2'], ['1', '2'], ['3', '4', '5']])
    assert utils.is_sorted(mserial.role_names)
    assert mserial == expected


def test_serialize_copy_role_lengths_unlikely():
    serial = serialize.gameserializer(
        ['a', 'aa'], [['1', '2'], ['3', '4', '5']])
    mserial = matgame.matgameserializer_copy(serial, [2, 1])
    expected = matgame.matgameserializer(
        ['0_ap0', '0_ap1', '1aap0'],
        [['1', '2'], ['1', '2'], ['3', '4', '5']])
    assert utils.is_sorted(mserial.role_names)
    assert mserial == expected


def random_names(num):
    """Produce `num` random sorted unique strings"""
    return tuple(sorted(itertools.islice(utils.iunique(
        utils.random_strings(1, 3)), num)))


@pytest.mark.parametrize('_', range(100))
def test_random_serialize_copy_role_lengths(_):
    num_roles = random.randint(2, 4)
    roles = random_names(num_roles)
    strats = tuple(random_names(random.randint(2, 4))
                   for _ in range(num_roles))
    serial = serialize.gameserializer(roles, strats)
    players = [random.randint(1, 3) for _ in range(num_roles)]
    mserial = matgame.matgameserializer_copy(serial, players)
    assert utils.is_sorted(mserial.role_names)
