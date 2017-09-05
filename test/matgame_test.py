import numpy as np
import numpy.random as rand
import pytest

from gameanalysis import matgame
from gameanalysis import rsgame


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

    profs = matg.random_profiles(20).T
    copy_profs = matg.uncompress_profile(matg.compress_profile(profs, 0), 0)
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


def test_samplematgame_payoffs():
    matrix = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
              [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
    matg = matgame.samplematgame(matrix)
    copy = rsgame.samplegame_copy(matg)
    profiles = [[1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1]]
    spayoffs = [[[[1, 2], [0] * 2, [3, 4], [0] * 2],
                 [[5, 6], [0] * 2, [0] * 2, [7, 8]],
                 [[0] * 2, [9, 10], [11, 12], [0] * 2],
                 [[0] * 2, [13, 14], [0] * 2, [15, 16]]]]
    game = rsgame.samplegame([1, 1], 2, profiles, spayoffs)
    assert copy == game


def test_get_sample_payoffs():
    payoffs = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
               [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
    smatg = matgame.samplematgame(payoffs)

    expected = [[1, 0, 3, 0],
                [2, 0, 4, 0]]
    assert np.allclose(expected, smatg.get_sample_payoffs([1, 0, 1, 0]))

    expected = [[5, 0, 0, 7],
                [6, 0, 0, 8]]
    assert np.allclose(expected, smatg.get_sample_payoffs([1, 0, 0, 1]))

    expected = [[0, 9, 11, 0],
                [0, 10, 12, 0]]
    assert np.allclose(expected, smatg.get_sample_payoffs([0, 1, 1, 0]))

    expected = [[0, 13, 0, 15],
                [0, 14, 0, 16]]
    assert np.allclose(expected, smatg.get_sample_payoffs([0, 1, 0, 1]))


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
def test_random_singleton_resample(strats):
    payoffs = rand.random(tuple(strats) + (len(strats),))
    matg = matgame.matgame(payoffs)
    smatg = matgame.samplematgame_copy(matg)

    for _ in range(5):
        assert matg == smatg.resample()


@pytest.mark.parametrize('strats', [
    [1],
    [3],
    [2, 3],
    [1, 2, 3],
    [2, 3, 1],
])
@pytest.mark.parametrize('samples', [1, 3])
def test_random_samplematgame_hash_eq(strats, samples):
    payoffs = rand.random(tuple(strats) + (len(strats), samples))
    smatg = matgame.samplematgame(payoffs)

    copy = matgame.samplematgame_copy(smatg)
    assert hash(smatg) == hash(copy)
    assert smatg == copy

    sgame = rsgame.samplegame_copy(smatg)
    copy = matgame.samplematgame_copy(sgame)
    assert hash(smatg) == hash(copy)
    assert smatg == copy

    perm = rand.permutation(samples)
    copy = matgame.samplematgame(smatg.spayoff_matrix[..., perm])
    assert hash(smatg) == hash(copy)
    assert smatg == copy


def test_from_samplegame_truncate():
    base = rsgame.basegame(1, [1, 2])
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5], [2], [0]],
        ],
        [
            [[5, 6], [0, 0], [2, 3]],
        ],
    ]
    game = rsgame.samplegame_copy(base, profiles, payoffs)
    smatg = matgame.samplematgame_copy(game)
    assert np.all(smatg.num_samples == [1])


def test_samplematgame_repr():
    smatg = matgame.samplematgame(rand.random((2, 1, 1)))
    assert repr(smatg) == 'SampleMatrixGame([2], 1)'
    smatg = matgame.samplematgame(rand.random((2, 3, 2, 4)))
    assert repr(smatg) == 'SampleMatrixGame([2 3], 4)'
