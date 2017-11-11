import collections

import numpy as np
import numpy.random as rand
import pytest

from gameanalysis import paygame
from gameanalysis import rsgame
from test import testutils


TINY = np.finfo(float).tiny
EPS = 5 * np.finfo(float).eps


def random_game(role_players, role_strats, *, prob=0.5):
    base = rsgame.emptygame(role_players, role_strats)
    profs = base.all_profiles()
    rand.shuffle(profs)
    num_profs = rand.binomial(base.num_all_profiles, prob)
    profs = profs[:num_profs].copy()
    pays = rand.random(profs.shape)
    pays *= profs > 0
    return paygame.game_replace(base, profs, pays)


def random_samplegame(role_players, role_strats, *, prob=0.5):
    base = rsgame.emptygame(role_players, role_strats)
    profs = base.all_profiles()
    rand.shuffle(profs)
    spays = []
    start = 0
    for n, c in collections.Counter(
            rand.geometric(prob, base.num_all_profiles) - 1).items():
        if n == 0:
            profs = np.delete(profs, slice(start, start + c), 0)
        else:
            mask = (0 < profs[start:start + c, None])
            start += c
            spays.append(rand.random((c, n, base.num_strats)) * mask)
    return paygame.samplegame_replace(base, profs, spays)


# ----
# Game
# ----


def test_game_properties():
    game = paygame.game(1, 1, np.empty((0, 1), int), np.empty((0, 1)))
    assert game.profiles().shape == (0, 1)
    assert game.payoffs().shape == (0, 1)
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    game = paygame.game(3, 2, [[3, 0]], [[0, 0]])
    assert np.all(game.profiles() == [[3, 0]])
    assert np.all(game.payoffs() == [[0, 0]])
    assert game.num_profiles == 1
    assert game.num_complete_profiles == 1

    profs = [[1, 0, 3, 0],
             [1, 0, 2, 1]]
    pays = [[0, 0, 0, 0],
            [np.nan, 0, 0, 0]]
    game = paygame.game([1, 3], 2, profs, pays)
    assert game.profiles().shape == (2, 4)
    assert game.payoffs().shape == (2, 4)
    assert game.num_profiles == 2
    assert game.num_complete_profiles == 1

    game = rsgame.emptygame(1, [3, 1])
    assert game.profiles().shape == (0, 4)
    assert game.payoffs().shape == (0, 4)
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    game = rsgame.emptygame([3, 2, 1], 3)
    assert game.profiles().shape == (0, 9)
    assert game.payoffs().shape == (0, 9)
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    game = rsgame.emptygame([3, 2, 1], [1, 2, 3])
    assert game.profiles().shape == (0, 6)
    assert game.payoffs().shape == (0, 6)
    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    with pytest.raises(AssertionError):
        paygame.game(1, 1, [[1]], [])
    with pytest.raises(AssertionError):
        paygame.game(1, 1, [[2]], [[0]])
    with pytest.raises(AssertionError):
        paygame.game(1, 2, [[1]], [[0]])
    with pytest.raises(AssertionError):
        paygame.game(1, 2, [[2, -1]], [[0, 0]])
    with pytest.raises(AssertionError):
        paygame.game(1, 2, [[1, 0]], [[0, 1]])
    with pytest.raises(AssertionError):
        paygame.game(1, 2, [[1, 0], [1, 0]], [[0, 0], [0, 0]])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_emptygame_const_properties(role_players, role_strats):
    game = paygame.game_copy(rsgame.emptygame(role_players, role_strats))

    assert game.num_profiles == 0
    assert game.num_complete_profiles == 0

    assert np.isnan(game.min_strat_payoffs()).all()
    assert game.min_strat_payoffs().shape == (game.num_strats,)
    assert np.isnan(game.max_strat_payoffs()).all()
    assert game.max_strat_payoffs().shape == (game.num_strats,)
    assert np.isnan(game.min_role_payoffs()).all()
    assert game.min_role_payoffs().shape == (game.num_roles,)
    assert np.isnan(game.max_role_payoffs()).all()
    assert game.max_role_payoffs().shape == (game.num_roles,)

    prof = game.random_profiles()
    pays = game.get_payoffs(prof)
    assert np.isnan(pays[0 < prof]).all()
    assert np.all(pays[0 == prof] == 0)
    assert pays.shape == (game.num_strats,)

    mix = game.random_mixtures()
    dev_pays = game.deviation_payoffs(mix)
    assert np.isnan(dev_pays).all()
    assert dev_pays.shape == (game.num_strats,)

    exp_pays = game.expected_payoffs(mix)
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)

    exp_pays = game.expected_payoffs(mix, deviations=dev_pays)
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)

    dev_pays, dev_jac = game.deviation_payoffs(mix, jacobian=True)
    assert np.isnan(dev_pays).all()
    assert dev_pays.shape == (game.num_strats,)
    assert np.isnan(dev_jac).all()
    assert dev_jac.shape == (game.num_strats, game.num_strats)

    exp_pays, exp_jac = game.expected_payoffs(mix, jacobian=True)
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)
    assert np.isnan(exp_jac).all()
    assert exp_jac.shape == (game.num_roles, game.num_strats)

    exp_pays, exp_jac = game.expected_payoffs(
        mix, jacobian=True, deviations=(dev_pays, dev_jac))
    assert np.isnan(exp_pays).all()
    assert exp_pays.shape == (game.num_roles,)
    assert np.isnan(exp_jac).all()
    assert exp_jac.shape == (game.num_roles, game.num_strats)

    br = game.best_response(mix)
    assert np.isnan(br).all()
    assert br.shape == (game.num_strats,)

    assert game.profiles().size == 0
    assert game.payoffs().size == 0

    assert game.is_empty()
    assert not game.is_complete()
    assert game.is_constant_sum()

    assert game.normalize() == game
    assert game.random_profiles() not in game


def test_game_verifications():
    game = rsgame.emptygame(2, 2)

    profiles = [[3, -1]]
    payoffs = [[4, 5]]
    with pytest.raises(AssertionError):
        paygame.game_replace(game, profiles, payoffs)

    profiles = [[3, 0]]
    with pytest.raises(AssertionError):
        paygame.game_replace(game, profiles, payoffs)

    profiles = [[2, 0]]
    with pytest.raises(AssertionError):
        paygame.game_replace(game, profiles, payoffs)

    profiles = [[1, 1]]
    payoffs = [[np.nan, np.nan]]
    with pytest.raises(AssertionError):
        paygame.game_replace(game, profiles, payoffs)

    profiles = [[2, 0]]
    payoffs = [[np.nan, 0]]
    with pytest.raises(AssertionError):
        paygame.game_replace(game, profiles, payoffs)


def test_dev_reps_on_large_games():
    profiles = [[1000, 0], [500, 500]]
    game = paygame.game(1000, 2, profiles, np.zeros_like(profiles))
    expected = [[0, -np.inf], [688.77411439] * 2]
    assert np.allclose(expected, game._dev_reps)

    profiles = [[12] + [0] * 11, [1] * 12]
    game = paygame.game(12, 12, profiles, np.zeros_like(profiles))
    expected = [[0] + [-np.inf] * 11, [17.50230785] * 12]
    assert np.allclose(expected, game._dev_reps)

    profiles = [[5] + [0] * 39, ([1] + [0] * 7) * 5]
    game = paygame.game(5, 40, profiles, np.zeros_like(profiles))
    expected = [[0] + [-np.inf] * 39, ([3.17805383] + [-np.inf] * 7) * 5]
    assert np.allclose(expected, game._dev_reps)

    profiles = [([2] + [0] * 39) * 2,
                [2] + [0] * 39 + ([1] + [0] * 19) * 2,
                ([1] + [0] * 19) * 4]
    game = paygame.game([2, 2], 40, profiles, np.zeros_like(profiles))
    expected = [([0] + [-np.inf] * 39) * 2,
                [0.69314718] + [-np.inf] * 39 + ([0] + [-np.inf] * 19) * 2,
                ([0.69314718] + [-np.inf] * 19) * 4]
    assert np.allclose(expected, game._dev_reps)


def test_min_max_payoffs():
    game = rsgame.emptygame([2, 2], 2)
    mins = game.min_strat_payoffs()
    assert np.allclose([np.nan] * 4, mins, equal_nan=True)
    mins = game.min_role_payoffs()
    assert np.allclose([np.nan] * 2, mins, equal_nan=True)
    maxs = game.max_strat_payoffs()
    assert np.allclose([np.nan] * 4, maxs, equal_nan=True)
    maxs = game.max_role_payoffs()
    assert np.allclose([np.nan] * 2, maxs, equal_nan=True)

    profs = [[1, 1, 1, 1, 2, 0, 2, 0],
             [2, 0, 2, 0, 2, 0, 2, 0]]
    pays = [[np.nan, 1, 2, np.nan, 3, 0, np.nan, 0],
            [4, 0, 5, 0, 6, 0, np.nan, 0]]
    game = paygame.game([2] * 4, 2, profs, pays)
    mins = game.min_strat_payoffs()
    assert np.allclose([4, 1, 2, np.nan, 3, np.nan, np.nan, np.nan], mins,
                       equal_nan=True)
    mins = game.min_role_payoffs()
    assert np.allclose([1, 2, 3, np.nan], mins, equal_nan=True)
    maxs = game.max_strat_payoffs()
    assert np.allclose([4, 1, 5, np.nan, 6, np.nan, np.nan, np.nan], maxs,
                       equal_nan=True)
    maxs = game.max_role_payoffs()
    assert np.allclose([4, 5, 6, np.nan], maxs, equal_nan=True)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_min_max_payoffs(role_players, role_strats):
    game = random_game(role_players, role_strats, prob=1)
    assert ((game.payoffs() >= game.min_strat_payoffs()) |
            (game.profiles() == 0)).all()
    assert ((game.payoffs() >= game.min_role_payoffs().repeat(
        game.num_role_strats)) | (game.profiles() == 0)).all()
    assert (game.payoffs() <= game.max_strat_payoffs()).all()
    assert (game.payoffs() <= game.max_role_payoffs().repeat(
        game.num_role_strats)).all()


def test_best_response_pure():
    profiles = [[1, 0, 2, 0],
                [1, 0, 1, 1],
                [1, 0, 0, 2],
                [0, 1, 2, 0],
                [0, 1, 1, 1],
                [0, 1, 0, 2]]
    payoffs = [[1, 0, 2, 0],
               [3, 0, 4, 5],
               [6, 0, 0, 7],
               [0, 8, 9, 0],
               [0, 10, 11, 12],
               [0, 13, 0, 14]]
    game = paygame.game([1, 2], 2, profiles, payoffs)

    br = game.best_response([1, 0, 1, 0])
    assert np.allclose(br, [0, 1, 0, 1])
    br = game.best_response([0, 1, 0, 1])
    assert np.allclose(br, [0, 1, 0, 1])


def test_best_response_mixed():
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[0, 0],
               [0.4, 0.6],
               [0, 0]]
    game = paygame.game(2, 2, profiles, payoffs)

    br = game.best_response([1, 0])
    assert np.allclose(br, [0, 1])
    br = game.best_response([0, 1])
    assert np.allclose(br, [1, 0])
    br = game.best_response([0.4, 0.6])
    assert np.allclose(br, [0.5, 0.5])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_best_response(role_players, role_strats):
    game = random_game(role_players, role_strats, prob=1)

    for mix in game.random_mixtures(20):
        br = game.best_response(mix)
        supp = br > 0
        sub_starts = np.insert(
            np.add.reduceat(supp, game.role_starts)[:-1].cumsum(), 0, 0)
        devs = game.deviation_payoffs(mix)
        avg = np.add.reduceat(br * devs, game.role_starts)
        mx = np.maximum.reduceat(devs[supp], sub_starts)
        assert np.allclose(avg, mx)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_game_normalize(role_players, role_strats):
    game = random_game(role_players, role_strats).normalize()
    mins = game.min_role_payoffs()
    assert np.all(np.isclose(mins, 0) | np.isnan(mins))
    maxs = game.max_role_payoffs()
    assert np.all(np.isclose(maxs, 1) | np.isclose(maxs, 0) | np.isnan(maxs))


def test_get_payoffs():
    profs = [[2, 0, 0, 2, 1],
             [1, 1, 0, 0, 3]]
    pays = [[1, 0, 0, 2, 3],
            [4, 5, 0, 0, np.nan]]
    game = paygame.game([2, 3], [3, 2], profs, pays)

    pay = game.get_payoffs([2, 0, 0, 2, 1])
    assert np.allclose([1, 0, 0, 2, 3], pay)
    pay = game.get_payoffs([1, 1, 0, 0, 3])
    assert np.allclose([4, 5, 0, 0, np.nan], pay, equal_nan=True)
    pay = game.get_payoffs([2, 0, 0, 3, 0])
    assert np.allclose([np.nan, 0, 0, np.nan, 0], pay, equal_nan=True)

    with pytest.raises(AssertionError):
        game.get_payoffs([1, 0, 0, 2, 1])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_get_payoffs(role_players, role_strats):
    game = random_game(role_players, role_strats)
    for prof, pay in zip(game.profiles(), game.payoffs()):
        assert np.allclose(pay, game.get_payoffs(prof))


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_empty_get_payoffs(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)

    for prof in game.all_profiles():
        supp = prof > 0
        pay = game.get_payoffs(prof)
        assert np.isnan(pay[supp]).all()
        assert np.all(pay[~supp] == 0)


def test_deviation_mixture_support():
    base = rsgame.emptygame([2, 2], 3)
    profiles1 = [
        [2, 0, 0, 2, 0, 0],
        [1, 1, 0, 2, 0, 0],
        [0, 2, 0, 2, 0, 0],
    ]
    payoffs1 = [
        [1, 0, 0, 2, 0, 0],
        [3, 4, 0, 5, 0, 0],
        [0, 6, 0, 7, 0, 0],
    ]
    profiles2 = [
        [2, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 2, 0, 1, 1, 0],
    ]
    payoffs2 = [
        [8, 0, 0, 9, 10, 0],
        [11, 12, 0, 13, 14, 0],
        [0, 15, 0, 16, 17, 0],
    ]
    game1 = paygame.game_replace(base, profiles1, payoffs1)
    game2 = paygame.game_replace(base, profiles2, payoffs2)
    game3 = paygame.game_replace(
        base, profiles1 + profiles2, payoffs1 + payoffs2)
    mix1 = [0.5, 0.5, 0, 0.3, 0.7, 0]
    mix2 = [0.5, 0.5, 0, 1, 0, 0]

    dev_payoffs = game1.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game1.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs, [2, 5, np.nan, 4.75, np.nan, np.nan],
                       equal_nan=True)
    dev_payoffs = game2.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game2.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs,
                       [np.nan, np.nan, np.nan, np.nan, 13.75, np.nan],
                       equal_nan=True)
    dev_payoffs = game3.deviation_payoffs(mix1)
    assert np.isnan(dev_payoffs).all()
    dev_payoffs = game3.deviation_payoffs(mix2)
    assert np.allclose(dev_payoffs, [2, 5, np.nan, 4.75, 13.75, np.nan],
                       equal_nan=True)


# Test sample game with different number of samples
def test_different_samples():
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5, 2, 0]],
        ],
        [
            [[5, 0, 2], [6, 0, 3]],
        ],
    ]
    game = paygame.samplegame(1, [1, 2], profiles, payoffs)

    assert np.all([1, 2] == game.num_samples), \
        "didn't get both sample sizes"
    assert repr(game) is not None


def test_deviation_payoffs_jacobian():
    game = rsgame.emptygame(2, 3)
    eqm = np.ones(3) / 3
    dp, dpj = game.deviation_payoffs(eqm, jacobian=True)
    assert np.isnan(dp).all()
    assert np.isnan(dpj).all()

    profiles = [[2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2]]
    payoffs = [[0, 0, 0],
               [-1, 1, 0],
               [1, 0, -1],
               [0, 0, 0],
               [0, -1, 1],
               [0, 0, 0]]
    game = paygame.game(2, 3, profiles, payoffs)
    eqm = np.ones(3) / 3
    dp, dpj = game.deviation_payoffs(eqm, jacobian=True)
    assert np.allclose(dp, 0)
    expected_jac = np.array([[0, -1, 1],
                             [1, 0, -1],
                             [-1, 1, 0]])
    assert np.allclose(dpj, expected_jac)


def test_flat_profile_payoffs():
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5, 2, 0]],
        ],
        [
            [[5, 0, 2], [6, 0, 3]],
        ],
    ]
    game = paygame.samplegame(1, [1, 2], profiles, payoffs)

    expected_profs = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
    ])
    expected_pays = np.array([
        [5, 2, 0],
        [5, 0, 2],
        [6, 0, 3],
    ], float)

    flat_profs = game.flat_profiles()
    assert np.all(flat_profs[np.lexsort(flat_profs.T)] ==
                  expected_profs[np.lexsort(expected_profs.T)])
    flat_pays = game.flat_payoffs()
    assert np.allclose(flat_pays[np.lexsort(flat_pays.T)],
                       expected_pays[np.lexsort(expected_pays.T)])


def test_nan_mask_for_dev_payoffs():
    profiles = [[3, 0, 0, 0],
                [2, 1, 0, 0],
                [2, 0, 1, 0]]
    payoffs = [[1, 0, 0, 0],
               [np.nan, 2, 0, 0],
               [5, 0, np.nan, 0]]
    game = paygame.game([3], [4], profiles, payoffs)
    devs = game.deviation_payoffs([1, 0, 0, 0])
    assert np.allclose(devs, [1, 2, np.nan, np.nan], equal_nan=True)


def test_nan_payoffs_for_dev_payoffs():
    profiles = [[3, 0, 3, 0],
                [2, 1, 3, 0],
                [3, 0, 2, 1]]
    payoffs = [[1, 0, 2, 0],
               [np.nan, 3, np.nan, 0],
               [np.nan, 0, np.nan, 4]]
    game = paygame.game([3, 3], [2, 2], profiles, payoffs)
    devs = game.deviation_payoffs([1, 0, 1, 0])
    assert np.allclose(devs, [1, 3, 2, 4])


@pytest.mark.parametrize('p', [2, 5, 10, 100])
def test_deviation_nans(p):
    profiles = [[p,     0, 0, 0, 1],
                [p - 1, 1, 0, 0, 1],
                [p - 1, 0, 1, 0, 1],
                [p - 1, 0, 0, 1, 1]]
    payoffs = [[1,      0, 0, 0, 2],
               [np.nan, 3, 0, 0, np.nan],
               [np.nan, 0, 4, 0, np.nan],
               [np.nan, 0, 0, 5, np.nan]]
    game = paygame.game([p, 1], [4, 1], profiles, payoffs)
    mix = np.array([1, 0, 0, 0, 1])
    pays = game.deviation_payoffs(mix)
    assert not np.isnan(pays).any()


@pytest.mark.parametrize('p', [2, 5, 10, 100])
@pytest.mark.parametrize('q', [2, 5, 10, 100])
def test_deviation_nans_2(p, q):
    profiles = [[p,     0, 0, 0, q,     0],
                [p - 1, 1, 0, 0, q,     0],
                [p - 1, 0, 1, 0, q,     0],
                [p - 1, 0, 0, 1, q,     0],
                [p,     0, 0, 0, q - 1, 1]]
    payoffs = [[1,      0, 0, 0, 2,      0],
               [np.nan, 3, 0, 0, np.nan, 0],
               [np.nan, 0, 4, 0, np.nan, 0],
               [np.nan, 0, 0, 5, np.nan, 0],
               [6,      0, 0, 0, np.nan, 7]]
    game = paygame.game([p, q], [4, 2], profiles, payoffs)
    mix = np.array([1, 0, 0, 0, 1, 0])
    pays = game.deviation_payoffs(mix)
    assert not np.isnan(pays).any()


def test_expected_payoffs():
    game = rsgame.emptygame(2, [2, 2])
    pays = game.expected_payoffs([0.2, 0.8, 0.4, 0.6])
    assert np.allclose([np.nan, np.nan], pays, equal_nan=True)

    profs = [[2, 0],
             [1, 1],
             [0, 2]]
    pays = [[1, 0],
            [2, 3],
            [0, 4]]
    game = paygame.game(2, 2, profs, pays)
    pays = game.expected_payoffs([0.2, 0.8])
    assert np.allclose(3.4, pays)

    profs = [[2, 0, 2, 0],
             [2, 0, 1, 1],
             [2, 0, 0, 2],
             [1, 1, 2, 0],
             [1, 1, 1, 1],
             [1, 1, 0, 2],
             [0, 2, 2, 0],
             [0, 2, 1, 1],
             [0, 2, 0, 2]]
    pays = [[1, 0, 2, 0],
            [3, 0, 4, 5],
            [6, 0, 0, 7],
            [8, 9, 10, 0],
            [11, 12, 13, 14],
            [15, 16, 0, 17],
            [0, 18, 19, 0],
            [0, 20, 21, 22],
            [0, 23, 0, 24]]
    game = paygame.game([2, 2], [2, 2], profs, pays)
    pays = game.expected_payoffs([0.2, 0.8, 0.4, 0.6])
    assert np.allclose([17.424, 18.824], pays)

    profs = [[2, 0, 2, 0],
             [2, 0, 1, 1],
             [2, 0, 0, 2]]
    pays = [[1, 0, 2, 0],
            [3, 0, 4, 5],
            [6, 0, 0, 7]]
    game = paygame.game([2, 2], [2, 2], profs, pays)
    pays = game.expected_payoffs([0.2, 0.8, 0.4, 0.6])
    assert np.allclose([np.nan, np.nan], pays, equal_nan=True)
    pays = game.expected_payoffs([1, 0, 0.4, 0.6])
    assert np.allclose([3.76, 5], pays)

    profs = [[2, 0, 2, 0],
             [2, 0, 1, 1],
             [2, 0, 0, 2]]
    pays = [[1, 0, 2, 0],
            [3, 0, 4, 5],
            [np.nan, 0, 0, 7]]
    game = paygame.game([2, 2], [2, 2], profs, pays)
    pays = game.expected_payoffs([1, 0, 0.4, 0.6])
    assert np.allclose([np.nan, 5], pays, equal_nan=True)


def test_expected_payoffs_jac():
    profiles = [[2, 0],
                [1, 1],
                [0, 2]]
    payoffs = [[1, 0],
               [3, 3],
               [0, 1]]
    game = paygame.game(2, 2, profiles, payoffs)
    ep, ep_jac = game.expected_payoffs([.5, .5], jacobian=True)
    ep_jac -= ep_jac.sum() / 2  # project on simplex
    assert np.allclose(ep, 2)
    assert np.allclose(ep_jac, 0), \
        "maximum surplus should have 0 jacobian"

    dev_data = game.deviation_payoffs([0.5, 0.5], jacobian=True)
    ep, ep_jac = game.expected_payoffs([.5, .5], jacobian=True,
                                       deviations=dev_data)
    ep_jac -= ep_jac.sum() / 2  # project on simplex
    assert np.allclose(ep, 2)
    assert np.allclose(ep_jac, 0), \
        "maximum surplus should have 0 jacobian"


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_is_empty(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    assert game.is_empty()

    game = paygame.game_replace(game, np.empty((0, game.num_strats), int),
                                np.empty((0, game.num_strats)))
    assert game.is_empty()

    game = paygame.game_replace(game, game.random_profiles()[None],
                                np.zeros((1, game.num_strats)))
    assert not game.is_empty()


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_is_complete(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    assert not game.is_complete()

    game = paygame.game_replace(
        game, game.all_profiles(),
        np.zeros((game.num_all_profiles, game.num_strats)))
    assert game.is_complete()

    game = paygame.game_replace(game, game.profiles()[1:], game.payoffs()[1:])
    assert not game.is_complete()


def test_is_constant_sum():
    game = rsgame.emptygame(2, 3)
    assert game.is_constant_sum()

    profiles = [
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
    ]
    payoffs = [
        [2, 0, -2, 0],
        [3, 0, 0, -3],
        [0, 5, -5, 0],
        [0, 1, 0, -1],
    ]
    game = paygame.game(1, [2, 2], profiles, payoffs)
    assert game.is_constant_sum()

    payoffs = game.payoffs().copy()
    payoffs[game.profiles() > 0] += 1
    game = paygame.game_replace(game, game.profiles(), payoffs)
    assert game.is_constant_sum()

    profiles = [
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
    ]
    payoffs = [
        [1, 0, 2, 0],
        [3, 0, 0, 4],
        [0, 5, 6, 0],
        [0, 7, 0, 8],
    ]
    game = paygame.game_replace(game, profiles, payoffs)
    assert not game.is_constant_sum()


def test_game_subgame():
    profiles = [
        [2, 0, 2, 0],
        [1, 1, 2, 0],
        [0, 2, 2, 0],
    ]
    payoffs = [
        [1, 0, 2, 0],
        [3, 4, 5, 0],
        [0, 6, 7, 0],
    ]
    game = paygame.game(2, [2, 2], profiles, payoffs)
    mask = [True, False, True, False]
    sprofiles = [[2, 2]]
    spayoffs = [[1, 2]]
    sgame = paygame.game_names(
        ['r0', 'r1'], 2, [['s0'], ['s2']], sprofiles, spayoffs)
    assert sgame == game.subgame(mask)

    mask = [True, True, False, True]
    sgame = paygame.game_copy(rsgame.emptygame_names(
        ['r0', 'r1'], 2, [['s0', 's1'], ['s3']]))
    assert sgame == game.subgame(mask)

    profiles = [
        [2, 0, 1, 1],
        [1, 1, 1, 1],
        [0, 2, 1, 1],
    ]
    payoffs = [
        [8, 0, 9, 10],
        [11, 12, 13, 14],
        [0, 15, 16, 17],
    ]
    game = paygame.game_replace(game, profiles, payoffs)
    mask = [True, False, True, True]
    sprofiles = [[2, 1, 1]]
    spayoffs = [[8, 9, 10]]
    sgame = paygame.game_names(
        ('r0', 'r1'), 2, (('s0',), ('s2', 's3')), sprofiles, spayoffs)
    assert sgame == game.subgame(mask)

    mask = [True, True, False, True]
    sgame = paygame.game_copy(rsgame.emptygame_names(
        ['r0', 'r1'], 2, [['s0', 's1'], ['s3']]))
    assert sgame == game.subgame(mask)


def test_contains():
    profs = [[2, 0, 0, 2, 1],
             [1, 1, 0, 0, 3]]
    pays = [[1, 0, 0, 2, 3],
            [4, 5, 0, 0, np.nan]]
    game = paygame.game([2, 3], [3, 2], profs, pays)
    assert [2, 0, 0, 2, 1] in game
    assert [1, 1, 0, 0, 3] not in game
    assert [1, 1, 0, 2, 1] not in game


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_contains(role_players, role_strats):
    game = rsgame.emptygame(role_players, role_strats)
    for prof in game.all_profiles():
        assert prof not in game

    game = paygame.game_replace(
        game, game.all_profiles(),
        np.zeros((game.num_all_profiles, game.num_strats)))
    for prof in game.all_profiles():
        assert prof in game


def test_to_from_prof_json():
    game = paygame.game_copy(rsgame.emptygame([11, 3], [2, 1]))
    prof = [6, 5, 3]
    json_prof = {'r0': {'s1': 5, 's0': 6}, 'r1': {'s2': 3}}
    assert game.to_prof_json(prof) == json_prof
    new_prof = game.from_prof_json(json_prof)
    assert np.all(new_prof == prof)
    assert new_prof.dtype == int
    new_prof = np.empty_like(new_prof)
    game.from_prof_json(json_prof, dest=new_prof)
    assert np.all(new_prof == prof)

    player_prof = {'players': [
        {'role': 'r0', 'strategy': 's1', 'payoff': 0},
        {'role': 'r0', 'strategy': 's1', 'payoff': 0},
        {'role': 'r0', 'strategy': 's1', 'payoff': 0},
        {'role': 'r0', 'strategy': 's1', 'payoff': 0},
        {'role': 'r0', 'strategy': 's1', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r1', 'strategy': 's2', 'payoff': 0},
        {'role': 'r1', 'strategy': 's2', 'payoff': 0},
        {'role': 'r1', 'strategy': 's2', 'payoff': 0},
    ]}
    new_prof = game.from_prof_json(player_prof)
    assert np.all(new_prof == prof)
    assert new_prof.dtype == int


def test_to_from_payoff_json():
    game = paygame.game_copy(rsgame.emptygame([11, 3], [2, 1]))
    pay = [1.0, 2.0, 3.0]
    json_pay = {'r0': {'s1': 2.0, 's0': 1.0}, 'r1': {'s2': 3.0}}
    assert game.to_payoff_json(pay) == json_pay
    new_pay = game.from_payoff_json(json_pay)
    assert np.allclose(new_pay, pay)
    assert new_pay.dtype == float
    new_pay = np.empty_like(new_pay)
    game.from_payoff_json(json_pay, dest=new_pay)
    assert np.allclose(new_pay, pay)

    player_pay = {'players': [
        {'role': 'r0', 'strategy': 's1', 'payoff': 4},
        {'role': 'r0', 'strategy': 's1', 'payoff': 2},
        {'role': 'r0', 'strategy': 's1', 'payoff': 0},
        {'role': 'r0', 'strategy': 's1', 'payoff': 4},
        {'role': 'r0', 'strategy': 's1', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 2},
        {'role': 'r0', 'strategy': 's0', 'payoff': 2},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r0', 'strategy': 's0', 'payoff': 2},
        {'role': 'r0', 'strategy': 's0', 'payoff': 0},
        {'role': 'r1', 'strategy': 's2', 'payoff': 0},
        {'role': 'r1', 'strategy': 's2', 'payoff': 6},
        {'role': 'r1', 'strategy': 's2', 'payoff': 3},
    ]}
    new_pay = game.from_payoff_json(player_pay)
    assert np.allclose(new_pay, pay)
    assert new_pay.dtype == float


def test_load_empty_observations():
    game = paygame.game_copy(rsgame.emptygame(1, [2, 1]))
    profile = {
        'symmetry_groups': [
            {'strategy': 's0', 'id': 0, 'role': 'r0', 'count': 1},
            {'strategy': 's2', 'id': 1, 'role': 'r1', 'count': 1}],
        'observations': []}
    payoff = game.from_payoff_json(profile)
    assert np.allclose(payoff, [np.nan, 0, np.nan], equal_nan=True)

    profile = {'r0': {'s0': []},
               'r1': {'s2': []}}
    payoff = game.from_payoff_json(profile)
    assert np.allclose(payoff, [np.nan, 0, np.nan], equal_nan=True)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_game_json_identity(role_players, role_strats):
    game = random_game(role_players, role_strats)
    copy = paygame.game_json(game.to_json())
    assert game == copy


def test_game_hash_eq():
    a = rsgame.emptygame(4, 5)
    b = rsgame.emptygame([4], [5])
    assert a == b and hash(a) == hash(b)

    a = paygame.game(4, 2, [[3, 1], [2, 2]], [[1, 2], [3, 4]])
    b = paygame.game([4], [2], [[2, 2], [3, 1]], [[3, 4], [1, 2]])
    assert a == b and hash(a) == hash(b)


def test_game_repr():
    game = paygame.game_copy(rsgame.emptygame(3, 4))
    expected = 'Game([3], [4], 0 / 20)'
    assert repr(game) == expected

    game = rsgame.emptygame(3, [4, 5])
    game = paygame.game_replace(game, game.all_profiles()[:21],
                                np.zeros((21, game.num_strats)))
    expected = 'Game([3 3], [4 5], 21 / 700)'
    assert repr(game) == expected


def test_game_str():
    base = rsgame.emptygame([3, 4], [3, 2])
    profs = base.all_profiles()[:21]
    pays = np.zeros_like(profs, float)
    game = paygame.game_replace(base, profs, pays)
    egame = paygame.game_copy(base)

    expected = """
Game:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
        r1:
            s3
            s4
payoff data for 0 out of 50 profiles
"""[1:-1]
    assert str(egame) == expected

    expected = """
Game:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
        r1:
            s3
            s4
payoff data for 21 out of 50 profiles
"""[1:-1]
    assert str(game) == expected


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_game_copy(role_players, role_strats):
    game = random_game(role_players, role_strats)
    copy = paygame.game_copy(game)
    assert game == copy and hash(game) == hash(copy)

    perm = rand.permutation(game.num_profiles)
    copy = paygame.game_replace(
        game, game.profiles()[perm], game.payoffs()[perm])
    assert game == copy and hash(game) == hash(copy)


def test_game_from_json():
    game = paygame.game(
        2, 2,
        [[2, 0],
         [1, 1],
         [0, 2]],
        [[0, 0],
         [10, 20],
         [0, 30]])
    for json in [_game_json, _samplegame_json, _summary_json,
                 _observations_json, _full_json]:
        assert game == paygame.game_json(json)
    game = paygame.game_copy(rsgame.emptygame_copy(game))
    for json in [_emptygame_json, _noprofs_json]:
        assert game == paygame.game_json(json)


# ----------
# SampleGame
# ----------


def test_samplegame_properties():
    game = paygame.samplegame_copy(rsgame.emptygame(2, 3))
    assert np.all([] == game.num_sample_profs)
    assert np.all([] == game.sample_starts)
    assert np.all([] == game.num_samples)

    base = rsgame.emptygame(1, [4, 3])
    game = paygame.samplegame_replace(
        base, base.all_profiles(), [np.zeros((12, 2, 7))])
    assert np.all([12] == game.num_sample_profs)
    assert np.all([0] == game.sample_starts)
    assert np.all([2] == game.num_samples)

    game = rsgame.emptygame([3, 4], [4, 3])
    profiles = game.all_profiles()[:30]
    spays = [np.zeros((9, 4, game.num_strats)),
             np.zeros((11, 1, game.num_strats)),
             np.zeros((10, 2, game.num_strats))]
    game = paygame.samplegame_replace(game, profiles, spays)
    assert np.all([9, 11, 10] == game.num_sample_profs)
    assert np.all([0, 9, 20] == game.sample_starts)
    assert np.all([4, 1, 2] == game.num_samples)


def test_empty_samplegame_resample():
    sgame = paygame.samplegame_copy(rsgame.emptygame([2, 3], [3, 2]))
    assert paygame.game_copy(sgame) == sgame.resample()
    assert paygame.game_copy(sgame) == sgame.resample(1)

    sgame = paygame.samplegame_copy(rsgame.emptygame([2, 3], [3, 2]))
    assert paygame.game_copy(sgame) == sgame.resample()
    assert paygame.game_copy(sgame) == sgame.resample(1)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_samplegame_singlesample_resample(role_players, role_strats):
    copy = random_game(role_players, role_strats)
    sgame = paygame.samplegame_copy(copy)

    game = sgame.resample()
    assert game == copy

    game = sgame.resample(100)
    assert game == copy

    game = sgame.resample(independent_role=True)
    assert game == copy
    game = sgame.resample(independent_strategy=True)
    assert game == copy
    game = sgame.resample(independent_profile=True)
    assert game == copy
    game = sgame.resample(independent_role=True, independent_strategy=True)
    assert game == copy
    game = sgame.resample(independent_role=True, independent_profile=True)
    assert game == copy
    game = sgame.resample(independent_strategy=True, independent_profile=True)
    assert game == copy
    game = sgame.resample(independent_role=True, independent_strategy=True,
                          independent_profile=True)
    assert game == copy


def test_samplegame_resample_changes():
    base = rsgame.emptygame(1, [3, 2])
    profiles = base.all_profiles()
    payoffs = rand.random((base.num_all_profiles, 1000, base.num_strats))
    payoffs *= 0 < profiles[:, None]
    sgame = paygame.samplegame_replace(base, profiles, [payoffs])
    copy = paygame.game_copy(sgame)

    # These aren't guaranteed to be true, but they're highly unlikely
    game = sgame.resample()
    assert game != copy

    game = sgame.resample(100)
    assert game != copy

    game = sgame.resample(independent_role=True)
    assert game != copy
    game = sgame.resample(independent_strategy=True)
    assert game != copy
    game = sgame.resample(independent_profile=True)
    assert game != copy
    game = sgame.resample(independent_role=True, independent_strategy=True)
    assert game != copy
    game = sgame.resample(independent_role=True, independent_profile=True)
    assert game != copy
    game = sgame.resample(independent_strategy=True, independent_profile=True)
    assert game != copy
    game = sgame.resample(independent_role=True, independent_strategy=True,
                          independent_profile=True)
    assert game != copy


def test_get_sample_payoffs():
    base = rsgame.emptygame(2, [1, 2])
    profiles = [
        [2, 2, 0],
        [2, 0, 2],
    ]
    spayoffs = [
        [
            [[5, 2, 0]],
        ],
        [
            [[5, 0, 2], [6, 0, 3]],
        ],
    ]
    game = paygame.samplegame_replace(base, profiles, spayoffs)
    pay = game.get_sample_payoffs([2, 1, 1])
    assert np.allclose(np.empty((0, 3)), pay)
    pay = game.get_sample_payoffs([2, 2, 0])
    assert np.allclose([[5, 2, 0]], pay)
    pay = game.get_sample_payoffs([2, 0, 2])
    assert np.allclose([[5, 0, 2], [6, 0, 3]], pay)

    with pytest.raises(AssertionError):
        game.get_sample_payoffs([2, 1, 2])
    with pytest.raises(AssertionError):
        game.get_sample_payoffs([2, 0, 2, 0])


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_samplegame_normalize(role_players, role_strats):
    game = random_samplegame(role_players, role_strats).normalize()
    mins = game.min_role_payoffs()
    assert np.all(np.isclose(mins, 0) | np.isnan(mins))
    maxs = game.max_role_payoffs()
    assert np.all(np.isclose(maxs, 1) | np.isclose(maxs, 0) | np.isnan(maxs))


def test_to_from_samplepay_json():
    game = paygame.samplegame_copy(rsgame.emptygame([3, 4], [2, 1]))
    prof = [3, 0, 4]
    spay = [[3, 0, 7], [4, 0, 8], [5, 0, 9]]
    json_spay = {'r0': {'s0': [3, 4, 5]}, 'r1': {'s2': [7, 8, 9]}}
    json_spay_0 = {'r0': {'s0': [3, 4, 5], 's1': [0, 0, 0]},
                   'r1': {'s2': [7, 8, 9]}}
    assert game.to_samplepay_json(spay, prof) == json_spay
    assert game.to_samplepay_json(spay) == json_spay_0
    assert np.allclose(game.from_samplepay_json(json_spay), spay)

    with pytest.raises(AssertionError):
        game.from_samplepay_json(
            json_spay, np.empty((0, 3)))

    json_profspay = {'r0': [('s0', 3, [3, 4, 5])],
                     'r1': [('s2', 4, [7, 8, 9])]}
    with pytest.raises(AssertionError):
        game.from_samplepay_json(
            json_profspay, np.empty((0, 3)))


def test_to_from_profsamplepay_json():
    game = paygame.samplegame_copy(rsgame.emptygame([3, 4], [2, 1]))
    prof = [3, 0, 4]
    spay = [[3, 0, 7], [4, 0, 8], [5, 0, 9]]
    json_profspay = {'r0': [('s0', 3, [3, 4, 5])],
                     'r1': [('s2', 4, [7, 8, 9])]}
    assert game.to_profsamplepay_json(spay, prof) == json_profspay
    p, sp = game.from_profsamplepay_json(json_profspay)
    assert np.all(p == prof)
    assert np.allclose(sp, spay)
    p = np.empty_like(p)
    _, sp = game.from_profsamplepay_json(json_profspay, dest_prof=p)
    assert np.all(p == prof)
    assert np.allclose(sp, spay)


def test_samplegame_hash_eq():
    a = paygame.samplegame_copy(rsgame.emptygame(4, 5))
    b = paygame.samplegame_copy(rsgame.emptygame([4], [5]))
    assert a == b and hash(a) == hash(b)

    a = paygame.samplegame(
        4, 2,
        [[3, 1], [2, 2]],
        [[[[1, 2]]], [[[3, 5], [4, 6]]]])
    b = paygame.samplegame(
        [4], [2],
        [[2, 2], [3, 1]],
        [[[[4, 6], [3, 5]]], [[[1, 2]]]])
    assert a == b and hash(a) == hash(b)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_samplegame_copy(role_players, role_strats):
    game = random_samplegame(role_players, role_strats)
    copy = paygame.samplegame_copy(game)
    assert game == copy and hash(game) == hash(copy)

    samp_perm = rand.permutation(game.num_samples.size)
    prof_list = np.split(game.profiles(), game.sample_starts[1:], 0)
    sprofs = []
    spays = []
    for i in samp_perm:
        perm = rand.permutation(game.num_sample_profs[i])
        sprofs.append(prof_list[i][perm])
        spays.append(game.sample_payoffs()[i][perm])

    profiles = np.concatenate(
        sprofs) if sprofs else np.empty((0, game.num_strats))
    copy = paygame.samplegame_replace(game, profiles, spays)
    assert game == copy and hash(game) == hash(copy)


# Test sample game with different number of samples
def test_samplegame_different_samples():
    base = rsgame.emptygame(1, [1, 2])
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5, 2, 0]],
        ],
        [
            [[5, 0, 2], [6, 0, 3]],
        ],
    ]
    sgame = paygame.samplegame_replace(base, profiles, payoffs)
    game = paygame.game_copy(sgame)

    assert not np.setxor1d([1, 2], sgame.num_samples).size
    # This could technically fail, but it's extremely unlikely
    assert any(game != sgame.resample()
               for _ in range(1000))  # pragma: no cover


def test_samplegame_subgame():
    profiles = [
        [2, 0, 2, 0],
        [1, 1, 2, 0],
        [0, 2, 2, 0],
    ]
    payoffs = [
        [
            [[1, 0, 2, 0]],
            [[3, 4, 5, 0]],
        ],
        [
            [[0, 6, 8, 0], [0, 7, 9, 0]],
        ]
    ]
    game = paygame.samplegame(2, [2, 2], profiles, payoffs)
    mask = [True, False, True, False]
    sprofiles = [[2, 2]]
    spayoffs = [[[[1, 2]]]]
    sgame = paygame.samplegame_names(
        ('r0', 'r1'), 2, (('s0',), ('s2',)), sprofiles, spayoffs)
    assert sgame == game.subgame(mask)

    mask = [True, True, False, True]
    sgame = paygame.samplegame_copy(rsgame.emptygame_names(
        ('r0', 'r1'), 2, (('s0', 's1'), ('s3',))))
    assert sgame == game.subgame(mask)

    profiles = [
        [2, 0, 1, 1],
        [1, 1, 1, 1],
        [0, 2, 1, 1],
    ]
    payoffs = [
        [
            [[8, 0, 9, 10]],
            [[11, 12, 13, 14]],
        ],
        [
            [[0, 15, 17, 19], [0, 16, 18, 20]],
        ]
    ]
    game = paygame.samplegame_replace(game, profiles, payoffs)
    mask = [False, True, True, True]
    sprofiles = [[2, 1, 1]]
    spayoffs = [[[[15, 17, 19], [16, 18, 20]]]]
    sgame = paygame.samplegame_names(
        ('r0', 'r1'), 2, (('s1',), ('s2', 's3')), sprofiles, spayoffs)
    assert sgame == game.subgame(mask)

    mask = [True, True, False, True]
    sgame = paygame.samplegame_copy(rsgame.emptygame_names(
        ('r0', 'r1'), 2, (('s0', 's1'), ('s3',))))
    assert sgame == game.subgame(mask)


@pytest.mark.parametrize('role_players,role_strats', testutils.games)
def test_random_samplegame_json_identity(role_players, role_strats):
    sgame = random_samplegame(role_players, role_strats)
    copy = paygame.samplegame_json(sgame.to_json())
    assert sgame == copy


def test_samplegame_repr():
    game = paygame.samplegame_copy(rsgame.emptygame(2, 3))
    expected = 'SampleGame([2], [3], 0 / 6, 0)'
    assert repr(game) == expected

    base = rsgame.emptygame(1, [4, 3])
    game = paygame.samplegame_replace(
        base, base.all_profiles(), [np.zeros((12, 2, 7))])
    expected = 'SampleGame([1 1], [4 3], 12 / 12, 2)'
    assert repr(game) == expected

    base = rsgame.emptygame(1, [1, 2])
    profiles = [
        [1, 1, 0],
        [1, 0, 1],
    ]
    payoffs = [
        [
            [[5, 2, 0]],
        ],
        [
            [[5, 0, 2], [6, 0, 3]],
        ],
    ]
    game = paygame.samplegame_replace(base, profiles, payoffs)
    expected = 'SampleGame([1 1], [1 2], 2 / 2, 1 - 2)'
    assert repr(game) == expected

    payoffs = [
        [
            [[5, 2, 0]],
        ],
        [
            [[5, 0, 2], [6, 0, 3], [7, 0, 4]],
        ],
    ]
    game = paygame.samplegame_replace(base, profiles, payoffs)
    expected = 'SampleGame([1 1], [1 2], 2 / 2, 1 - 3)'
    assert repr(game) == expected


def test_samplegame_str():
    base = rsgame.emptygame([3, 4], [3, 2])
    profs = base.all_profiles()

    game = paygame.samplegame_copy(base)
    expected = """
SampleGame:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
        r1:
            s3
            s4
payoff data for 0 out of 50 profiles
no observations
"""[1:-1]
    assert str(game) == expected

    game = paygame.samplegame_replace(base, profs[:13], [np.zeros((13, 1, 5))])
    expected = """
SampleGame:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
        r1:
            s3
            s4
payoff data for 13 out of 50 profiles
1 observation per profile
"""[1:-1]
    assert str(game) == expected

    game = paygame.samplegame_replace(base, profs[:13], [np.zeros((13, 2, 5))])
    expected = """
SampleGame:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
        r1:
            s3
            s4
payoff data for 13 out of 50 profiles
2 observations per profile
"""[1:-1]
    assert str(game) == expected

    game = paygame.samplegame_replace(base, profs[:35], [
        np.zeros((13, 2, 5)),
        np.zeros((12, 3, 5)),
        np.zeros((10, 4, 5))])
    expected = """
SampleGame:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
        r1:
            s3
            s4
payoff data for 35 out of 50 profiles
2 to 4 observations per profile
"""[1:-1]
    assert str(game) == expected

    game = paygame.samplegame_replace(base, profs[:40], [
        np.zeros((13, 2, 5)),
        np.zeros((12, 3, 5)),
        np.zeros((10, 4, 5)),
        np.zeros((5, 6, 5))])
    expected = """
SampleGame:
    Roles: r0, r1
    Players:
        3x r0
        4x r1
    Strategies:
        r0:
            s0
            s1
            s2
        r1:
            s3
            s4
payoff data for 40 out of 50 profiles
2 to 6 observations per profile
"""[1:-1]
    assert str(game) == expected


def test_samplegame_from_json():
    game = paygame.samplegame(
        2, 2,
        [[2, 0],
         [1, 1],
         [0, 2]],
        [
            [[[-1, 0], [0, 0], [1, 0]],
             [[9, 21], [10, 20], [11, 19]]],
            [[[0, 32], [0, 28], [0, 30], [0, 30]]],
        ])
    for json in [_samplegame_json, _observations_json, _full_json]:
        assert game == paygame.samplegame_json(json)
    game = paygame.samplegame_copy(paygame.game_copy(game))
    for json in [_game_json, _summary_json]:
        assert game == paygame.samplegame_json(json)
    game = paygame.samplegame_copy(rsgame.emptygame_copy(game))
    for json in [_emptygame_json, _noprofs_json]:
        assert game == paygame.samplegame_json(json)


_emptygame_json = {'players': {'r0': 2}, 'strategies': {'r0': ['s0', 's1']}}
_game_json = {'players': {'r0': 2}, 'strategies': {'r0': ['s0', 's1']}, 'profiles': [{'r0': [['s0', 2, 0]]}, {'r0': [['s0', 1, 10], ['s1', 1, 20]]}, {'r0': [['s1', 2, 30]]}]}  # noqa
_samplegame_json = {'players': {'r0': 2}, 'strategies': {'r0': ['s0', 's1']}, 'profiles': [{'r0': [['s0', 2, [-1, 0, 1]]]}, {'r0': [['s0', 1, [9, 10, 11]], ['s1', 1, [21, 20, 19]]]}, {'r0': [['s1', 2, [32, 28, 30, 30]]]}]}  # noqa
_noprofs_json = {'roles': [{'name': 'r0', 'strategies': ['s0', 's1'], 'count':2}]}  # noqa
_summary_json = {'roles': [{'name': 'r0', 'strategies': ['s0', 's1'], 'count':2}], 'profiles': [{'symmetry_groups': [{'payoff': 0, 'count': 2, 'strategy': 's0', 'role': 'r0'}]}, {'symmetry_groups': [{'payoff': 10, 'count': 1, 'strategy': 's0', 'role': 'r0'}, {'payoff': 20, 'count': 1, 'strategy': 's1', 'role': 'r0'}]}, {'symmetry_groups': [{'payoff': 30, 'count': 2, 'strategy': 's1', 'role': 'r0'}]}]}  # noqa
_observations_json = {'roles': [{'name': 'r0', 'strategies': ['s0', 's1'], 'count':2}], 'profiles': [{'symmetry_groups': [{'strategy': 's0', 'id': 0, 'role': 'r0', 'count': 2}], 'observations': [{'symmetry_groups': [{'id': 0, 'payoff': -1}]}, {'symmetry_groups': [{'id': 0, 'payoff': 0}]}, {'symmetry_groups': [{'id': 0, 'payoff': 1}]}]}, {'symmetry_groups': [{'strategy': 's0', 'id': 1, 'role': 'r0', 'count': 1}, {'strategy': 's1', 'id': 2, 'role': 'r0', 'count': 1}], 'observations': [{'symmetry_groups': [{'id': 1, 'payoff': 9}, {'id': 2, 'payoff': 21}]}, {'symmetry_groups': [{'id': 1, 'payoff': 10}, {'id': 2, 'payoff': 20}]}, {'symmetry_groups': [{'id': 1, 'payoff': 11}, {'id': 2, 'payoff': 19}]}]}, {'symmetry_groups': [{'strategy': 's1', 'id': 3, 'role': 'r0', 'count': 2}], 'observations': [{'symmetry_groups': [{'id': 3, 'payoff': 32}]}, {'symmetry_groups': [{'id': 3, 'payoff': 28}]}, {'symmetry_groups': [{'id': 3, 'payoff': 30}]}, {'symmetry_groups': [{'id': 3, 'payoff': 30}]}]}]}  # noqa
_full_json = {'roles': [{'name': 'r0', 'strategies': ['s0', 's1'], 'count':2}], 'profiles': [{'symmetry_groups': [{'strategy': 's0', 'id': 0, 'role': 'r0', 'count': 2}], 'observations': [{'players': [{'sid': 0, 'p': -2}, {'sid': 0, 'p': 0}]}, {'players': [{'sid': 0, 'p': 0}, {'sid': 0, 'p': 0}]}, {'players': [{'sid': 0, 'p': 0}, {'sid': 0, 'p': 2}]}]}, {'symmetry_groups': [{'strategy': 's0', 'id': 1, 'role': 'r0', 'count': 1}, {'strategy': 's1', 'id': 2, 'role': 'r0', 'count': 1}], 'observations': [{'players': [{'sid': 1, 'p': 9}, {'sid': 2, 'p': 21}]}, {'players': [{'sid': 1, 'p': 10}, {'sid': 2, 'p': 20}]}, {'players': [{'sid': 1, 'p': 11}, {'sid': 2, 'p': 19}]}]}, {'symmetry_groups': [{'strategy': 's1', 'id': 3, 'role': 'r0', 'count': 2}], 'observations': [{'players': [{'sid': 3, 'p': 32}, {'sid': 3, 'p': 32}]}, {'players': [{'sid': 3, 'p': 30}, {'sid': 3, 'p': 26}]}, {'players': [{'sid': 3, 'p': 34}, {'sid': 3, 'p': 26}]}, {'players': [{'sid': 3, 'p': 28}, {'sid': 3, 'p': 32}]}]}]}  # noqa
