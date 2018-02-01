import numpy as np
import pytest
from collections import abc

from gameanalysis import agggen
from gameanalysis import gamegen
from gameanalysis import regret
from gameanalysis import rsgame
from gameanalysis import utils


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('strategies', [
    [1],
    [2],
    [1, 1],
    [2, 2],
    [4, 4, 4],
    [1, 3],
])
def test_independent_game(strategies, _):
    game = gamegen.independent_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == len(strategies), \
        "didn't generate correct number of players"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
])
def test_game(players, strategies, _):
    game = gamegen.game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(players == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
])
def test_sparse_game(players, strategies, _):
    game = gamegen.sparse_game(players, strategies, 1)
    assert not game.is_empty()
    assert game.num_profiles == 1
    assert np.all(players == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
])
def test_samplegame(players, strategies, _):
    game = gamegen.samplegame(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(players == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


def test_sparse_samplegame():
    game = gamegen.samplegame([4, 4], [4, 4], 0.5, 0)
    # Very unlikely to fail
    assert not game.is_complete()


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
])
def test_gen_profiles(players, strategies, _):
    base = rsgame.emptygame(players, strategies)
    game = gamegen.gen_profiles(base)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(players == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"

    game = gamegen.gen_profiles(base, 0.0)
    assert game.is_empty(), "didn't generate a full game"

    game = gamegen.gen_profiles(base, 0.5)

    game = gamegen.gen_num_profiles(base, base.num_all_profiles // 2)
    assert game.num_profiles == game.num_all_profiles // 2


def test_gen_profiles_large_game():
    base = rsgame.emptygame([100] * 2, 30)
    game = gamegen.gen_profiles(base, 1e-55)
    assert game.num_profiles == 363


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', [
    ([3], [2]),
    ([2, 4], [3, 3]),
    ([1, 4], [2, 2]),
    ([2, 4], [1, 2]),
    ([1, 4], [1, 2]),
])
def test_keep_profiles(players, strategies, _):
    game = gamegen.game(players, strategies)
    test = gamegen.keep_num_profiles(game, 4)
    assert test.num_profiles == 4

    test = gamegen.keep_profiles(game, 0.0)
    assert test.is_empty(), "didn't generate a full game"
    test = gamegen.keep_num_profiles(game, 0)
    assert test.is_empty(), "didn't generate a full game"

    gamegen.keep_profiles(game, 0.5)


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', [
    ([3], [2]),
    ([2, 4], [3, 3]),
    ([1, 4], [2, 2]),
    ([2, 4], [1, 2]),
    ([1, 4], [1, 2]),
])
def test_keep_num_profiles(players, strategies, _):
    game = gamegen.game(players, strategies, 0.5)
    num = game.num_profiles // 2
    test = gamegen.keep_num_profiles(game, num)
    assert test.num_profiles == num


def test_keep_profiles_large_game():
    base = agggen.normal_aggfn([100] * 2, 30, 10)
    game = gamegen.keep_profiles(base, 1e-55)
    assert game.num_profiles == 363

    game = gamegen.keep_num_profiles(base, 362)
    assert game.num_profiles == 362


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('strategies', [
    [1],
    [2],
    [1, 1],
    [2, 2],
    [4] * 3,
    [1, 3],
])
def test_covariant_game(strategies, _):
    game = gamegen.covariant_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('strategies', [1, 2, 4, 6] * 20)
def test_two_player_zero_sum_game(strategies):
    game = gamegen.two_player_zero_sum_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == 2, "not two player"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate right number of strategies"
    assert game.is_constant_sum(), "game not constant sum"


@pytest.mark.parametrize('_', range(20))
def test_chicken(_):
    game = gamegen.chicken()
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_symmetric(), \
        "didn't generate a symmetric game"
    assert np.all(2 == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(2 == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('_', range(20))
def test_prisonzers_dilemma(_):
    game = gamegen.prisoners_dilemma()
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_symmetric(), \
        "didn't generate a symmetric game"
    assert np.all(2 == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(2 == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('eq_prob', (p / 10 for p in range(11)))
def test_sym_2p2s_known_eq(eq_prob):
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_symmetric(), \
        "didn't generate a symmetric game"
    assert np.all(2 == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(2 == game.num_role_strats), \
        "didn't generate correct number of strategies"
    eqm = np.array([eq_prob, 1 - eq_prob])
    reg = regret.mixture_regret(game, eqm)
    assert np.isclose(reg, 0), \
        "expected equilibrium wasn't an equilibrium, reg: {}".format(reg)
    for non_eqm in game.pure_mixtures():
        reg = regret.mixture_regret(game, non_eqm)
        # If eq_prob is 0 or 1, then pure is the desired mixture
        assert non_eqm[0] == eq_prob or not np.isclose(reg, 0), \
            "pure mixtures was equilibrium, {} {}".format(non_eqm, reg)


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies,matrix_players', [
    (1, 1, 1),
    (2, 1, 1),
    (2, 1, 2),
    (1, 3, 1),
    (3, 3, 2),
    (3, 3, 3),
])
def test_polymatrix_game(players, strategies, matrix_players, _):
    game = gamegen.polymatrix_game(players, strategies,
                                   players_per_matrix=matrix_players)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies,lower,prob', [
    (2 * [1], 1, 1, 0),
    (2 * [1], 1, 0, 1 / 3),
    (2 * [1], 2, 1, 0),
    (2 * [1], 2, 0, 1 / 3),
    (2 * [2], 1, 1, 0),
    (2 * [2], 1, 0, 1 / 3),
    (2 * [2], 2, 1, 0),
    (2 * [2], 2, 0, 1 / 3),
    ([3], 4, 1, 0),
    ([3], 4, 0, 1 / 3),
])
def test_gen_noise(players, strategies, lower, prob, _):
    roles = max(np.array(players).size, np.array(strategies).size)
    base_game = gamegen.game(players, strategies)
    game = gamegen.gen_noise(base_game, prob, lower)
    assert lower == 0 or game.is_complete(), "didn't generate a full game"
    assert game.num_roles == roles, \
        "didn't generate correct number of players"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"
    assert np.all(game.num_samples >= min(lower, 1)), \
        "didn't generate appropriate number of samples"


def test_empty_add_noise():
    base_game = rsgame.emptygame([3, 3], [4, 4])
    game = gamegen.gen_noise(base_game)
    assert game.is_empty()

    base_game = gamegen.game([3] * 3, 4)
    game = gamegen.gen_noise(base_game, 0, 0)
    assert game.is_empty()


def _first(val):
    """Retrun first val if its an iterable"""
    if isinstance(val, abc.Iterable):
        return next(iter(val))
    else:
        return val


@pytest.mark.parametrize('win,loss', [
    (1, -1),
    (2, -1),
    ([2, 2, 3], -1),
    (1, -2),
    (1, [-2, -2, -3]),
])
def test_rock_paper_scissors(win, loss):
    game = gamegen.rock_paper_scissors(win, loss)
    assert game.strat_names == (('paper', 'rock', 'scissors'),)
    assert np.allclose(game.get_payoffs([1, 1, 0]),
                       [_first(loss), _first(win), 0])


def test_rock_paper_scissors_defaults():
    game = gamegen.rock_paper_scissors()
    assert np.allclose(game.get_payoffs([1, 1, 0]), [-1, 1, 0])


def test_travellers_dilemma():
    game = gamegen.travellers_dilemma(2, 10)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(2 == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(9 == game.num_role_strats), \
        "didn't generate correct number of strategies"
    assert game.num_profiles == 45


FUNCTIONS = [
    gamegen.width_gaussian,
    gamegen.width_bimodal,
    gamegen.width_uniform,
    gamegen.width_gumbel,
]


@pytest.mark.parametrize('func', FUNCTIONS)
def test_width_distribution(func):
    widths = np.random.uniform(0, 1, 5)
    stddevs = np.std(func(widths, 100000), 1)
    assert np.allclose(widths, stddevs, rtol=0.05)


@pytest.mark.parametrize('players,strats', [([2], [3]), ([2, 3], [3, 2])])
@pytest.mark.parametrize('func', FUNCTIONS)
def test_add_widths(players, strats, func):
    sgame = gamegen.samplegame(players, strats, noise_distribution=func)
    assert sgame.is_complete()


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strats', [([2], [3]), ([2, 3], [3, 2])])
def test_sample_profiles(players, strats, _):
    game = gamegen.game(players, strats)
    profiles = gamegen.sample_profiles(game, 5)
    uprofs = utils.unique_axis(profiles)
    assert uprofs.shape == (5, game.num_strats)
