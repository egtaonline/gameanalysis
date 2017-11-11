import numpy as np
import pytest
from collections import abc

from gameanalysis import agggen
from gameanalysis import gamegen
from gameanalysis import regret
from gameanalysis import rsgame


@pytest.mark.parametrize('strategies', [
    [1],
    [2],
    [1, 1],
    [2, 2],
    [4, 4, 4],
    [1, 3],
] * 20)
def test_independent_game(strategies):
    game = gamegen.independent_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == len(strategies), \
        "didn't generate correct number of players"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
] * 20)
def test_role_symmetric_game(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(players == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
] * 20)
def test_add_profiles(players, strategies):
    base = rsgame.emptygame(players, strategies)
    game = gamegen.add_profiles(base)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(players == game.num_role_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"

    game = gamegen.add_profiles(base, 0.0)
    assert game.is_empty(), "didn't generate a full game"

    game = gamegen.add_profiles(base, 0.5)

    game = gamegen.add_profiles(base, base.num_all_profiles // 2)
    assert game.num_profiles == game.num_all_profiles // 2


def test_add_profiles_large_game():
    base = rsgame.emptygame([100] * 2, 30)
    game = gamegen.add_profiles(base, 1e-55)
    assert game.num_profiles == 363


@pytest.mark.parametrize('players,strategies', [
    ([3], [2]),
    ([2, 4], [3, 3]),
    ([1, 4], [2, 2]),
    ([2, 4], [1, 2]),
    ([1, 4], [1, 2]),
] * 20)
def test_drop_profiles(players, strategies):
    base = rsgame.emptygame(players, strategies)
    game = gamegen.add_profiles(base)
    test = gamegen.drop_profiles(game, 4)
    assert test.num_profiles == 4

    test = gamegen.drop_profiles(game, 0.0)
    assert test.is_empty(), "didn't generate a full game"
    test = gamegen.drop_profiles(game, 0)
    assert test.is_empty(), "didn't generate a full game"

    gamegen.drop_profiles(game, 0.5)


def test_drop_profiles_large_game():
    # TODO Switch to a "RandomGame" when implemented
    base = agggen.random_aggfn([100] * 2, 30, 10)
    game = gamegen.drop_profiles(base, 1e-55)
    assert game.num_profiles == 363


@pytest.mark.parametrize('strategies', [
    [1],
    [2],
    [1, 1],
    [2, 2],
    [4] * 3,
    [1, 3],
] * 20)
def test_covariant_game(strategies):
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
def test_sym_2p2s_game(_):
    game = gamegen.sym_2p2s_game()
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


@pytest.mark.parametrize('players,strategies,matrix_players', [
    (1, 1, 1),
    (2, 1, 1),
    (2, 1, 2),
    (1, 3, 1),
    (3, 3, 2),
    (3, 3, 3),
] * 20)
def test_polymatrix_game(players, strategies, matrix_players):
    game = gamegen.polymatrix_game(players, strategies,
                                   players_per_matrix=matrix_players)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"


@pytest.mark.parametrize('players,strategies,lower,upper', [
    (2 * [1], 1, 1, 1),
    (2 * [1], 1, 0, 3),
    (2 * [1], 2, 1, 1),
    (2 * [1], 2, 0, 3),
    (2 * [2], 1, 1, 1),
    (2 * [2], 1, 0, 3),
    (2 * [2], 2, 1, 1),
    (2 * [2], 2, 0, 3),
    ([3], 4, 1, 1),
    ([3], 4, 0, 3),
] * 20)
def test_add_noise(players, strategies, lower, upper):
    roles = max(np.array(players).size, np.array(strategies).size)
    base_game = gamegen.role_symmetric_game(players, strategies)
    game = gamegen.add_noise(base_game, lower, upper)
    assert lower == 0 or game.is_complete(), "didn't generate a full game"
    assert game.num_roles == roles, \
        "didn't generate correct number of players"
    assert np.all(strategies == game.num_role_strats), \
        "didn't generate correct number of strategies"
    assert (np.all(game.num_samples >= min(lower, 1)) and
            np.all(game.num_samples <= upper)), \
        "didn't generate appropriate number of samples"


def test_empty_add_noise():
    base_game = rsgame.emptygame([3, 3], [4, 4])
    game = gamegen.add_noise(base_game, 1)
    assert game.is_empty()

    base_game = gamegen.role_symmetric_game([3] * 3, 4)
    game = gamegen.add_noise(base_game, 0)
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
    gamegen.width_gaussian_old(),
    gamegen.width_gaussian_old(0.1),
    gamegen.width_bimodal,
    gamegen.width_bimodal_old(),
    gamegen.width_bimodal_old(0.1),
    gamegen.width_uniform,
    gamegen.width_gumbel,
]


@pytest.mark.parametrize('max_width', [0.1, 1])
@pytest.mark.parametrize('num_profiles', [1, 10, 100])
@pytest.mark.parametrize('num_samples', [1, 10, 100])
@pytest.mark.parametrize('func', FUNCTIONS)
def test_width_distribution(max_width, num_profiles, num_samples, func):
    samples = func(max_width, num_profiles, num_samples)
    assert samples.shape == (num_profiles, num_samples)


@pytest.mark.parametrize('max_width', [0.1, 1])
@pytest.mark.parametrize('game_desc', [([2], [3]), ([2, 3], [3, 2])])
@pytest.mark.parametrize('num_samples', [1, 10, 100])
@pytest.mark.parametrize('func', FUNCTIONS)
def test_add_width(game_desc, max_width, num_samples, func):
    game = gamegen.role_symmetric_game(*game_desc)
    sgame = gamegen.add_noise_width(game, num_samples, max_width, func)
    assert np.all(sgame.num_samples == num_samples)
