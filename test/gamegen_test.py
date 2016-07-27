import numpy as np
import scipy.misc as scm

from gameanalysis import gamegen
from gameanalysis import regret
from gameanalysis import rsgame
from test import testutils


@testutils.apply([
    ([1],),
    ([2],),
    ([1, 1],),
    ([2, 2],),
    ([4, 4, 4],),
    ([1, 3],),
], repeat=20)
def test_independent_game(strategies):
    game = gamegen.independent_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == len(strategies), \
        "didn't generate correct number of players"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate correct number of strategies"


@testutils.apply([
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
], repeat=20)
def test_role_symmetric_game(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(players == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate correct number of strategies"

    conv = gamegen.game_serializer(game)
    assert all(r.startswith('r') for r in conv.role_names)
    assert all(all(s.startswith('s') for s in strats)
               for strats in conv.strat_names)


@testutils.apply([
    ([1], [1]),
    ([1] * 3, [2] * 3),
    ([3], [2]),
    ([2, 2], [3, 3]),
    ([1, 2], [2, 2]),
    ([2, 2], [1, 2]),
    ([1, 2], [1, 2]),
], repeat=20)
def test_add_profiles(players, strategies):
    base = rsgame.BaseGame(players, strategies)
    game = gamegen.add_profiles(base)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(players == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate correct number of strategies"

    game = gamegen.add_profiles(base, 0.0)
    assert game.is_empty(), "didn't generate a full game"

    game = gamegen.add_profiles(base, 0.5)

    game = gamegen.add_profiles(base, base.num_all_profiles // 2)
    assert game.num_profiles == game.num_all_profiles // 2


def test_add_profiles_large_game():
    base = rsgame.BaseGame([100] * 2, 30)
    game = gamegen.add_profiles(base, 1e-55)
    assert game.num_profiles == 363


@testutils.apply([
    ([1],),
    ([2],),
    ([1, 1],),
    ([2, 2],),
    ([4] * 3,),
    ([1, 3],),
], repeat=20)
def test_covariant_game(strategies):
    game = gamegen.covariant_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate correct number of strategies"


@testutils.apply([
    [1],
    [2],
    [4],
    [6],
], repeat=20)
def test_two_player_zero_sum_game(strategies):
    game = gamegen.two_player_zero_sum_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == 2, "not two player"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate right number of strategies"
    assert game.is_constant_sum(), "game not constant sum"


@testutils.apply(repeat=20)
def test_sym_2p2s_game():
    game = gamegen.sym_2p2s_game()
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_symmetric(), \
        "didn't generate a symmetric game"
    assert np.all(2 == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(2 == game.num_strategies), \
        "didn't generate correct number of strategies"


@testutils.apply(repeat=20)
def test_prisonzers_dilemma():
    game = gamegen.prisoners_dilemma()
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_symmetric(), \
        "didn't generate a symmetric game"
    assert np.all(2 == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(2 == game.num_strategies), \
        "didn't generate correct number of strategies"


@testutils.apply(zip(p / 10 for p in range(11)))
def test_sym_2p2s_known_eq(eq_prob):
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_symmetric(), \
        "didn't generate a symmetric game"
    assert np.all(2 == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(2 == game.num_strategies), \
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


@testutils.apply([
    (1, 1, 1),
    (2, 1, 1),
    (1, 3, 3),
    (1, 3, 2),
    (3, 3, 3),
    (3, 3, 2),
], repeat=20)
def test_congestion_game(players, facilities, required):
    game = gamegen.congestion_game(players, facilities, required)
    assert game.is_complete(), "didn't generate a full game"
    assert game.num_roles == 1, \
        "didn't generate correct number of players"
    assert np.all(players == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(scm.comb(facilities, required) == game.num_strategies), \
        "didn't generate correct number of strategies"


def test_congestion_game_names():
    game, conv = gamegen.congestion_game(3, 3, 2, return_serial=True)
    assert conv.role_names == ('all',)
    assert all(s.count('_') == 2 - 1 for s in conv.strat_names[0])


@testutils.apply([
    (1, 1),
    (2, 1),
    (1, 3),
    (3, 3),
], repeat=20)
def test_local_effect_game(players, strategies):
    game = gamegen.local_effect_game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_symmetric(), \
        "didn't generate a symmetric game"
    assert np.all(players == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate correct number of strategies"


@testutils.apply([
    (1, 1, 1),
    (2, 1, 1),
    (2, 1, 2),
    (1, 3, 1),
    (3, 3, 2),
    (3, 3, 3),
], repeat=20)
def test_polymatrix_game(players, strategies, matrix_players):
    game = gamegen.polymatrix_game(players, strategies,
                                   players_per_matrix=matrix_players)
    assert game.is_complete(), "didn't generate a full game"
    assert game.is_asymmetric(), \
        "didn't generate an asymmetric game"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate correct number of strategies"


@testutils.apply([
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
], repeat=20)
def test_add_noise(players, strategies, lower, upper):
    roles = max(np.array(players).size, np.array(strategies).size)
    base_game = gamegen.role_symmetric_game(players, strategies)
    game = gamegen.add_noise(base_game, lower, upper)
    assert lower == 0 or game.is_complete(), "didn't generate a full game"
    assert game.num_roles == roles, \
        "didn't generate correct number of players"
    assert np.all(strategies == game.num_strategies), \
        "didn't generate correct number of strategies"
    assert (np.all(game.num_samples >= min(lower, 1)) and
            np.all(game.num_samples <= upper)), \
        "didn't generate appropriate number of samples"


def test_empty_add_noise():
    base_game = rsgame.Game([3, 3], [4, 4])
    game = gamegen.add_noise(base_game, 1)
    assert game.is_empty()

    base_game = gamegen.role_symmetric_game([3] * 3, 4)
    game = gamegen.add_noise(base_game, 0)
    assert game.is_empty()


@testutils.apply([
    (1, 1),
    ([1] * 3, 2),
    (3, 2),
    ([2, 2], 3),
    ([1, 2], 2),
    (2, [1, 2]),
    ([1, 2], [1, 2]),
], repeat=20)
def test_drop_profiles(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    # Since independent drops might drop nothing, we keep nothing
    dropped = gamegen.drop_profiles(game, 0)
    assert dropped.is_empty(), "didn't drop any profiles"
    # 40% mean even one profile games will be incomplete
    dropped = gamegen.drop_profiles(game, 0.4, independent=False)
    assert not dropped.is_complete(), "didn't drop any profiles"

    sgame = gamegen.add_noise(game, 3)
    dropped = gamegen.drop_profiles(sgame, 0)
    assert dropped.is_empty(), "didn't drop any profiles"
    # 40% mean even one profile games will be incomplete
    dropped = gamegen.drop_profiles(sgame, 0.4, independent=False)
    assert not dropped.is_complete(), "didn't drop any profiles"


@testutils.apply([
    (1, 1),
    ([1] * 3, 2),
    (3, 2),
    ([2] * 2, 3),
    ([1, 2], 2),
    (2, [1, 2]),
    ([1, 2], [1, 2]),
], repeat=20)
def test_drop_samples(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    num_samples = 10000 // game.num_profiles
    game = gamegen.add_noise(game, num_samples)
    # Since independent drops might drop nothing, we keep nothing
    dropped = gamegen.drop_samples(game, 0)
    assert dropped.is_empty(), "didn't drop any profiles"
    # 40% mean even one profile games will be incomplete
    dropped = gamegen.drop_samples(game, 1)
    assert (dropped.is_complete() and
            np.all(dropped.num_samples == [num_samples]))
    # We drop half of samples, meaning is highly unlikely the game is complete
    # or empty, but these "can" still happen
    dropped = gamegen.drop_samples(game, .5)
    assert (not dropped.is_complete() or
            not np.all(dropped.num_samples == [num_samples]))
    assert not dropped.is_empty()


@testutils.apply([
    (1, -1),
    (2, -1),
    (1, -2),
])
def test_rock_paper_scissors(win, loss):
    game, conv = gamegen.rock_paper_scissors(win, loss, return_serial=True)
    assert conv.strat_names == (('rock', 'paper', 'scissors'),)
    assert np.allclose(game.get_payoffs([1, 1, 0]), [loss, win, 0])


def test_rock_paper_scissors_defaults():
    game = gamegen.rock_paper_scissors()
    assert np.allclose(game.get_payoffs([1, 1, 0]), [-1, 1, 0])


def test_travellers_dilemma():
    game = gamegen.travellers_dilemma(2, 10)
    assert game.is_complete(), "didn't generate a full game"
    assert np.all(2 == game.num_players), \
        "didn't generate correct number of strategies"
    assert np.all(9 == game.num_strategies), \
        "didn't generate correct number of strategies"
    assert game.num_profiles == 45
