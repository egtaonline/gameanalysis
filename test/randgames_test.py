import itertools

import scipy.misc as scm

from gameanalysis import randgames

from test import testutils


@testutils.apply([
    (1, 1, [1]),
    (1, 2, [2]),
    (2, 1, [1, 1]),
    (2, 2, [2, 2]),
    (3, 4, [4, 4, 4]),
    (2, [1, 3], [1, 3]),
], repeat=20)
def independent_game_test(players, strategies, exp_strats):
    game = randgames.independent_game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == players, \
        "didn't generate correct number of players"
    assert all(p == 1 for p in game.players.values()), \
        "didn't generate an asymmetric game"
    assert all(len(s) == e for e, s in zip(exp_strats,
                                           game.strategies.values())), \
        "didn't generate correct number of strategies"


@testutils.apply([
    (1, 1, 1, [1], [1]),
    (3, 1, 2, [1, 1, 1], [2, 2, 2]),
    (1, 3, 2, [3], [2]),
    (2, 2, 3, [2, 2], [3, 3]),
    (2, [1, 2], 2, [1, 2], [2, 2]),
    (2, 2, [1, 2], [2, 2], [1, 2]),
    (2, [1, 2], [1, 2], [1, 2], [1, 2]),
], repeat=20)
def role_symmetric_game_test(roles, players, strategies, exp_players,
                             exp_strats):
    game = randgames.role_symmetric_game(roles, players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == roles, \
        "didn't generate correct number of players"
    assert all(p == e for e, p in zip(exp_players, game.players.values())), \
        "didn't generate correct number of strategies"
    assert all(len(s) == e for e, s in zip(exp_strats,
                                           game.strategies.values())), \
        "didn't generate correct number of strategies"


@testutils.apply([
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (3, 4),
], repeat=20)
def symmetric_game_test(players, strategies):
    game = randgames.symmetric_game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == 1, \
        "didn't generate correct number of players"
    assert all(p == players for p in game.players.values()), \
        "didn't generate correct number of strategies"
    assert all(len(s) == strategies for s in game.strategies.values()), \
        "didn't generate correct number of strategies"


@testutils.apply([
    (1, 1, [1]),
    (1, 2, [2]),
    (2, 1, [1, 1]),
    (2, 2, [2, 2]),
    (3, 4, [4, 4, 4]),
    (2, [1, 3], [1, 3]),
], repeat=20)
def covariant_game_test(players, strategies, exp_strats):
    game = randgames.covariant_game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == players, \
        "didn't generate correct number of players"
    assert all(len(s) == e for e, s in zip(exp_strats,
                                           game.strategies.values())), \
        "didn't generate correct number of strategies"


@testutils.apply([
    [1],
    [2],
    [4],
    [6],
], repeat=20)
def zero_sum_game_test(strategies):
    game = randgames.zero_sum_game(strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == 2, "not two player"
    assert all(p == 1 for p in game.players.values()), \
        "not appropriate player number"
    assert all(len(s) == strategies for s in game.strategies.values()), \
        "didn't generate right number of strategies"
    assert all(sum(itertools.chain.from_iterable(s.values()
                                                 for s in p.values())) == 0
               for _, p in game.payoffs())


@testutils.apply(repeat=20)
def sym_2p2s_game_test():
    game = randgames.sym_2p2s_game()
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == 1, \
        "didn't generate correct number of players"
    assert all(p == 2 for p in game.players.values()), \
        "didn't generate correct number of strategies"
    assert all(len(s) == 2 for s in game.strategies.values()), \
        "didn't generate correct number of strategies"


@testutils.apply([
    (1, 1, 1),
    (2, 1, 1),
    (1, 3, 3),
    (1, 3, 2),
    (3, 3, 3),
    (3, 3, 2),
], repeat=20)
def congestion_game_test(players, facilities, required):
    game = randgames.congestion_game(players, facilities, required)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == 1, \
        "didn't generate correct number of players"
    assert players == next(iter(game.players.values())), \
        "didn't generate correct number of strategies"
    assert scm.comb(facilities, required) == \
        len(next(iter(game.strategies.values()))), \
        "didn't generate correct number of strategies"


@testutils.apply([
    (1, 1),
    (2, 1),
    (1, 3),
    (3, 3),
], repeat=20)
def local_effect_game_test(players, strategies):
    game = randgames.local_effect_game(players, strategies)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == 1, \
        "didn't generate correct number of players"
    assert players == next(iter(game.players.values())), \
        "didn't generate correct number of strategies"
    assert strategies == len(next(iter(game.strategies.values()))), \
        "didn't generate correct number of strategies"


@testutils.apply([
    (1, 1, 1),
    (2, 1, 1),
    (2, 1, 2),
    (1, 3, 1),
    (3, 3, 2),
    (3, 3, 3),
], repeat=20)
def polymatrix_game_test(players, strategies, matrix_players):
    game = randgames.polymatrix_game(players, strategies,
                                     players_per_matrix=matrix_players)
    assert game.is_complete(), "didn't generate a full game"
    assert len(game.strategies) == players, \
        "didn't generate correct number of players"
    assert all(p == 1 for p in game.players.values()), \
        "didn't generate an asymmetric game"
    assert all(len(s) == strategies for s in game.strategies.values()), \
        "didn't generate correct number of strategies"
