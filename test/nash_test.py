from gameanalysis import nash
from gameanalysis import randgames

from test import testutils


@testutils.apply([
    (1, 1, [1]),
    (1, 2, [2]),
    (2, 1, [1, 1]),
    (2, 2, [2, 2]),
    (3, 4, [4, 4, 4]),
    (2, [1, 3], [1, 3]),
])
def mixed_nash_test(players, strategies, exp_strats):
    game = randgames.independent_game(players, strategies)
    eqa = list(nash.mixed_nash(game, at_least_one=True, max_iters=100))
    assert eqa, "Didn't find an equilibria with at_least_one on"
