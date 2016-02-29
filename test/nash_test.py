import sys
from unittest import mock

import numpy as np

from gameanalysis import nash
from gameanalysis import randgames
from gameanalysis import regret
from gameanalysis import rsgame
from test import testutils


ROSHAMBO = rsgame.Game.from_payoff_format(
    {'a': 2},
    {'a': ['r', 'p', 's']},
    [
        {'a': [('r', 2, [0])]},
        {'a': [('p', 2, [0])]},
        {'a': [('s', 2, [0])]},
        {'a': [('r', 1, [1]), ('s', 1, [-1])]},
        {'a': [('s', 1, [1]), ('p', 1, [-1])]},
        {'a': [('p', 1, [1]), ('r', 1, [-1])]},
    ]
)

HARD = rsgame.Game.from_payoff_format(
    {'a': 2},
    {'a': ['r', 'p', 's', ':(']},
    [
        {'a': [('r', 2, [0])]},
        {'a': [('p', 2, [0])]},
        {'a': [('s', 2, [0])]},
        {'a': [(':(', 2, [0])]},
        {'a': [('r', 1, [1]), ('s', 1, [-1])]},
        {'a': [('s', 1, [1]), ('p', 1, [-1])]},
        {'a': [('p', 1, [1]), ('r', 1, [-1])]},
        {'a': [('r', 1, [1]), (':(', 1, [-1])]},
        {'a': [('p', 1, [1]), (':(', 1, [-1])]},
        {'a': [('s', 1, [1]), (':(', 1, [-1])]},
    ]
)


@testutils.apply(repeat=20)
def pure_prisoners_dilemma_test():
    game = randgames.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    eqa = list(nash.pure_nash(game))

    role = next(iter(game.strategies))
    strats = list(game.strategies[role])

    assert len(eqa) == 1, "didn't find exactly one equilibria in pd"
    assert {role: {strats[1]: 2}} == eqa[0], \
        "didn't find pd equilibrium"


@testutils.apply(repeat=20)
def mixed_prisoners_dilemma_test():
    game = randgames.sym_2p2s_game(2, 0, 3, 1)  # prisoners dilemma
    eqa = list(nash.mixed_nash(game))

    role = next(iter(game.strategies))
    strats = list(game.strategies[role])

    assert len(eqa) == 1, "didn't find exactly one equilibria in pd"
    assert {role: {strats[1]: 1}} == eqa[0], \
        "didn't find pd equilibrium"


def pure_roshambo_test():
    eqa = list(nash.pure_nash(ROSHAMBO))
    assert len(eqa) == 0, "found a pure equilibrium in roshambo"
    eqa = list(nash.pure_nash(ROSHAMBO, 1))
    assert len(eqa) == 3, \
        "didn't find low regret ties in roshambo"
    eqa = list(nash.pure_nash(ROSHAMBO, 2))
    assert len(eqa) == ROSHAMBO.size, \
        "found profiles with more than 2 regret in roshambo"


def minreg_roshambo_test():
    eqm = nash.min_regret_profile(ROSHAMBO)
    assert 2 == next(iter(next(iter(eqm.values())).values())), \
        "min regret profile was not rr, pp, or ss"


def minreg_grid_roshambo_test():
    eqm = nash.min_regret_grid_mixture(ROSHAMBO, 3)  # Not enough for eq
    assert abs(regret.mixture_regret(ROSHAMBO, eqm) - .5) < 1e-7, \
        "min regret grid didn't find [.5, .5, 0] profile with regret .5"
    eqm = nash.min_regret_grid_mixture(ROSHAMBO, 4)  # hit eqa perfectly
    assert abs(regret.mixture_regret(ROSHAMBO, eqm) - 0) < 1e-7, \
        "min regret grid didn't find equilibrium"


def minreg_rand_roshambo_test():
    eqm = nash.min_regret_rand_mixture(ROSHAMBO, 20)
    assert regret.mixture_regret(ROSHAMBO, eqm) < 2 + 1e-7, \
        "Found a mixture with greater than maximum regret"


def mixed_roshambo_test():
    eqa = list(nash.mixed_nash(ROSHAMBO))
    assert len(eqa) == 1, "didn't find right number of equilibria in roshambo"
    assert np.allclose(np.array([[1/3]*3]), ROSHAMBO.as_array(eqa[0])), \
        "roshambo equilibria wasn't uniform"


def at_least_one_test():
    eqa = list(nash.mixed_nash(HARD, max_iters=1))
    assert len(eqa) == 0, "found an equilibrium normally"
    eqa = list(nash.mixed_nash(HARD, max_iters=1, at_least_one=True))
    assert len(eqa) == 1, "at_least_one didn't return anything"


@mock.patch.object(sys, 'stderr')
def mixed_verbose_test(mock):
    # Next necessary to execute function due to yield
    assert not mock.write.called, "wrote to err before call"
    next(nash.mixed_nash(HARD))
    assert not mock.write.called, "wrote to err without verbose"
    next(nash.mixed_nash(HARD, verbose=True))
    assert mock.write.called, "did not write to error"


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
