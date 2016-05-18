import random

from gameanalysis import gamegen
from gameanalysis import reduction
from gameanalysis import subgame

from test import testutils


def pure_subgame_test():
    game = gamegen.empty_role_symmetric_game(2, [3, 4], [3, 2])
    subgames = list(subgame.pure_subgames(game))
    assert len(subgames) == len(game.pure_mixtures(as_array=True))


def empty_subgame_test():
    game = gamegen.empty_role_symmetric_game(2, [3, 4], [3, 2])
    subg = subgame.EmptySubgame(game, {'r0': ['s0', 's2'], 'r1': ['s1']})
    devs = len(list(subg.deviation_profiles()))
    assert devs == 7, "didn't generate the right number of deviating profiles"
    adds = len(list(subg.additional_strategy_profiles('r0', 's1')))
    assert adds == 6, \
        "didn't generate the right number of additional profiles"
    subg2 = subg.add_strategy('r0', 's1')
    assert subg2.size == adds + subg.size, \
        "additional profiles didn't return the proper amount"
    assert subg2.support_set() > subg.support_set(), \
        "adding a strategy didn't create a larger subgame"
    assert subg2 > subg, \
        "adding a strategy didn't create a larger subgame"


def add_existing_strategy_test():
    game = gamegen.empty_role_symmetric_game(2, [3, 4], [3, 2])
    subg = subgame.EmptySubgame(game, {'r0': ['s0', 's2'], 'r1': ['s1']})
    subg2 = subg.add_strategy('r0', 's0')
    assert subg == subg2, \
        "adding strategy in support didn't return same subgame"


def subgame_test():
    game = gamegen.role_symmetric_game(2, [3, 4], [3, 2])
    strategies = {'r0': ['s0', 's2'], 'r1': ['s1']}
    subg = subgame.subgame(game, strategies)
    esubg = subgame.EmptySubgame(game, strategies)
    assert subg.size == esubg.size, \
        "empty subgame and true subgame had different sizes"


def maximal_subgames_test():
    game = gamegen.role_symmetric_game(2, [3, 4], [3, 2])
    subs = list(subgame.maximal_subgames(game))
    assert len(subs) == 1, \
        "found more than maximal subgame in a complete game"
    assert subs[0] == subgame.EmptySubgame(game, game.strategies), \
        "found subgame wasn't the full one"


@testutils.apply(zip([0, 0.1, 0.4, 0.6]))
def executing_maximal_subgames_test(prob):
    game = gamegen.role_symmetric_game(2, [3, 4], [3, 2])
    game = gamegen.drop_profiles(game, prob)
    subs = list(subgame.maximal_subgames(game))
    assert subs is not None


@testutils.apply(testutils.game_sizes(), repeat=20)
def deviation_profile_count_test(roles, players, strategies):
    game = gamegen.empty_role_symmetric_game(roles, players, strategies)
    sub_strats = {r: random.sample(s, 1 + random.randrange(0, len(s)))
                  for r, s in game.strategies.items()}
    sub = subgame.EmptySubgame(game, sub_strats)

    num_devs = sum(1 for _ in sub.deviation_profiles())
    assert type(sub.num_deviation_profiles()) == int, \
        "num_deviation_profiles didn't return an int {}".format(sub_strats)
    assert num_devs == sub.num_deviation_profiles(), \
        "num_deviation_profiles didn't return correct number {}".format(
            sub_strats)

    full_players = {r: c ** 2 for r, c in game.players.items()}
    red = reduction.DeviationPreserving(full_players, game.players)
    num_dpr_devs = sum(sum(1 for _ in red.expand_profile(p[0]))
                       for p in sub.deviation_profiles())
    assert type(sub.num_dpr_deviation_profiles()) == int, \
        "num_dpr_deviation_profiles didn't return an int {}".format(
            sub_strats)
    assert num_dpr_devs == sub.num_dpr_deviation_profiles(), \
        "num_dpr_deviation_profiles didn't return correct number {}".format(
            sub_strats)
