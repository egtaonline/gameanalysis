from gameanalysis import gamegen
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
