import itertools
import os
import random

from gameanalysis import gamegen
from gameanalysis import profile
from gameanalysis import reduction
from test import testutils


@testutils.apply(itertools.product(
    [0, 0.6, 1],
    [
        (1, [1], [1], [1]),
        (1, [2], [1], [2]),
        (1, [3], [1], [2]),
        (1, [1], [2], [1]),
        (2, [1, 2], [2, 1], [1, 2]),
        (2, [1, 4], [2, 1], [1, 2]),
        (2, [4, 9], [3, 2], [2, 3]),
    ]))
# Simple test of exact DPR
def dpr_test(keep_prob, game_desc):
    roles, players, strategies, red_players = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(roles, players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    red_players = {r: p for r, p in zip(game.players, red_players)}
    red = reduction.DeviationPreserving(game.players, red_players)

    # Try to reduce game
    red_game = red.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    full_reduced_profiles = set(itertools.chain.from_iterable(
        red.reduce_profile(prof) for prof in game))
    reduced_profiles = set(red_game)
    assert full_reduced_profiles >= reduced_profiles, \
        "reduced game contained profiles it shouldn't have"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game
    contributing_profiles = {prof for prof in game
                             if any(reduced in reduced_profiles
                                    for reduced in red.reduce_profile(prof))}
    reduced_full_profiles = set(itertools.chain.from_iterable(
        red.expand_profile(prof) for prof in red_game))
    assert contributing_profiles <= reduced_full_profiles, \
        "full game did not have data for all profiles required of reduced"


@testutils.apply(itertools.product(
    [0, 0.6, 1],
    [
        (1, [1], [1]),
        (1, [2], [1]),
        (1, [3], [1]),
        (1, [1], [2]),
        (2, [1, 2], [2, 1]),
        (2, [1, 4], [2, 1]),
        (2, [4, 8], [3, 2]),
    ]))
# Simple test of Twins Reduction
def twins_test(keep_prob, game_desc):
    roles, players, strategies = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(roles, players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    red = reduction.Twins(game.players)

    # Try to reduce game
    red_game = red.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    full_reduced_profiles = set(itertools.chain.from_iterable(
        red.reduce_profile(prof) for prof in game))
    reduced_profiles = set(red_game)
    assert full_reduced_profiles >= reduced_profiles, \
        "reduced game contained profiles it shouldn't have"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game
    contributing_profiles = {prof for prof in game
                             if any(reduced in reduced_profiles
                                    for reduced in red.reduce_profile(prof))}
    reduced_full_profiles = set(itertools.chain.from_iterable(
        red.expand_profile(prof) for prof in red_game))
    assert contributing_profiles <= reduced_full_profiles, \
        "full game did not have data for all profiles required of reduced"


@testutils.apply(itertools.product(
    [0, 0.6, 1],
    [
        (1, [1], [1], [1]),
        (1, [2], [1], [2]),
        (1, [4], [1], [2]),
        (1, [1], [2], [1]),
        (2, [1, 2], [2, 1], [1, 2]),
        (2, [1, 4], [2, 1], [1, 2]),
        (2, [4, 9], [3, 2], [2, 3]),
    ]))
# Simple test of Hierarchical
def hierarchical_test(keep_prob, game_desc):
    roles, players, strategies, red_players = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(roles, players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    red_players = {r: p for r, p in zip(game.players, red_players)}
    red = reduction.Hierarchical(game.players, red_players)

    # Try to reduce game
    red.reduce_game(game)


@testutils.apply(itertools.product(
    [0, 0.6, 1],
    [
        (1, [1], [1]),
        (1, [2], [1]),
        (1, [3], [1]),
        (1, [1], [2]),
        (2, [1, 2], [2, 1]),
        (2, [1, 4], [2, 1]),
        (2, [4, 9], [3, 2]),
    ]))
# Simple test of identity reduction
def identity_test(keep_prob, game_desc):
    roles, players, strategies = game_desc
    # Create game and reduction
    game = gamegen.role_symmetric_game(roles, players, strategies)
    game = gamegen.drop_profiles(game, keep_prob)
    red = reduction.Identity()

    # Try to reduce game
    red_game = red.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    full_reduced_profiles = set(itertools.chain.from_iterable(
        red.reduce_profile(prof) for prof in game))
    assert full_reduced_profiles == set(red_game), \
        "reduced game didn't match full game"

    reduced_full_profiles = set(itertools.chain.from_iterable(
        red.expand_profile(prof) for prof in red_game))
    assert set(game) == reduced_full_profiles, \
        "full game did not match reduced game"


def sym_expand_tie_breaking_test():
    """Test that standard expansion breaks ties appropriately"""
    full = reduction._expand_sym_profile({'a': 4, 'b': 2}, 20, 6)
    assert full == {'a': 13, 'b': 7}, \
        "Didn't tie break on approximation first"
    full = reduction._expand_sym_profile({'a': 1, 'b': 3}, 10, 4)
    assert full == {'a': 2, 'b': 8}, \
        "Didn't tie break on larger count if approximations same"
    full = reduction._expand_sym_profile({'a': 2, 'b': 2}, 11, 4)
    assert full == {'a': 6, 'b': 5}, \
        "Didn't tie break on strategy name if all else same"


def sym_reduce_tie_breaking_test():
    """Test that the standard reduction breaks ties appropriately"""
    red = reduction._reduce_sym_profile({'a': 13, 'b': 7}, 20, 6)
    assert red == {'a': 4, 'b': 2}, \
        "Didn't tie break on approximation first"
    # Expanded profile has no valid reduction
    red = reduction._reduce_sym_profile({'a': 14, 'b': 6}, 20, 6)
    assert red is None, "Didn't properly find no reduced profile"
    red = reduction._reduce_sym_profile({'a': 2, 'b': 8}, 10, 4)
    assert red == {'a': 1, 'b': 3}, \
        "Didn't tie break on larger count if approximations same"
    red = reduction._reduce_sym_profile({'a': 6, 'b': 5}, 11, 4)
    assert red == {'a': 2, 'b': 2}, \
        "Didn't tie break on strategy name if all else same"


def approximate_dpr_expansion_test():
    """Test expansion on approximate dpr"""
    red = reduction.DeviationPreserving({'r': 8, 's': 11}, {'r': 3, 's': 4})
    red_prof = {'r': {'a': 1, 'b': 2}, 's': {'c': 2, 'd': 2}}
    profs = set(red.expand_profile(red_prof))
    expected = set(map(profile.Profile, [
        {'r': {'a': 4, 'b': 4}, 's': {'c': 6, 'd': 5}},
        {'r': {'a': 1, 'b': 7}, 's': {'c': 6, 'd': 5}},
        {'r': {'a': 3, 'b': 5}, 's': {'c': 4, 'd': 7}},
        {'r': {'a': 3, 'b': 5}, 's': {'c': 7, 'd': 4}},
    ]))
    assert profs == expected, "generated different profiles than expected"


def approximate_dpr_reduce_game_test():
    """Test approximate dpr game reduction"""
    game = gamegen.role_symmetric_game(2, [3, 4], 2)
    red = reduction.DeviationPreserving({'r0': 3, 'r1': 4}, {'r0': 2, 'r1': 2})
    redgame = red.reduce_game(game)
    # Pure strategies are reduced properly
    assert (redgame.get_payoff({'r0': {'s0': 2}, 'r1': {'s1': 2}}, 'r0', 's0')
            == game.get_payoff({'r0': {'s0': 3}, 'r1': {'s1': 4}}, 'r0', 's0'))
    # Mixed strategies are reduced properly
    red = {'r0': {'s0': 1, 's1': 1}, 'r1': {'s0': 1, 's1': 1}}
    full = {'r0': {'s0': 1, 's1': 2}, 'r1': {'s0': 2, 's1': 2}}
    assert (redgame.get_payoff(red, 'r0', 's0')
            == game.get_payoff(full, 'r0', 's0'))
    red = {'r0': {'s0': 1, 's1': 1}, 'r1': {'s0': 1, 's1': 1}}
    full = {'r0': {'s0': 2, 's1': 1}, 'r1': {'s0': 2, 's1': 2}}
    assert (redgame.get_payoff(red, 'r0', 's1')
            == game.get_payoff(full, 'r0', 's1'))
    red = {'r0': {'s0': 1, 's1': 1}, 'r1': {'s0': 1, 's1': 1}}
    full = {'r0': {'s0': 2, 's1': 1}, 'r1': {'s0': 1, 's1': 3}}
    assert (redgame.get_payoff(red, 'r1', 's0')
            == game.get_payoff(full, 'r1', 's0'))
    red = {'r0': {'s0': 1, 's1': 1}, 'r1': {'s0': 1, 's1': 1}}
    full = {'r0': {'s0': 2, 's1': 1}, 'r1': {'s0': 3, 's1': 1}}
    assert (redgame.get_payoff(red, 'r1', 's1')
            == game.get_payoff(full, 'r1', 's1'))


@testutils.apply([
    (1, [1], [1]),
    (1, [2], [1]),
    (1, [3], [1]),
    (1, [1], [2]),
    (2, [1, 2], [2, 1]),
    (2, [1, 4], [2, 1]),
    (2, [4, 9], [3, 2]),
] + ([
    (3, 3, 3),
    (3, [3, 4, 9], [4, 3, 2]),
] if os.getenv('BIG_TESTS') == 'ON' else []), repeat=20)
def random_approximate_dpr_test(roles, players, strategies):
    """Test approximate dpr preserves completeness on random games"""
    game = gamegen.role_symmetric_game(roles, players, strategies)
    red_counts = {role: random.randint(2, c) if c > 1 else 1
                  for role, c in game.players.items()}
    red = reduction.DeviationPreserving(game.players, red_counts)

    # Try to reduce game
    red_game = red.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    assert red_game.is_complete(), "DPR did not preserve completeness"
