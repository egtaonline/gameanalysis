"""Module for performing actions on subgames

A subgame is a game with a restricted set of strategies that usually make
analysis tractable. Most representations just use a subgame mask, which is a
bitmask over included strategies."""
import numpy as np

from gameanalysis import gameio
from gameanalysis import rsgame
from gameanalysis import utils


def pure_subgame_masks(game):
    """Returns every pure subgame mask in a game

    A pure subgame is a subgame where each role only has one strategy. This
    returns the pure subgames in sorted order based off of role and
    strategy."""
    return game.pure_profiles() > 0


def num_deviation_profiles(game, subgame_mask):
    """Returns the number of deviation profiles

    This is a closed form way to compute `deviation_profiles(game,
    subgame_mask).shape[0]`.
    """
    subgame_mask = np.asarray(subgame_mask, bool)
    num_strategies = game.role_reduce(subgame_mask)
    num_devs = game.num_strategies - num_strategies
    dev_players = game.num_players - np.eye(game.num_roles, dtype=int)
    return np.sum(utils.game_size(dev_players, num_strategies).prod(1) *
                  num_devs)


def num_deviation_payoffs(game, subgame_mask):
    """Returns the number of deviation payoffs

    This is a closed form way to compute `np.sum(deviation_profiles(game,
    subgame_mask) > 0)`."""
    subgame_mask = np.asarray(subgame_mask, bool)
    num_strategies = game.role_reduce(subgame_mask)
    num_devs = game.num_strategies - num_strategies
    dev_players = (game.num_players - np.eye(game.num_roles, dtype=int) -
                   np.eye(game.num_roles, dtype=int)[:, None])
    temp = utils.game_size(dev_players, num_strategies).prod(2)
    non_deviators = np.sum(np.sum(temp * num_strategies, 1) * num_devs)
    return non_deviators + num_deviation_profiles(game, subgame_mask)


def num_dpr_deviation_profiles(game, subgame_mask):
    """Returns the number of dpr deviation profiles"""
    subgame_mask = np.asarray(subgame_mask, bool)
    num_strategies = game.role_reduce(subgame_mask)
    num_devs = game.num_strategies - num_strategies

    pure = (np.arange(3, 1 << game.num_roles)[:, None] &
            (1 << np.arange(game.num_roles))).astype(bool)
    cards = pure.sum(1)
    pure = pure[cards > 1]
    card_counts = cards[cards > 1, None] - 1 - ((game.num_players > 1) & pure)
    # For each combination of pure roles, compute the number of profiles
    # conditioned on those roles being pure, then multiply them by the
    # cardinality of the pure roles.
    sp_dev = np.eye(game.num_roles, dtype=bool) & (game.num_players == 1)
    pure_counts = num_strategies * ~sp_dev + sp_dev
    dev_players = game.num_players - np.eye(game.num_roles, dtype=int)
    unpure_counts = utils.game_size(dev_players, num_strategies) - pure_counts
    pure_counts = np.prod(pure_counts * pure[:, None] + ~pure[:, None], 2)
    unpure_counts = np.prod(unpure_counts * ~pure[:, None] + pure[:, None], 2)
    overcount = np.sum(card_counts * pure_counts * unpure_counts * num_devs)
    return num_deviation_payoffs(game, subgame_mask) - overcount


def deviation_profiles(game, subgame_mask):
    """Return every deviation profile"""
    subgame_mask = np.asarray(subgame_mask, bool)
    num_supp = game.role_reduce(subgame_mask)
    num_dev = game.num_strategies - num_supp
    dev_offsets = np.insert(np.cumsum(num_supp * num_dev), 0, 0)
    devs = np.zeros((dev_offsets[-1], game.num_role_strats), int)

    for mask, ns, nd, so, do in zip(game.role_split(subgame_mask), num_supp,
                                    num_dev, game.role_starts, dev_offsets):
        num_strat = mask.size
        view = devs[do:do + ns * nd,
                    so:so + num_strat]
        view.shape = (ns, nd, num_strat)
        view[..., ~mask] += np.eye(nd, dtype=int)
        view[..., mask] -= np.eye(ns, dtype=int)[:, None]

    sub = subgame(rsgame.BaseGame(game), subgame_mask)
    full_profs = np.zeros((sub.num_all_profiles, game.num_role_strats),
                          int)
    full_profs[:, subgame_mask] = sub.all_profiles()
    dev_profs = np.reshape(full_profs[:, None] + devs,
                           (-1, game.num_role_strats))
    return utils.unique_axis(
        dev_profs[np.all(dev_profs[:, subgame_mask] >= 0, 1)])


def additional_strategy_profiles(game, subgame_mask, role_strat_ind):
    """Returns all profiles added by strategy at index"""
    # This uses the observation that the added profiles are all of the
    # profiles of the new subgame with one less player in role, and then
    # where that last player always plays strat
    subgame_mask = np.asarray(subgame_mask, bool)
    new_players = game.num_players.copy()
    new_players[game.role_index[role_strat_ind]] -= 1
    base = rsgame.BaseGame(new_players, game.num_strategies)
    new_mask = subgame_mask.copy()
    new_mask[role_strat_ind] = True
    profs = subgame(base, new_mask).all_profiles()
    expand_profs = np.zeros((profs.shape[0], game.num_role_strats), int)
    expand_profs[:, new_mask] = profs
    expand_profs[:, role_strat_ind] += 1
    return expand_profs


def subgame(game, subgame_mask):
    """Returns a new game that only has data for profiles in subgame_mask"""
    subgame_mask = np.asarray(subgame_mask, bool)
    num_strats = game.role_reduce(subgame_mask)
    assert np.all(num_strats > 0), \
        "Not all roles have at least one strategy"

    # There's some duplication here in order to allow base games
    if isinstance(game, rsgame.SampleGame):
        prof_mask = ~np.any(game.profiles * ~subgame_mask, 1)
        profiles = game.profiles[prof_mask][:, subgame_mask]
        sample_payoffs = [pays[pmask][:, subgame_mask]
                          for pays, pmask
                          in zip(game.sample_payoffs,
                                 np.split(prof_mask, game.sample_starts[1:]))
                          if pmask.any()]
        return rsgame.SampleGame(game.num_players, num_strats, profiles,
                                 sample_payoffs)

    elif isinstance(game, rsgame.Game):
        prof_mask = ~np.any(game.profiles * ~subgame_mask, 1)
        profiles = game.profiles[prof_mask][:, subgame_mask]
        payoffs = game.payoffs[prof_mask][:, subgame_mask]
        return rsgame.Game(game.num_players, num_strats, profiles, payoffs)

    else:
        return rsgame.BaseGame(game.num_players, num_strats)


def subserializer(serial, subgame_mask):
    """Return a serializer for a subgame"""
    new_strats = [[s for s, m in zip(strats, mask) if m]
                  for strats, mask
                  in zip(serial.strat_names, serial.role_split(subgame_mask))]
    return gameio.GameSerializer(serial.role_names, new_strats)


def translate(profiles, subgame_mask):
    """Translate a mixture or profile from a subgame to the full game"""
    assert profiles.shape[-1] == subgame_mask.sum()
    new_profs = np.zeros(profiles.shape[:-1] + (subgame_mask.size,),
                         profiles.dtype)
    new_profs[..., subgame_mask] = profiles
    return new_profs


def maximal_subgames(game):
    """Returns all maximally complete subgame masks"""
    # invariant that we have data for every subgame in queue
    pure_profs = game.pure_profiles()[::-1]
    queue = [p > 0 for p in pure_profs if p in game]
    maximals = []
    while queue:
        sub = queue.pop()

        if any(np.all(s >= sub) for s in maximals):
            continue

        maximal = True
        devs = sub.astype(int)
        devs[game.role_starts[1:]] -= game.role_reduce(sub)[:-1]
        devs = np.nonzero((devs.cumsum() > 0) & ~sub)[0]
        for dev_ind in devs:
            profs = additional_strategy_profiles(game, sub, dev_ind)
            if all(p in game for p in profs):
                maximal = False
                sub_copy = sub.copy()
                sub_copy[dev_ind] = True
                queue.append(sub_copy)

        # This checks that no duplicates are emitted.  This algorithm will
        # always find the largest subset first, but subsequent 'maximal'
        # subsets may actually be subsets of previous maximal subsets.
        if maximal:
            maximals.append(sub)

    return np.asarray(maximals)
