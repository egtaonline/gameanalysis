"""Module for performing actions on subgames

A subgame is a game with a restricted set of strategies that usually make
analysis tractable. Most representations just use a subgame mask, which is a
bitmask over included strategies."""
import numpy as np

from gameanalysis import rsgame
from gameanalysis import utils


# TODO Recently this has been called a restricted game to remove confusion with
# the extensive form game concept. This change in nomenclature should probably
# happen at some point.


def num_deviation_profiles(game, subgame_mask):
    """Returns the number of deviation profiles

    This is a closed form way to compute `deviation_profiles(game,
    subgame_mask).shape[0]`.
    """
    subgame_mask = np.asarray(subgame_mask, bool)
    assert game.num_strats == subgame_mask.size
    num_role_strats = np.add.reduceat(subgame_mask, game.role_starts)
    num_devs = game.num_role_strats - num_role_strats
    dev_players = game.num_role_players - np.eye(game.num_roles, dtype=int)
    return np.sum(utils.game_size(dev_players, num_role_strats).prod(1) *
                  num_devs)


def num_deviation_payoffs(game, subgame_mask):
    """Returns the number of deviation payoffs

    This is a closed form way to compute `np.sum(deviation_profiles(game,
    subgame_mask) > 0)`."""
    subgame_mask = np.asarray(subgame_mask, bool)
    assert game.num_strats == subgame_mask.size
    num_role_strats = np.add.reduceat(subgame_mask, game.role_starts)
    num_devs = game.num_role_strats - num_role_strats
    dev_players = (game.num_role_players - np.eye(game.num_roles, dtype=int) -
                   np.eye(game.num_roles, dtype=int)[:, None])
    temp = utils.game_size(dev_players, num_role_strats).prod(2)
    non_deviators = np.sum(np.sum(temp * num_role_strats, 1) * num_devs)
    return non_deviators + num_deviation_profiles(game, subgame_mask)


def num_dpr_deviation_profiles(game, subgame_mask):
    """Returns the number of dpr deviation profiles"""
    subgame_mask = np.asarray(subgame_mask, bool)
    assert game.num_strats == subgame_mask.size
    num_role_strats = np.add.reduceat(subgame_mask, game.role_starts)
    num_devs = game.num_role_strats - num_role_strats

    pure = (np.arange(3, 1 << game.num_roles)[:, None] &
            (1 << np.arange(game.num_roles))).astype(bool)
    cards = pure.sum(1)
    pure = pure[cards > 1]
    card_counts = cards[cards > 1, None] - 1 - \
        ((game.num_role_players > 1) & pure)
    # For each combination of pure roles, compute the number of profiles
    # conditioned on those roles being pure, then multiply them by the
    # cardinality of the pure roles.
    sp_dev = np.eye(game.num_roles, dtype=bool) & (game.num_role_players == 1)
    pure_counts = num_role_strats * ~sp_dev + sp_dev
    dev_players = game.num_role_players - np.eye(game.num_roles, dtype=int)
    unpure_counts = utils.game_size(dev_players, num_role_strats) - pure_counts
    pure_counts = np.prod(pure_counts * pure[:, None] + ~pure[:, None], 2)
    unpure_counts = np.prod(unpure_counts * ~pure[:, None] + pure[:, None], 2)
    overcount = np.sum(card_counts * pure_counts * unpure_counts * num_devs)
    return num_deviation_payoffs(game, subgame_mask) - overcount


def deviation_profiles(game, subgame_mask, role_index=None):
    """Return strict deviation profiles

    Strict means that all returned profiles will have exactly one player where
    subgame_mask is false, i.e.

    `np.all(np.sum(profiles * ~subgame_mask, 1) == 1)`

    If `role_index` is specified, only profiles for that role will be
    returned."""
    subgame_mask = np.asarray(subgame_mask, bool)
    assert (game.num_strats,) == subgame_mask.shape
    support = np.add.reduceat(subgame_mask, game.role_starts)

    def dev_profs(players, mask, rs):
        subg = rsgame.emptygame(players, support)
        non_devs = translate(subg.all_profiles(), subgame_mask)
        ndevs = np.sum(~mask)
        devs = np.zeros((ndevs, game.num_strats), int)
        devs[:, rs:rs + mask.size][:, ~mask] = np.eye(ndevs, dtype=int)
        profs = non_devs[:, None] + devs
        profs.shape = (-1, game.num_strats)
        return profs

    if role_index is None:
        profs = [dev_profs(players, mask, rs) for players, mask, rs
                 in zip(game.num_role_players - np.eye(game.num_roles,
                                                       dtype=int),
                        np.split(subgame_mask, game.role_starts[1:]),
                        game.role_starts)]
        return np.concatenate(profs)

    else:
        players = game.num_role_players.copy()
        players[role_index] -= 1
        mask = np.split(subgame_mask, game.role_starts[1:])[role_index]
        rs = game.role_starts[role_index]
        return dev_profs(players, mask, rs)


def additional_strategy_profiles(game, subgame_mask, role_strat_ind):
    """Returns all profiles added by strategy at index"""
    # This uses the observation that the added profiles are all of the
    # profiles of the new subgame with one less player in role, and then
    # where that last player always plays strat
    subgame_mask = np.asarray(subgame_mask, bool)
    assert game.num_strats == subgame_mask.size
    new_players = game.num_role_players.copy()
    new_players[game.role_indices[role_strat_ind]] -= 1
    base = rsgame.emptygame(new_players, game.num_role_strats)
    new_mask = subgame_mask.copy()
    new_mask[role_strat_ind] = True
    profs = base.subgame(new_mask).all_profiles()
    expand_profs = np.zeros((profs.shape[0], game.num_strats), int)
    expand_profs[:, new_mask] = profs
    expand_profs[:, role_strat_ind] += 1
    return expand_profs


def translate(profiles, subgame_mask):
    """Translate a mixture or profile from a subgame to the full game"""
    assert profiles.shape[-1] == subgame_mask.sum()
    if subgame_mask.all():
        return profiles
    else:
        new_profs = np.zeros(profiles.shape[:-1] + (subgame_mask.size,),
                             profiles.dtype)
        new_profs[..., subgame_mask] = profiles
        return new_profs


def to_id(game, subgame_mask):
    """Return a unique integer representing a subgame"""
    bits = np.ones(game.num_strats, int)
    bits[0] = 0
    bits[game.role_starts[1:]] -= game.num_role_strats[:-1]
    bits = 2 ** bits.cumsum()
    roles = np.insert(np.cumprod(2 ** game.num_role_strats[:-1] - 1), 0, 1)
    return np.sum(roles * (np.add.reduceat(subgame_mask * bits,
                                           game.role_starts, -1) - 1), -1)


def from_id(game, subgame_id):
    """Return a subgame mask from its unique indicator"""
    subgame_id = np.asarray(subgame_id)
    bits = np.ones(game.num_strats, int)
    bits[0] = 0
    bits[game.role_starts[1:]] -= game.num_role_strats[:-1]
    bits = 2 ** bits.cumsum()
    roles = 2 ** game.num_role_strats - 1
    rolesc = np.insert(np.cumprod(roles[:-1]), 0, 1)
    return (np.repeat(subgame_id[..., None] // rolesc % roles + 1,
                      game.num_role_strats, -1) // bits % 2).astype(bool)


def maximal_subgames(game):
    """Returns all maximally complete subgame masks"""
    # invariant that we have data for every subgame in queue
    pure_profs = game.pure_profiles()[::-1]
    queue = [p > 0 for p in pure_profs if p in game]
    maximals = []
    while queue:
        sub = queue.pop()
        maximal = True
        devs = sub.astype(int)
        devs[game.role_starts[1:]
             ] -= np.add.reduceat(sub, game.role_starts)[:-1]
        devs = np.nonzero((devs.cumsum() > 0) & ~sub)[0][::-1]
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
        if maximal and not any(np.all(sub <= s) for s in maximals):
            maximals.append(sub)

    return np.array(maximals)
