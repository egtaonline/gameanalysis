"""Module for performing actions on restrictions

A restriction is a subset of strategies that are considered viable. They are
represented as a bit-mask over strategies with at least one true value per
role."""
import numpy as np

from gameanalysis import rsgame
from gameanalysis import utils


# TODO Do these really need to be in a separate file, they should probably just
# be in rsgame

def num_deviation_profiles(game, rest):
    """Returns the number of deviation profiles

    This is a closed form way to compute `deviation_profiles(game,
    rest).shape[0]`.
    """
    rest = np.asarray(rest, bool)
    utils.check(game.is_restriction(rest), 'restriction must be valid')
    num_role_strats = np.add.reduceat(rest, game.role_starts)
    num_devs = game.num_role_strats - num_role_strats
    dev_players = game.num_role_players - np.eye(game.num_roles, dtype=int)
    return np.sum(utils.game_size(dev_players, num_role_strats).prod(1) *
                  num_devs)


def num_deviation_payoffs(game, rest):
    """Returns the number of deviation payoffs

    This is a closed form way to compute `np.sum(deviation_profiles(game, rest)
    > 0)`."""
    rest = np.asarray(rest, bool)
    utils.check(game.is_restriction(rest), 'restriction must be valid')
    num_role_strats = np.add.reduceat(rest, game.role_starts)
    num_devs = game.num_role_strats - num_role_strats
    dev_players = (game.num_role_players - np.eye(game.num_roles, dtype=int) -
                   np.eye(game.num_roles, dtype=int)[:, None])
    temp = utils.game_size(dev_players, num_role_strats).prod(2)
    non_deviators = np.sum(np.sum(temp * num_role_strats, 1) * num_devs)
    return non_deviators + num_deviation_profiles(game, rest)


def num_dpr_deviation_profiles(game, rest):
    """Returns the number of dpr deviation profiles"""
    rest = np.asarray(rest, bool)
    utils.check(game.is_restriction(rest), 'restriction must be valid')
    num_role_strats = np.add.reduceat(rest, game.role_starts)
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
    return num_deviation_payoffs(game, rest) - overcount


def deviation_profiles(game, rest, role_index=None):
    """Return strict deviation profiles

    Strict means that all returned profiles will have exactly one player where
    rest is false, i.e.

    `np.all(np.sum(profiles * ~rest, 1) == 1)`

    If `role_index` is specified, only profiles for that role will be
    returned."""
    rest = np.asarray(rest, bool)
    utils.check(game.is_restriction(rest), 'restriction must be valid')
    support = np.add.reduceat(rest, game.role_starts)

    def dev_profs(players, mask, rst):
        """Get deviation profiles"""
        rgame = rsgame.empty(players, support)
        non_devs = translate(rgame.all_profiles(), rest)
        ndevs = np.sum(~mask)
        devs = np.zeros((ndevs, game.num_strats), int)
        devs[:, rst:rst + mask.size][:, ~mask] = np.eye(ndevs, dtype=int)
        profs = non_devs[:, None] + devs
        profs.shape = (-1, game.num_strats)
        return profs

    if role_index is None: # pylint: disable=no-else-return
        dev_players = game.num_role_players - np.eye(game.num_roles, dtype=int)
        profs = [dev_profs(players, mask, rs) for players, mask, rs
                 in zip(dev_players, np.split(rest, game.role_starts[1:]),
                        game.role_starts)]
        return np.concatenate(profs)

    else:
        players = game.num_role_players.copy()
        players[role_index] -= 1
        mask = np.split(rest, game.role_starts[1:])[role_index]
        rstart = game.role_starts[role_index]
        return dev_profs(players, mask, rstart)


def additional_strategy_profiles(game, rest, role_strat_ind):
    """Returns all profiles added by strategy at index"""
    # This uses the observation that the added profiles are all of the profiles
    # of the new restricted game with one less player in role, and then where
    # that last player always plays strat
    rest = np.asarray(rest, bool)
    utils.check(game.is_restriction(rest), 'restriction must be valid')
    new_players = game.num_role_players.copy()
    new_players[game.role_indices[role_strat_ind]] -= 1
    base = rsgame.empty(new_players, game.num_role_strats)
    new_mask = rest.copy()
    new_mask[role_strat_ind] = True
    profs = base.restrict(new_mask).all_profiles()
    expand_profs = np.zeros((profs.shape[0], game.num_strats), int)
    expand_profs[:, new_mask] = profs
    expand_profs[:, role_strat_ind] += 1
    return expand_profs


def translate(profiles, rest):
    """Translate a strategy object to the full game"""
    utils.check(
        profiles.shape[-1] == rest.sum(),
        'profiles must be valid for the restriction')
    if rest.all():
        return profiles
    new_profs = np.zeros(
        profiles.shape[:-1] + (rest.size,), profiles.dtype)
    new_profs[..., rest] = profiles
    return new_profs


def to_id(game, rest):
    """Return a unique integer representing a restriction"""
    bits = np.ones(game.num_strats, int)
    bits[0] = 0
    bits[game.role_starts[1:]] -= game.num_role_strats[:-1]
    bits = 2 ** bits.cumsum()
    roles = np.insert(np.cumprod(2 ** game.num_role_strats[:-1] - 1), 0, 1)
    return np.sum(roles * (np.add.reduceat(
        rest * bits, game.role_starts, -1) - 1), -1)


def from_id(game, rest_id):
    """Return a restriction mask from its unique id"""
    rest_id = np.asarray(rest_id)
    bits = np.ones(game.num_strats, int)
    bits[0] = 0
    bits[game.role_starts[1:]] -= game.num_role_strats[:-1]
    bits = 2 ** bits.cumsum()
    roles = 2 ** game.num_role_strats - 1
    rolesc = np.insert(np.cumprod(roles[:-1]), 0, 1)
    return (np.repeat(rest_id[..., None] // rolesc % roles + 1,
                      game.num_role_strats, -1) // bits % 2).astype(bool)


def maximal_restrictions(game):
    """Returns all maximally complete restrictions

    This function returns an array of restrictions, such that no restriction is
    a sub_restriction (i.e. `np.all(sub <= rest)`) and that no restriction
    could be increased, and still contain complete payoff data for the game.
    This is reducible to clique finding, and as such is NP Hard"""
    # invariant that we have data for every restriction in queue
    # The reverse order is necessary for the order we explore
    pure_profs = game.pure_profiles()[::-1]
    queue = [p > 0 for p in pure_profs if p in game]
    maximals = []
    while queue:
        rest = queue.pop()
        maximal = True
        devs = rest.astype(int)
        devs[game.role_starts[1:]] -= np.add.reduceat(
            rest, game.role_starts)[:-1]
        devs = np.nonzero((devs.cumsum() > 0) & ~rest)[0][::-1]
        for dev_ind in devs:
            profs = additional_strategy_profiles(game, rest, dev_ind)
            # TODO Some anecdotal evidence suggests that when checking multiple
            # profiles, np.isin(profs, game.profiles()) is faster for checking
            # multiple profiles. We can potentially avoid using a dictionary
            # and instead use numpy set operations
            # TODO More to the point, the idea of checking if a profile is in
            # the game, may be slow for certain game types, but fast for
            # others. Additionally, the idea of `in` only working for complete
            # profiles is a little counter intuitive given that now all games
            # will return payoff data that may have nans. However, we also
            # don't want to just expose is a restricted game is complete (or
            # maybe complete payoff data) because that will require a lot of
            # extra checks that this doesn't make. In some sense we want each
            # game to implement something aking to "all additional strategy
            # profiles in" or maybe, "deviation in".
            if all(p in game for p in profs):
                maximal = False
                rest_copy = rest.copy()
                rest_copy[dev_ind] = True
                queue.append(rest_copy)

        # This checks that no duplicates are emitted. This algorithm will
        # always find the largest subset first, but subsequent 'maximal'
        # subsets may actually be subsets of previous maximal subsets.
        if maximal and not any(np.all(rest <= s) for s in maximals):
            maximals.append(rest)

    return np.array(maximals)
