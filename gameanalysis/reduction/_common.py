"""Common routines for reductions"""
import numpy as np

from gameanalysis import utils


def reduce_profiles(sarr, reduced_players, profiles): # pylint: disable=too-many-locals
    """Hierarchically reduces several role symmetric array profiles

    This returns the reduced profiles, and a boolean mask showing which of
    the input profiles were actually reduced, as they might not all reduce.
    This allows having different numbers of players for reducded_players
    and profiles, but reduced players must be two dimensional."""
    full_players = np.add.reduceat(profiles, sarr.role_starts, 1)
    utils.check(
        np.all(full_players >= reduced_players),
        'full_players must be at least reduced_players')
    utils.check(
        np.all((reduced_players > 0) | ((full_players == 0) &
                                        (reduced_players == 0))),
        'reduced players must be greater than 0')

    rep_red_players = reduced_players.repeat(sarr.num_role_strats, 1)
    rep_full_players = np.maximum(
        full_players, 1).repeat(sarr.num_role_strats, 1)
    red_profs = np.ceil(profiles * rep_red_players /
                        rep_full_players).astype(int)
    alternates = np.ceil((profiles - 1) * rep_red_players /
                         rep_full_players).astype(int)

    # What if every strategy was tie broken
    overassigned = np.add.reduceat(
        red_profs, sarr.role_starts, 1) - reduced_players
    # The strategies that could have been tie broken i.e. the profile
    # changed
    diffs = alternates != red_profs
    # These are "possible" to reduce
    reduced = np.all(overassigned <= np.add.reduceat(
        diffs, sarr.role_starts, 1), 1)

    # Move everything into reduced space
    num_reduceable = reduced.sum()
    profiles = profiles[reduced]
    red_profs = red_profs[reduced]
    alternates = alternates[reduced]
    overassigned = overassigned[reduced]
    diffs = diffs[reduced]
    full_players = full_players[reduced]
    if reduced_players.shape[0] > 1:
        reduced_players = reduced_players[reduced]

    # Now we take the hypothetical reduced profiles, and see which ones
    # would have won the tie breaker
    role_order = np.broadcast_to(sarr.role_indices,
                                 (num_reduceable, sarr.num_strats))
    alpha_inds = np.arange(sarr.num_strats)
    alpha_ord = np.broadcast_to(alpha_inds,
                                (num_reduceable, sarr.num_strats))
    rep_red_players = np.maximum(
        reduced_players, 1).repeat(sarr.num_role_strats, -1)
    rep_full_players = full_players.repeat(sarr.num_role_strats, -1)
    errors = alternates * rep_full_players / rep_red_players - profiles + 1
    inds = np.asarray(np.argsort(np.rec.fromarrays(
        [role_order, ~diffs, -errors, -alternates, alpha_ord]), 1))

    # Same as with expansion, map to new space
    rectified_inds = (inds + np.arange(num_reduceable)[:, None] *
                      sarr.num_strats)
    ind_mask = (np.arange(sarr.num_strats) < np.repeat(
        sarr.role_starts + overassigned, sarr.num_role_strats, 1))
    inc_inds = rectified_inds[ind_mask]
    red_profs.flat[inc_inds] = alternates.flat[inc_inds]

    # Our best guesses might not be accurate, so we have to filter out
    # profiles that don't order correctly
    expand_profs = expand_profiles(
        sarr, full_players, red_profs)
    valid = np.all(expand_profs == profiles, 1)
    reduced[reduced] = valid
    return red_profs[valid], reduced


def expand_profiles(sarr, full_players, profiles): # pylint: disable=too-many-locals
    """Hierarchically expands several role symmetric array profiles

    In the event that `full_players` isn't divisible by `reduced_players`,
    we first assign by rounding error and break ties in favor of
    more-played strategies. The final tie-breaker is index / alphabetical
    order."""
    reduced_players = np.add.reduceat(profiles, sarr.role_starts, 1)
    utils.check(
        np.all(full_players >= reduced_players),
        'full_players must be at least as large as reduced_players')
    utils.check(
        np.all((reduced_players > 0) | ((full_players == 0) &
                                        (reduced_players == 0))),
        'reduced_players must be greater than zero')
    # Maximum prevents divide by zero error; equivalent to + eps
    rep_red_players = np.maximum(
        reduced_players, 1).repeat(sarr.num_role_strats, -1)
    rep_full_players = full_players.repeat(sarr.num_role_strats, -1)
    num_profs = profiles.shape[0]
    expand_profs = profiles * rep_full_players // rep_red_players
    unassigned = full_players - \
        np.add.reduceat(expand_profs, sarr.role_starts, 1)

    # Order all possible strategies to find which to increment
    role_order = np.broadcast_to(sarr.role_indices,
                                 (num_profs, sarr.num_strats))
    error = profiles * rep_full_players / rep_red_players - expand_profs
    alpha_inds = np.arange(sarr.num_strats)
    alpha_ord = np.broadcast_to(alpha_inds, (num_profs, sarr.num_strats))
    inds = np.asarray(np.argsort(np.rec.fromarrays(
        [role_order, -error, -profiles, alpha_ord]), 1))

    # Map them to indices in the expand_profs array, and mask out the first
    # that are necessary to meet unassigned
    rectified_inds = (inds + np.arange(num_profs)[:, None] *
                      sarr.num_strats)
    ind_mask = (
        np.arange(sarr.num_strats) <
        np.repeat(sarr.role_starts + unassigned, sarr.num_role_strats, 1))
    expand_profs.flat[rectified_inds[ind_mask]] += 1
    return expand_profs
