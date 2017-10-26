"""Module for computing player reductions"""
import numpy as np

from gameanalysis import rsgame
from gameanalysis import paygame
from gameanalysis import subgame
from gameanalysis import utils


class hierarchical(object):
    def __init__(self):
        raise AttributeError('hierarchical is not constructable')

    @staticmethod
    def reduce_game(full_game, red_players):
        """Reduce a game using hierarchical reduction

        Parameters
        ----------
        full_game : Game
            The game to reduce.
        red_players : ndarray-like
            The reduced number of players for each role. This will be coerced
            into the proper shape if necessary.
        """
        red_game = rsgame.emptygame_names(
            full_game.role_names, red_players, full_game.strat_names)
        assert np.all(red_game.num_role_players > 0), \
            "all reduced players must be greater than zero"
        assert np.all(full_game.num_role_players >=
                      red_game.num_role_players), \
            "all full counts must not be less than reduced counts"

        if full_game.is_empty():
            return red_game
        elif full_game.num_profiles < red_game.num_all_profiles:
            profiles = full_game.profiles()
            payoffs = full_game.payoffs()
        else:
            profiles = hierarchical.expand_profiles(
                full_game, red_game.all_profiles())
            payoffs = full_game.get_payoffs(profiles)
            valid = ~np.all(np.isnan(payoffs) | (profiles == 0), 1)
            profiles = profiles[valid]
            payoffs = payoffs[valid]

        red_profiles, mask = hierarchical._reduce_profiles(
            full_game, red_game.num_role_players[None], profiles)
        return paygame.game_replace(red_game, red_profiles, payoffs[mask])

    def _reduce_profiles(sarr, reduced_players, profiles):
        """Hierarchically reduces several role symmetric array profiles

        This returns the reduced profiles, and a boolean mask showing which of
        the input profiles were actually reduced, as they might not all reduce.
        This allows having different numbers of players for reducded_players
        and profiles, but reduced players must be two dimensional."""
        full_players = np.add.reduceat(profiles, sarr.role_starts, 1)
        assert np.all(full_players >= reduced_players), \
            "full_players must be at least reduced_players"
        assert np.all((reduced_players > 0) |
                      ((full_players == 0) & (reduced_players == 0))), \
            "reduced players must be greater than 0"

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
        inds = np.lexsort(
            (alpha_ord, -alternates, -errors, ~diffs, role_order), 1)

        # Same as with expansion, map to new space
        rectified_inds = (inds + np.arange(num_reduceable)[:, None] *
                          sarr.num_strats)
        ind_mask = (np.arange(sarr.num_strats) < np.repeat(
            sarr.role_starts + overassigned, sarr.num_role_strats, 1))
        inc_inds = rectified_inds[ind_mask]
        red_profs.flat[inc_inds] = alternates.flat[inc_inds]

        # Our best guesses might not be accurate, so we have to filter out
        # profiles that don't order correctly
        expand_profs = hierarchical._expand_profiles(
            sarr, full_players, red_profs)
        valid = np.all(expand_profs == profiles, 1)
        reduced[reduced] = valid
        return red_profs[valid], reduced

    @staticmethod
    def reduce_profiles(red_game, profiles):
        """Reduce profiles hierarchically

        Parameters
        ----------
        red_game : Game
            Game that reduced profiles will be profiles for.
        profiles : ndarray-like
            The profiles to reduce.
        """
        profiles = np.asarray(profiles, int)
        assert profiles.shape[-1] == red_game.num_strats, \
            "profiles must be appropriate shape"
        return hierarchical._reduce_profiles(
            red_game, red_game.num_role_players[None],
            profiles.reshape((-1, red_game.num_strats)))[0]

    def _expand_profiles(sarr, full_players, profiles):
        """Hierarchically expands several role symmetric array profiles

        In the event that `full_players` isn't divisible by `reduced_players`,
        we first assign by rounding error and break ties in favor of
        more-played strategies. The final tie-breaker is index / alphabetical
        order."""
        reduced_players = np.add.reduceat(profiles, sarr.role_starts, 1)
        assert np.all(full_players >= reduced_players), \
            "full_players must be at least as large as reduced_players"
        assert np.all((reduced_players > 0) |
                      ((full_players == 0) & (reduced_players == 0))), \
            "reduced_players must be greater than zero"
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
        inds = np.lexsort((alpha_ord, -profiles, -error, role_order), 1)

        # Map them to indices in the expand_profs array, and mask out the first
        # that are necessary to meet unassigned
        rectified_inds = (inds + np.arange(num_profs)[:, None] *
                          sarr.num_strats)
        ind_mask = (
            np.arange(sarr.num_strats) <
            np.repeat(sarr.role_starts + unassigned, sarr.num_role_strats, 1))
        expand_profs.flat[rectified_inds[ind_mask]] += 1
        return expand_profs

    @staticmethod
    def expand_profiles(full_game, profiles):
        """Expand profiles hierarchically

        Parameters
        ----------
        full_game : Game
            Game that expanded profiles will be valid for.
        profiles : ndarray-like
            The profiles to expand
        """
        profiles = np.asarray(profiles, int)
        assert profiles.shape[-1] == full_game.num_strats, \
            "profiles must be appropriate shape"
        return hierarchical._expand_profiles(
            full_game, full_game.num_role_players[None],
            profiles.reshape((-1, full_game.num_strats)))

    @staticmethod
    def expand_deviation_profiles(
            full_game, subgame_mask, red_players, role_index=None):
        """Expand all deviation profiles from a subgame

        Parameters
        ----------
        full_game : Game
            The game the deviations profiles will be valid for.
        subgame_mask : ndarray-like, bool
            The subgame to get deviations from.
        red_players : ndarray-like
            The number of players in each role in the reduced game.
        role_index : int, optional
            If specified , only expand deviations for the role selected.
        """
        assert full_game.is_subgame(subgame_mask)
        return hierarchical.expand_profiles(
            full_game, subgame.deviation_profiles(
                rsgame.emptygame(red_players,
                                 full_game.num_role_strats),
                subgame_mask, role_index))


class deviation_preserving(object):
    def __init__(self):
        raise AttributeError('deviation_preserving is not constructable')

    @staticmethod
    def _devs(game, num_profs):
        """Return an array of the player counts after deviation"""
        return np.tile(np.repeat(
            game.num_role_players - np.eye(game.num_roles, dtype=int),
            game.num_role_strats, 0), (num_profs, 1))

    @staticmethod
    def reduce_game(full_game, red_players):
        """Reduce a game using deviation preserving reduction

        Parameters
        ----------
        full_game : Game
            The game to reduce.
        red_players : ndarray-like
            The reduced number of players for each role. This will be coerced
            into the proper shape if necessary.
        """
        red_game = rsgame.emptygame_names(
            full_game.role_names, red_players, full_game.strat_names)
        assert np.all((red_game.num_role_players > 1) |
                      (full_game.num_role_players == 1)), \
            "all reduced players must be greater than zero"
        assert np.all(full_game.num_role_players >=
                      red_game.num_role_players), \
            "all full counts must not be less than reduced counts"

        if full_game.is_empty():
            return red_game
        elif full_game.num_profiles < red_game.num_all_dpr_profiles:
            full_profiles = full_game.profiles()
            full_payoffs = full_game.payoffs()
        else:
            full_profiles = deviation_preserving.expand_profiles(
                full_game, red_game.all_profiles())
            full_payoffs = full_game.get_payoffs(full_profiles)
            valid = ~np.all(np.isnan(full_payoffs) |
                            (full_profiles == 0), 1)
            full_profiles = full_profiles[valid]
            full_payoffs = full_payoffs[valid]

        # Reduce
        red_profiles, red_inds, full_inds, strat_inds = \
            deviation_preserving.reduce_profiles(
                red_game, full_profiles, return_contributions=True)

        if red_profiles.size == 0:  # Empty reduction
            return red_game

        # Build mapping from payoffs to reduced profiles, and use bincount
        # to count the number of payoffs mapped to a specific location, and
        # sum the number of payoffs mapped to a specific location
        cum_inds = red_inds * full_game.num_strats + strat_inds
        payoff_vals = full_payoffs[full_inds, strat_inds]
        red_payoffs = np.bincount(
            cum_inds, payoff_vals, red_profiles.size).reshape(
                red_profiles.shape)
        red_payoff_counts = np.bincount(
            cum_inds, minlength=red_profiles.size).reshape(
                red_profiles.shape)
        mask = red_payoff_counts > 1
        red_payoffs[mask] /= red_payoff_counts[mask]

        unknown = (red_profiles > 0) & (red_payoff_counts == 0)
        red_payoffs[unknown] = np.nan
        valid = ~np.all((red_profiles == 0) | np.isnan(red_payoffs), 1)
        return paygame.game_replace(red_game, red_profiles[valid],
                                    red_payoffs[valid])

    @staticmethod
    def expand_profiles(full_game, profiles, *, return_contributions=False):
        """Expand profiles using dpr

        Parameters
        ----------
        full_game : Game
            Game that expanded profiles will be valid for.
        profiles : ndarray-like
            The profiles to expand
        return_contributions : bool, optional
            If specified, returns a boolean array matching the shape is
            returned indicating the payoffs that are needed for the initial
            profiles.
        """
        profiles = np.asarray(profiles, int)
        assert profiles.shape[-1] == full_game.num_strats, \
            "profiles not a valid shape"
        if not profiles.size:
            return np.empty((0, full_game.num_strats), int)
        profiles = profiles.reshape((-1, full_game.num_strats))
        all_red_players = np.add.reduceat(profiles, full_game.role_starts, 1)
        red_players = all_red_players[0]
        assert np.all(all_red_players == red_players)

        num_profs = profiles.shape[0]
        dev_profs = profiles[:, None] - np.eye(full_game.num_strats, dtype=int)
        dev_profs = np.reshape(dev_profs, (-1, full_game.num_strats))
        dev_full_players = deviation_preserving._devs(full_game, num_profs)

        mask = ~np.any(dev_profs < 0, 1)
        devs = (np.eye(full_game.num_strats, dtype=bool)[None]
                .repeat(num_profs, 0)
                .reshape((-1, full_game.num_strats))[mask])
        dev_full_profs = hierarchical._expand_profiles(
            full_game, dev_full_players[mask], dev_profs[mask]) + devs
        ids = utils.axis_to_elem(dev_full_profs)
        if not return_contributions:
            return dev_full_profs[np.unique(ids, return_index=True)[1]]
        else:
            # This is more complicated because we need to line up devs for the
            # same profile se we can "reduceat" to merge them
            order = np.argsort(ids)
            sids = ids[order]
            mask = np.insert(sids[1:] != sids[:-1], 0, True)
            profs = dev_full_profs[order[mask]]
            ored_devs = np.bitwise_or.reduceat(devs[order],
                                               mask.nonzero()[0], 0)
            return profs, ored_devs

    @staticmethod
    def reduce_profiles(red_game, profiles, *, return_contributions=False):
        """Reduce profiles using dpr

        Parameters
        ----------
        red_game : Game
            Game that reduced profiles will be profiles for.
        profiles : ndarray-like
            The profiles to reduce.
        return_contributions : bool, optional
            If true return ancillary information about where the payoffs come
            from.
        """
        profiles = np.asarray(profiles, int)
        assert profiles.shape[-1] == red_game.num_strats, \
            "profiles not a valid shape"
        if not profiles.size:
            return np.empty((0, red_game.num_strats), int)
        profiles = profiles.reshape((-1, red_game.num_strats))
        all_full_players = np.add.reduceat(profiles, red_game.role_starts, 1)
        full_players = all_full_players[0]
        assert np.all(all_full_players == full_players)

        num_profs = profiles.shape[0]
        dev_profs = profiles[:, None] - np.eye(red_game.num_strats, dtype=int)
        dev_profs = np.reshape(dev_profs, (-1, red_game.num_strats))
        dev_red_players = deviation_preserving._devs(red_game, num_profs)
        mask = ~np.any(dev_profs < 0, 1)
        red_profs, reduced = hierarchical._reduce_profiles(
            red_game, dev_red_players[mask], dev_profs[mask])
        devs = (np.eye(red_game.num_strats, dtype=int)[None]
                .repeat(num_profs, 0)
                .reshape((-1, red_game.num_strats))[mask][reduced])
        red_profs += devs
        red_profs, red_inds = utils.unique_axis(red_profs, return_inverse=True)
        if not return_contributions:
            return red_profs
        else:
            full_inds = np.arange(num_profs)[:, None].repeat(
                red_game.num_strats, 1).flat[mask][reduced]
            strat_inds = devs.nonzero()[1]
            return red_profs, red_inds, full_inds, strat_inds

    @staticmethod
    def expand_deviation_profiles(
            full_game, subgame_mask, red_players, role_index=None):
        """Expand all deviation profiles from a subgame

        Parameters
        ----------
        full_game : Game
            The game the deviations profiles will be valid for.
        subgame_mask : ndarray-like, bool
            The subgame to get deviations from.
        red_players : ndarray-like
            The number of players in each role in the reduced game.
        role_index : int, optional
            If specified , only expand deviations for the role selected.
        """
        subgame_mask = np.asarray(subgame_mask, bool)
        rdev = np.eye(full_game.num_roles, dtype=int)
        red_players = np.broadcast_to(np.asarray(red_players, int),
                                      full_game.num_roles)
        support = np.add.reduceat(subgame_mask, full_game.role_starts)

        def dev_profs(red_players, full_players, mask, rs):
            subg = rsgame.emptygame(red_players, support)
            sub_profs = subgame.translate(subg.all_profiles(), subgame_mask)
            game = rsgame.emptygame(full_players, full_game.num_role_strats)
            non_devs = hierarchical.expand_profiles(game, sub_profs)
            ndevs = np.sum(~mask)
            devs = np.zeros((ndevs, full_game.num_strats), int)
            devs[:, rs:rs + mask.size][:, ~mask] = np.eye(ndevs, dtype=int)
            profs = non_devs[:, None] + devs
            profs.shape = (-1, full_game.num_strats)
            return profs

        if role_index is None:
            expanded_profs = [dev_profs(red_players, full_players, mask, rs)
                              for red_players, full_players, mask, rs
                              in zip(red_players - rdev,
                                     full_game.num_role_players - rdev,
                                     np.split(subgame_mask,
                                              full_game.role_starts[1:]),
                                     full_game.role_starts)]
            return np.concatenate(expanded_profs)

        else:
            full_players = full_game.num_role_players.copy()
            full_players[role_index] -= 1
            red_players = red_players.copy()
            red_players[role_index] -= 1
            mask = np.split(subgame_mask, full_game.role_starts[1:])[
                role_index]
            rs = full_game.role_starts[role_index]
            return dev_profs(red_players, full_players, mask, rs)


class twins(object):
    def __init__(self):
        raise AttributeError('twins is not constructable')

    @staticmethod
    def reduce_game(full_game, red_players=None):
        """Reduce a game using twins reduction

        Parameters
        ----------
        full_game : Game
            The game to reduce.
        red_players : ndarray-like, optional
            The reduced number of players for each role. This must be None or
            the reduced number of players for the twins reductions.
        """
        exp_red_players = np.minimum(full_game.num_role_players, 2)
        assert (red_players is None or
                np.all(exp_red_players == red_players)), \
            "twins reduction didn't get expected reduced players"
        return deviation_preserving.reduce_game(full_game, exp_red_players)

    @staticmethod
    def expand_profiles(full_game, profiles):
        """Expand profiles using twins reduction

        Parameters
        ----------
        full_game : Game
            Game that expanded profiles will be valid for.
        profiles : ndarray-like
            The profiles to expand
        """
        red_players = np.minimum(full_game.num_role_players, 2)
        profiles = np.asarray(profiles, int)
        red_game = rsgame.emptygame(red_players, full_game.num_role_strats)
        assert red_game.is_profile(profiles).all()
        return deviation_preserving.expand_profiles(full_game, profiles)

    @staticmethod
    def reduce_profiles(red_game, profiles):
        """Reduce profiles using twins

        Parameters
        ----------
        red_game : Game
            Game that reduced profiles will be profiles for. This game must
            have the valid twins reduction number of players.
        profiles : ndarray-like
            The profiles to reduce.
        """
        profiles = np.asarray(profiles, int)
        assert np.all(red_game.num_role_players <= 2)
        return deviation_preserving.reduce_profiles(red_game, profiles)

    @staticmethod
    def expand_deviation_profiles(full_game, subgame_mask, red_players=None,
                                  role_index=None):
        """Expand all deviation profiles from a subgame

        Parameters
        ----------
        full_game : Game
            The game the deviations profiles will be valid for.
        subgame_mask : ndarray-like, bool
            The subgame to get deviations from.
        red_players : ndarray-like, optional
            The number of players in each role in the reduced game.IF
            specified, it must match the expected number for twins reduction.
        role_index : int, optional
            If specified , only expand deviations for the role selected.
        """
        exp_red_players = np.minimum(full_game.num_role_players, 2)
        assert (red_players is None or
                np.all(exp_red_players == red_players)), \
            "twins reduction didn't get expected reduced players"
        return deviation_preserving.expand_deviation_profiles(
            full_game, subgame_mask, exp_red_players, role_index)


class identity(object):
    def __init__(self):
        raise AttributeError('identity is not constructable')

    @staticmethod
    def reduce_game(full_game, red_players=None):
        """Return original game

        Parameters
        ----------
        full_game : Game
            The game to reduce.
        red_players : ndarray-like, optional
            If specified, this must match the number of players per role in
            full_game.
        """
        assert (red_players is None or
                np.all(full_game.num_role_players == red_players)), \
            "identity reduction must have same number of players"
        return paygame.game_copy(full_game)

    @staticmethod
    def expand_profiles(full_game, profiles):
        """Return input profiles

        Parameters
        ----------
        full_game : Game
            Game that all profiles must be valid for.
        profiles : ndarray-like
            The profiles.
        axis : int, optional
            The axis the profiles lie on.
        """
        profiles = np.asarray(profiles, int)
        assert full_game.is_profile(profiles).all()
        return profiles.reshape((-1, full_game.num_strats))

    @staticmethod
    def reduce_profiles(red_game, profiles):
        """Return original profiles

        Parameters
        ----------
        red_game : Game
            Game that all profiles must be valid for.
        profiles : ndarray-like
            The profiles.
        axis : int, optional
            The axis the profiles are on.
        """
        profiles = np.asarray(profiles, int)
        assert red_game.is_profile(profiles).all()
        return profiles.reshape((-1, red_game.num_strats))

    @staticmethod
    def expand_deviation_profiles(full_game, subgame_mask, red_players=None,
                                  role_index=None):
        """Expand all deviation profiles from a subgame

        Parameters
        ----------
        full_game : Game
            The game the deviations profiles will be valid for.
        subgame_mask : ndarray-like, bool
            The subgame to get deviations from.
        red_players : ndarray-like, optional
            The number of players in each role in the reduced game.IF
            specified, it must match the number for full_game.
        role_index : int, optional
            If specified , only expand deviations for the role selected.
        """
        assert (red_players is None or
                np.all(full_game.num_role_players == red_players)), \
            "identity reduction must have same number of players"
        return subgame.deviation_profiles(full_game, subgame_mask, role_index)
