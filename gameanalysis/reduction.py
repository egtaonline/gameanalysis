"""Module for computing player reductions"""
import abc

import numpy as np
import numpy.random as rand

from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import utils


# TODO Make reductions handle partial profiles


def _expand_rsym_profiles(game, profiles, full_players, reduced_players):
    """Hierarchically expands several role symmetric array profiles

    In the event that `full_players` isn't divisible by `reduced_players`, we
    first assign by rounding error and break ties in favor of more-played
    strategies. The final tie-breaker is index / alphabetical order."""
    assert np.all(game.role_reduce(profiles) == reduced_players), \
        "Not all profiles were valid {} {}".format(profiles, reduced_players)
    assert (np.all(full_players >= reduced_players) and
            np.all((reduced_players > 0) |
                   ((full_players == 0) & (reduced_players == 0)))), \
        "not all reductions were valid\nfull: {}\nreduced: {}".format(
            full_players, reduced_players)
    # Maximum prevents divide by zero error; equivalent to + eps
    rep_red_players = game.role_repeat(np.maximum(reduced_players, 1))
    rep_full_players = game.role_repeat(full_players)
    num_profs = profiles.shape[0]
    expand_profs = profiles * rep_full_players // rep_red_players
    unassigned = full_players - game.role_reduce(expand_profs)

    # Order all possible strategies to find which to increment
    role_order = np.broadcast_to(game.role_repeat(np.arange(game.num_roles)),
                                 (num_profs, game.num_role_strats))
    error = profiles * rep_full_players / rep_red_players - expand_profs
    alpha_inds = np.arange(game.num_role_strats)
    alpha_ord = np.broadcast_to(alpha_inds, (num_profs, game.num_role_strats))
    inds = np.lexsort((alpha_ord, -profiles, -error, role_order), 1)

    # Map them to indices in the expand_profs array, and mask out the first
    # that are necessary to meet unassigned
    rectified_inds = (inds + np.arange(num_profs)[:, None] *
                      game.num_role_strats)
    ind_mask = (np.arange(game.num_role_strats) <
                game.role_repeat(game.role_starts + unassigned))
    expand_profs.flat[rectified_inds[ind_mask]] += 1
    return expand_profs


def _reduce_rsym_profiles(game, profiles, full_players, reduced_players):
    """Hierarchically reduces several role symmetric array profiles

    This returns the reduced profiles, and a boolean mask showing which of the
    input profiles were actually reduced, as they might not all reduce."""
    assert np.all(game.role_reduce(profiles) == full_players), \
        "Not all profiles were valid"
    assert (np.all(full_players >= reduced_players) and
            np.all((reduced_players > 0) |
                   ((full_players == 0) & (reduced_players == 0)))), \
        "not all reductions were valid\nfull: {}\nreduced: {}".format(
            full_players, reduced_players)

    rep_red_players = game.role_repeat(reduced_players)
    rep_full_players = game.role_repeat(np.maximum(full_players, 1))
    red_profs = np.ceil(profiles * rep_red_players /
                        rep_full_players).astype(int)
    alternates = np.ceil((profiles - 1) * rep_red_players /
                         rep_full_players).astype(int)

    # What if every strategy was tie broken
    overassigned = game.role_reduce(red_profs) - reduced_players
    # The strategies that could have been tie broken i.e. the profile changed
    diffs = alternates != red_profs
    # These are "possible" to reduce
    reduced = np.all(overassigned <= game.role_reduce(diffs), 1)

    # Move everything into reduced space
    num_reduceable = reduced.sum()
    profiles = profiles[reduced]
    red_profs = red_profs[reduced]
    alternates = alternates[reduced]
    overassigned = overassigned[reduced]
    diffs = diffs[reduced]
    if full_players.ndim > 1:
        full_players = full_players[reduced]
    if reduced_players.ndim > 1:
        reduced_players = reduced_players[reduced]

    # Now we take the hypothetical reduced profiles, and see which ones would
    # have won the tie breaker
    role_order = np.broadcast_to(game.role_repeat(np.arange(game.num_roles)),
                                 (num_reduceable, game.num_role_strats))
    alpha_inds = np.arange(game.num_role_strats)
    alpha_ord = np.broadcast_to(alpha_inds,
                                (num_reduceable, game.num_role_strats))
    rep_red_players = game.role_repeat(np.maximum(reduced_players, 1))
    rep_full_players = game.role_repeat(full_players)
    errors = alternates * rep_full_players / rep_red_players - profiles + 1
    inds = np.lexsort((alpha_ord, -alternates, -errors, ~diffs, role_order), 1)

    # Same as with expansion, map to new space
    rectified_inds = (inds + np.arange(num_reduceable)[:, None] *
                      game.num_role_strats)
    ind_mask = (np.arange(game.num_role_strats) <
                game.role_repeat(game.role_starts + overassigned))
    inc_inds = rectified_inds[ind_mask]
    red_profs.flat[inc_inds] = alternates.flat[inc_inds]

    # Our best guesses might not be accurate, so we have to filter out profiles
    # that don't order correctly
    expand_profs = _expand_rsym_profiles(game, red_profs, full_players,
                                         reduced_players)
    valid = np.all(expand_profs == profiles, 1)
    reduced[reduced] = valid
    return red_profs[valid], reduced


class _Reduction(metaclass=abc.ABCMeta):
    """Base Reduction class

    This defines the reduction interface"""

    def __init__(self, full_game, red_game):
        self.full_game = full_game
        self.red_game = red_game
        assert np.all(full_game.num_players >= red_game.num_players), \
            "All full counts must not be less than reduced counts"

    @abc.abstractmethod
    def expand_profiles(self, profiles):
        """Returns full game profiles that contribute to reduced profile"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def reduce_profiles(self, profiles):
        """Returns reduced profiles that contribute to the full profile"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def reduce_game(self, game, allow_incomplete=False):
        """Convert an input game to a reduced game with new players

        `allow_incomplete` will fill in profiles that only partial payoff data
        is known for.."""
        pass  # pragma: no cover

    @abc.abstractmethod
    def expand_deviation_profiles(self, subgame_mask, role_index=None):
        """Expand profiles that contribute to deviation payoffs"""
        pass  # pragma: no cover


class Hierarchical(_Reduction):
    """Hierarchical Reduction

    Either reduced or full players must be specified, the other will be taken
    from the game.
    """

    def __init__(self, num_strats, full_players, reduced_players):
        super().__init__(rsgame.basegame(full_players, num_strats),
                         rsgame.basegame(reduced_players, num_strats))

    def reduce_game(self, game, allow_incomplete=False):
        """Convert an input game to a reduced game with new players

        Allow incomplete is unused for hierarchical reduction."""
        assert (np.all(game.num_players == self.full_game.num_players) and
                np.all(game.num_strategies == self.full_game.num_strategies)),\
            "The games players don't match up with this reduction"

        if isinstance(game, rsgame.SampleGame):
            if game.num_profiles == 0:
                return rsgame.samplegame(self.red_game.num_players,
                                         game.num_strategies)

            sample_payoffs = []
            profiles = []
            for profs, pays in zip(
                    np.split(game.profiles, game.sample_starts[1:]),
                    game.sample_payoffs):
                red_profiles, mask = _reduce_rsym_profiles(
                    self.full_game, profs, self.full_game.num_players,
                    self.red_game.num_players)
                if mask.any():
                    profiles.append(red_profiles)
                    sample_payoffs.append(pays[mask])

            if profiles:
                profiles = np.concatenate(profiles, 0)
            else:  # No data
                profiles = np.empty((0, game.num_role_strats), dtype=int)
            return rsgame.samplegame(self.red_game.num_players,
                                     game.num_strategies, profiles,
                                     sample_payoffs, False)
        elif isinstance(game, rsgame.Game):
            if game.num_profiles == 0:
                return rsgame.game(self.red_game.num_players,
                                   game.num_strategies)

            profiles = game.profiles
            payoffs = game.payoffs
            red_profiles, mask = _reduce_rsym_profiles(
                self.full_game, profiles, self.full_game.num_players,
                self.red_game.num_players)

            return rsgame.game(self.red_game.num_players, game.num_strategies,
                               red_profiles, payoffs[mask], False)

        else:
            return rsgame.basegame(self.red_game.num_players,
                                   game.num_strategies)

    def reduce_profiles(self, profiles):
        """Reduce a set of profiles"""
        return _reduce_rsym_profiles(
            self.full_game, np.asarray(
                profiles, int), self.full_game.num_players,
            self.red_game.num_players)[0]

    def expand_profiles(self, profiles):
        """Expand a set of profiles"""
        return _expand_rsym_profiles(self.full_game, np.asarray(profiles, int),
                                     self.full_game.num_players,
                                     self.red_game.num_players)

    def expand_deviation_profiles(self, subgame_mask, role_index=None):
        """Expand profiles that contribute to deviation payoffs"""
        return self.expand_profiles(
            subgame.deviation_profiles(self.red_game, subgame_mask,
                                       role_index))

    def __repr__(self):
        return '{}({}, {}, {})'.format(
            self.__class__.__name__,
            self.full_game.num_strategies,
            self.full_game.num_players,
            self.red_game.num_players)


class DeviationPreserving(_Reduction):
    """Deviation Preserving Reduction

    Either reduced or full players must be specified, the other will be taken
    from the game."""

    def __init__(self, num_strats, full_players, reduced_players):
        super().__init__(rsgame.basegame(full_players, num_strats),
                         rsgame.basegame(reduced_players, num_strats))

    def _devs(self, players, num_profs):
        """Return an array of the player counts after deviation"""
        return np.tile(self.full_game.role_repeat(
            players - np.eye(self.full_game.num_roles, dtype=int), 0),
            (num_profs, 1))

    def expand_profiles(self, profiles, return_contributions=False):
        """Expand a set of profiles

        If `return_contributions` then a boolean array of matching shape is
        returned indicating the payoffs that are needed for the initial
        profiles."""
        profiles = np.asarray(profiles, int)
        num_profs = profiles.shape[0]
        dev_profs = profiles[:, None] - np.eye(self.full_game.num_role_strats,
                                               dtype=int)
        dev_profs = np.reshape(dev_profs, (-1, self.full_game.num_role_strats))
        dev_full_players = self._devs(self.full_game.num_players, num_profs)
        dev_red_players = self._devs(self.red_game.num_players, num_profs)

        mask = ~np.any(dev_profs < 0, 1)
        devs = np.eye(self.full_game.num_role_strats, dtype=bool)[None]\
            .repeat(num_profs, 0)\
            .reshape((-1, self.full_game.num_role_strats))[mask]
        dev_full_profs = _expand_rsym_profiles(
            self.full_game, dev_profs[mask], dev_full_players[mask],
            dev_red_players[mask]) + devs
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

    def reduce_profiles(self, profiles, return_contributions=False):
        """Reduces a set of profiles

        If `return_contributions` returns ancillary information.

        Returns
        -------
        red_profs
            The reduced profiles
        red_inds
            Index in red_profs for each payoff value that was reduced.
        full_inds
            Index into profiles for each payoff value that was reduced.
        strat_inds
            Index into a profile for the index of each payoff.  Parallel with
            red_inds and full_inds.
        """
        profiles = np.asarray(profiles, int)
        num_profs = profiles.shape[0]
        dev_profs = profiles[:, None] - np.eye(self.full_game.num_role_strats,
                                               dtype=int)
        dev_profs = np.reshape(dev_profs, (-1, self.full_game.num_role_strats))
        dev_full_players = self._devs(self.full_game.num_players, num_profs)
        dev_red_players = self._devs(self.red_game.num_players, num_profs)
        mask = ~np.any(dev_profs < 0, 1)
        red_profs, reduced = _reduce_rsym_profiles(
            self.full_game, dev_profs[mask], dev_full_players[mask],
            dev_red_players[mask])
        devs = np.eye(self.full_game.num_role_strats, dtype=int)[None]\
            .repeat(num_profs, 0)\
            .reshape((-1, self.full_game.num_role_strats))[mask][reduced]
        red_profs += devs
        red_profs, red_inds = utils.unique_axis(red_profs, return_inverse=True)
        if not return_contributions:
            return red_profs
        else:
            full_inds = np.arange(num_profs)[:, None].repeat(
                self.full_game.num_role_strats, 1).flat[mask][reduced]
            strat_inds = devs.nonzero()[1]
            return red_profs, red_inds, full_inds, strat_inds

    def reduce_game(self, game, allow_incomplete=False):
        """Convert an input game to a reduced game with new players

        If `allow_incomplete` is true, then profiles with incomplete payoff
        data will still be returned. If game is a SampleGame, then the payoff
        with the smallest number of nonzero values will be used."""
        assert (np.all(game.num_players == self.full_game.num_players) and
                np.all(game.num_strategies == self.full_game.num_strategies)),\
            "The games players don't match up with this reduction"

        if isinstance(game, rsgame.SampleGame):
            if game.num_profiles == 0:
                return rsgame.samplegame(self.red_game.num_players,
                                         game.num_strategies)
            # Reduce profiles
            red_profiles, red_inds, full_inds, strat_inds = \
                self.reduce_profiles(game.profiles, True)
            if red_profiles.size == 0:
                return rsgame.samplegame(self.red_game.num_players,
                                         game.num_strategies)

            # Count the number of payoffs for every profile
            counts = game.num_samples.repeat(game.num_sample_profs)[full_inds]
            rprof_index = red_inds * game.num_role_strats + strat_inds
            pay_counts = np.rint(np.bincount(
                rprof_index, counts, red_profiles.size)).astype(int)\
                .reshape(red_profiles.shape)
            # Minimum valid counts for every profile
            mask = red_profiles == 0
            if allow_incomplete:
                # If allow incomplete, then ignore payoffs where count is 0
                mask |= pay_counts == 0
            obs_counts = np.ma.masked_array(pay_counts, mask).min(1).filled(0)
            # Number of profiles with zero samples
            num_zeros = np.sum(obs_counts == 0)
            # Permutation of red_profiles so that number of samples are in
            # order
            perm = np.argsort(obs_counts)
            new_profiles = red_profiles[perm[num_zeros:]]
            obs_counts = obs_counts[perm[num_zeros:]]
            sample_sizes, num_sample_profs = np.unique(obs_counts,
                                                       return_counts=True)
            iperm = np.empty(perm.size, int)
            iperm[perm] = np.arange(perm.size)
            red_inds = iperm[red_inds]
            # All of the payoffs that are still valid
            mask = red_inds >= num_zeros
            red_inds = red_inds[mask] - num_zeros
            full_inds = full_inds[mask]
            strat_inds = strat_inds[mask]
            obs_counts = obs_counts[red_inds]

            # Iterate through payoff data and create arrays of the index in the
            # profile array, and the payoffs from sample payoffs
            payoff_indices = [([], []) for _ in range(sample_sizes.size)]
            size_map = dict(zip(sample_sizes, payoff_indices))

            parts = np.sum(full_inds < game.sample_starts[1:, None], 1)
            spay_inds = full_inds - game.sample_starts.repeat(np.diff(
                np.insert(parts, [0, parts.size], [0, full_inds.size])))
            prof_inds = red_inds * game.num_role_strats + strat_inds
            for pays, pr_inds, sp_inds, s_inds, o_counts in zip(
                    game.sample_payoffs, np.split(prof_inds, parts),
                    np.split(spay_inds, parts), np.split(strat_inds, parts),
                    np.split(obs_counts, parts)):
                uo_counts = np.unique(o_counts)
                num_obs = pays.shape[2]
                for ssize, mask in zip(uo_counts,
                                       o_counts == uo_counts[:, None]):
                    num = mask.sum()
                    spi = np.broadcast_to(sp_inds[mask, None], (num, num_obs))
                    si = np.broadcast_to(s_inds[mask, None], (num, num_obs))
                    samp_inds = np.broadcast_to(np.arange(num_obs),
                                                (num, num_obs))
                    pinds = np.broadcast_to(pr_inds[mask, None],
                                            (num, num_obs)).flat
                    spays = pays[spi.flat, si.flat, samp_inds.flat]
                    plist, splist = size_map[ssize]
                    plist.append(pinds)
                    splist.append(spays)

            offsets = np.insert(num_sample_profs[:-1].cumsum() *
                                game.num_role_strats, 0, 0)
            sample_payoffs = []
            for (prof_list, payoff_list), num_samples, offset in zip(
                    payoff_indices, sample_sizes, offsets):
                prof_inds = np.concatenate(prof_list) - offset
                payoffs = np.concatenate(payoff_list)
                # Permute data to drop unbiasedly
                perm = rand.permutation(payoffs.size)
                prof_inds = prof_inds[perm]
                # We now need to filter out all samples beyond num_samples for
                # each profile. This is done using sorting and some array
                # tricks
                # The fact that this sort is not stable could add some bias,
                # but that seems unlikely given that we already do a uniform
                # shuffle
                order = np.argsort(prof_inds)
                prof_inds = prof_inds[order]
                starts = np.insert(np.nonzero(
                    prof_inds[1:] != prof_inds[:-1])[0] + 1, 0, 0)
                selected = starts[:, None] + np.arange(num_samples)
                payoff_inds = prof_inds[selected] * num_samples
                payoff_inds.shape = (-1, num_samples)
                payoff_inds += np.arange(num_samples)
                payoffs = payoffs[perm][order][selected]
                num_profiles = prof_inds[-1] // game.num_role_strats + 1
                # With all payoffs and indices, put together sample payoffs
                # array
                sample_pays = np.bincount(
                    payoff_inds.flat, payoffs.flat,
                    num_profiles * game.num_role_strats * num_samples)\
                    .reshape(num_profiles, game.num_role_strats, num_samples)
                if allow_incomplete:
                    prof_offset = offset // game.num_role_strats
                    unknown = new_profiles[prof_offset:prof_offset +
                                           num_profiles, :, None] > 0
                    unknown.flat[prof_inds] = False
                    unknown = np.broadcast_to(unknown, sample_pays.shape)
                    sample_pays[unknown] = np.nan
                sample_payoffs.append(sample_pays)

            return rsgame.samplegame(self.red_game.num_players,
                                     game.num_strategies, new_profiles,
                                     sample_payoffs, False)
        elif isinstance(game, rsgame.Game):
            if game.num_profiles == 0:  # Empty
                return rsgame.game(self.red_game.num_players,
                                   game.num_strategies)

            # Reduce
            full_profiles = game.profiles
            full_payoffs = game.payoffs
            red_profiles, red_inds, full_inds, strat_inds = \
                self.reduce_profiles(full_profiles, True)

            if red_profiles.size == 0:  # Empty reduction
                return rsgame.game(self.red_game.num_players,
                                   game.num_strategies)

            # Build mapping from payoffs to reduced profiles, and use bincount
            # to count the number of payoffs mapped to a specific location, and
            # sum the number of payoffs mapped to a specific location
            cum_inds = red_inds * game.num_role_strats + strat_inds
            payoff_vals = full_payoffs[full_inds, strat_inds]
            red_payoffs = np.bincount(
                cum_inds, payoff_vals, red_profiles.size).reshape(
                    red_profiles.shape)
            red_payoff_counts = np.bincount(
                cum_inds, minlength=red_profiles.size).reshape(
                    red_profiles.shape)
            mask = red_payoff_counts > 1
            red_payoffs[mask] /= red_payoff_counts[mask]
            if not allow_incomplete:
                complete_mask = np.all((red_profiles > 0) ==
                                       (red_payoff_counts > 0), 1)
                red_profiles = red_profiles[complete_mask]
                red_payoffs = red_payoffs[complete_mask]
            else:
                unknown = (red_profiles > 0) & (red_payoff_counts == 0)
                red_payoffs[unknown] = np.nan
            return rsgame.game(self.red_game.num_players, game.num_strategies,
                               red_profiles, red_payoffs, False)

        else:
            return rsgame.basegame(self.red_game.num_players,
                                   game.num_strategies)

    def expand_deviation_profiles(self, subgame_mask, role_index=None):
        """Expand profiles that contribute to deviation payoffs"""
        subgame_mask = np.asarray(subgame_mask, bool)
        rdev = np.eye(self.full_game.num_roles, dtype=int)
        support = self.full_game.role_reduce(subgame_mask)

        def dev_profs(red_players, full_players, mask, rs):
            subg = rsgame.basegame(red_players, support)
            sub_profs = subgame.translate(subg.all_profiles(), subgame_mask)
            non_devs = _expand_rsym_profiles(self.full_game, sub_profs,
                                             full_players, red_players)
            ndevs = np.sum(~mask)
            devs = np.zeros((ndevs, self.full_game.num_role_strats), int)
            devs[:, rs:rs + mask.size][:, ~mask] = np.eye(ndevs, dtype=int)
            profs = non_devs[:, None] + devs
            profs.shape = (-1, self.full_game.num_role_strats)
            return profs

        if role_index is None:
            expanded_profs = [dev_profs(red_players, full_players, mask, rs)
                              for red_players, full_players, mask, rs
                              in zip(self.red_game.num_players - rdev,
                                     self.full_game.num_players - rdev,
                                     self.full_game.role_split(subgame_mask),
                                     self.full_game.role_starts)]
            return np.concatenate(expanded_profs)

        else:
            full_players = self.full_game.num_players.copy()
            full_players[role_index] -= 1
            red_players = self.red_game.num_players.copy()
            red_players[role_index] -= 1
            mask = self.full_game.role_split(subgame_mask)[role_index]
            rs = self.full_game.role_starts[role_index]
            return dev_profs(red_players, full_players, mask, rs)

    def __repr__(self):
        return '{}({}, {}, {})'.format(
            self.__class__.__name__,
            self.full_game.num_strategies,
            self.full_game.num_players,
            self.red_game.num_players)


def reduce_game_dpr(game, reduced_players, *args, **kwargs):
    return DeviationPreserving(
        game.num_strategies, game.num_players,
        reduced_players).reduce_game(game, *args, **kwargs)


class Twins(DeviationPreserving):
    """Twins Reduction

    Same as Deviation Preserving, but where the reduced players are two."""

    def __init__(self, num_strats, full_players):
        super().__init__(num_strats, full_players, np.minimum(2, full_players))

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.full_game.num_strategies,
            self.full_game.num_players)


class Identity(_Reduction):
    """Identity reduction (lack of reduction)

    The second parameter can be ignored, but if specified, it must be the same
    as the number pf players."""

    def __init__(self, num_strats, num_players, num_players_=None):
        assert num_players_ is None or np.all(num_players == num_players_)
        game = rsgame.basegame(num_players, num_strats)
        super().__init__(game, game)

    def expand_profiles(self, profiles):
        """Returns full game profiles that contribute to reduced profile"""
        return profiles

    def reduce_profiles(self, profiles):
        """Returns reduced profiles that contribute to the full profile"""
        return profiles

    def reduce_game(self, game, allow_incomplete=False):
        """Convert an input game to a reduced game with new players

        Allow complete is not used."""
        assert (np.all(game.num_players == self.full_game.num_players) and
                np.all(game.num_strategies == self.full_game.num_strategies)),\
            "The games players don't match up with this reduction"
        return game

    def expand_deviation_profiles(self, subgame_mask, role_index=None):
        """Expand profiles that contribute to deviation payoffs"""
        return subgame.deviation_profiles(self.full_game, subgame_mask,
                                          role_index)

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.full_game.num_strategies,
            self.full_game.num_players)
