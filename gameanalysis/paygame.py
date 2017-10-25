"""Module for games with potentially sparse payoff data"""
import itertools
from collections import abc

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import rsgame
from gameanalysis import utils


# TODO Using array set operations would allow for convenient array operations
# like, "are all of these profiles in the game", however, it requires sorting
# of large void types which is very expensive, less so than just hashing the
# data. Maybe pandas or other libraries have more efficient variants?


class Game(rsgame.RsGame):
    """Role-symmetric data game representation

    This representation uses a sparse mapping from profiles to payoffs for role
    symmetric games. This allows it to capture arbitrary games, as well as
    games that are generated from data.  Payoffs for specific players in a
    profile can be nan to indicate they are missing. The profiles will not be
    listed in `num_complete_profiles` or counted as `in` the game, but their
    data can be accessed via `get_payoffs`, and they will be used for
    calculating deviation payoffs if possible.

    Parameters
    ----------
    role_names : (str,)
        The name of each role.
    strat_names : ((str,),)
        The name of each strategy for each role.
    num_role_players : ndarray
        The number of players per role.
    profiles : ndarray, (num_payoffs, num_strats)
        The profiles for the game. These must be unique, and all valid for the
        game.
    payoffs : ndarray, (num_payoffs, num_strats)
        The payoffs for the game. This must contain zeros for profile, strategy
        pairs that are not played (i.e. zero). All valid payoffs for a profile
        can't be nan, the profile should be omitted instead.
    """

    def __init__(self, role_names, strat_names, num_role_players, profiles,
                 payoffs):
        super().__init__(role_names, strat_names, num_role_players)
        self._profiles = profiles
        self._profiles.setflags(write=False)
        self._payoffs = payoffs
        self._payoffs.setflags(write=False)
        self.num_profiles = profiles.shape[0]

        # compute log dev reps
        player_factorial = np.sum(sps.gammaln(profiles + 1), 1)
        totals = (np.sum(sps.gammaln(self.num_role_players + 1)) -
                  player_factorial)
        with np.errstate(divide='ignore'):
            self._dev_reps = (
                totals[:, None] + np.log(profiles) -
                np.log(self.num_role_players).repeat(self.num_role_strats))
        self._dev_reps.setflags(write=False)

        # Add profile lookup
        self._profile_map = dict(zip(map(utils.hash_array, profiles),
                                     payoffs))
        if np.isnan(payoffs).any():
            self._complete_profiles = frozenset(
                prof for prof, pay in self._profile_map.items()
                if not np.isnan(pay).any())
        else:  # Don't need to store duplicate lookup object
            self._complete_profiles = self._profile_map
        self.num_complete_profiles = len(self._complete_profiles)

    def profiles(self):
        return self._profiles.view()

    def payoffs(self):
        return self._payoffs.view()

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns the minimum payoff for each role"""
        if not self.num_profiles:
            pays = np.empty(self.num_strats)
            pays.fill(np.nan)
        else:
            pays = np.fmin.reduce(np.where(
                self._profiles > 0, self._payoffs, np.nan), 0)
        pays.setflags(write=False)
        return pays

    @utils.memoize
    def max_strat_payoffs(self):
        """Returns the maximum payoff for each role"""
        if not self.num_profiles:
            pays = np.empty(self.num_strats)
            pays.fill(np.nan)
        else:
            pays = np.fmax.reduce(np.where(
                self._profiles > 0, self._payoffs, np.nan), 0)
        pays.setflags(write=False)
        return pays

    def get_payoffs(self, profiles):
        """Returns an array of profile payoffs

        If profile is not in game, an array of nans is returned where profile
        has support."""
        profiles = np.asarray(profiles, int)
        assert self.is_profile(profiles).all()
        prof_view = profiles.reshape((-1, self.num_strats))
        payoffs = np.empty(prof_view.shape, float)
        for prof, pay in zip(prof_view, payoffs):
            hashed = utils.hash_array(prof)
            if hashed not in self._profile_map:
                pay[prof == 0] = 0
                pay[prof > 0] = np.nan
            else:
                np.copyto(pay, self._profile_map[hashed])
        return payoffs.reshape(profiles.shape)

    def deviation_payoffs(self, mix, *, jacobian=False):
        """Computes the expected value of deviating

        More specifically, this is the expected payoff of playing each pure
        strategy played against all opponents playing mix.

        Parameters
        ----------
        mix : ndarray
            The mix all other players are using
        jacobian : bool
            If true, the second returned argument will be the jacobian of the
            deviation payoffs with respect to the mixture. The first axis is
            the deviating strategy, the second axis is the strategy in the mix
            the jacobian is taken with respect to. The values that are marked
            nan are not very aggressive, so don't rely on accurate nan values
            in the jacobian. Additionally, this uses sparse data, so jacobian
            values for mixtures that are close to zero are intentionally
            reported as zero instead of looking for data to support some value.
        """
        # TODO It wouldn't be hard to extend this to multiple mixtures, which
        # would allow array calculation of mixture regret. Support would have
        # to be iterative though.
        mix = np.asarray(mix, float)
        nan_mask = np.empty_like(mix, dtype=bool)

        # Fill out mask where we don't have data
        if self.is_complete():
            nan_mask.fill(False)
        elif self.is_empty():
            nan_mask.fill(True)
        else:
            # These calculations are approximate, but for games we can do
            # anything with, the size is bounded, and so numeric methods are
            # actually exact.
            support = mix > 0
            strats = np.add.reduceat(support, self.role_starts)
            devs = self._profiles[:, ~support]
            num_supp = utils.game_size(self.num_role_players, strats).prod()
            dev_players = self.num_role_players - \
                np.eye(self.num_roles, dtype=int)
            role_num_dev = utils.game_size(dev_players, strats).prod(1)
            num_dev = role_num_dev.repeat(self.num_role_strats)[~support]

            nan_mask[support] = np.all(devs == 0, 1).sum() < num_supp
            nan_mask[~support] = devs[devs.sum(1) == 1].sum(0) < num_dev

        # Compute values
        if not nan_mask.all():
            # zero_prob effectively makes 0^0=1 and 0/0=0.
            log_mix = np.log(mix + _TINY)
            prof_prob = np.sum(self._profiles * log_mix, 1, keepdims=True)
            with np.errstate(under='ignore'):
                # Ignore underflow caused when profile probability is not
                # representable in floating point.
                probs = np.exp(prof_prob + self._dev_reps - log_mix)
            zero_prob = _TINY * self.num_players
            # Mask out nans
            weighted_payoffs = probs * np.where(probs > zero_prob,
                                                self._payoffs, 0)
            devs = np.sum(weighted_payoffs, 0)

        else:
            devs = np.empty(self.num_strats)

        devs[nan_mask] = np.nan

        if not jacobian:
            return devs

        if not nan_mask.all():
            tmix = mix + self.zero_prob.repeat(self.num_role_strats)
            product_rule = self._profiles[:, None] / tmix - np.diag(1 / tmix)
            dev_jac = np.sum(weighted_payoffs[..., None] * product_rule, 0)
            dev_jac -= np.repeat(
                np.add.reduceat(dev_jac, self.role_starts, 1) /
                self.num_role_strats, self.num_role_strats, 1)
        else:
            dev_jac = np.empty((self.num_strats, self.num_strats))

        dev_jac[nan_mask] = np.nan
        return devs, dev_jac

    def normalize(self):
        """Return a normalized game"""
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0)] = 1
        offset = np.repeat(self.min_role_payoffs(), self.num_role_strats)
        payoffs = (self._payoffs - offset) / scale.repeat(self.num_role_strats)
        payoffs[0 == self._profiles] = 0
        return game_replace(self, self._profiles, payoffs)

    def subgame(self, subgame_mask):
        """Remove possible strategies from consideration"""
        subgame_mask = np.asarray(subgame_mask, bool)
        base = rsgame.emptygame_copy(self).subgame(subgame_mask)
        prof_mask = ~np.any(self._profiles * ~subgame_mask, 1)
        profiles = self._profiles[prof_mask][:, subgame_mask]
        payoffs = self._payoffs[prof_mask][:, subgame_mask]
        return Game(base.role_names, base.strat_names, base.num_role_players,
                    profiles, payoffs)

    def __contains__(self, profile):
        """Returns true if all data for that profile exists"""
        return (utils.hash_array(np.asarray(profile, int))
                in self._complete_profiles)

    def from_prof_json(self, prof, dest=None, verify=True):
        """Read a profile from json

        A profile is an assignment from role-strategy pairs to counts. This
        method reads from several formats as specified in parameters.

        Parameters
        ----------
        prof : json
            A description of a profile in a number of formats. The correct
            format will be auto detected and used. The most common are {role:
            {strat: count}}, {role: [(strat, count, payoff)]},
            {symmetry_groups: [{role: role, strategy: strategy, count:
            count}]}.
        dest : ndarray, optional
            If supplied, ``dest`` will be written to instead of allocating a
            new array.
        """
        if dest is None:
            dest = np.empty(self.num_strats, int)
        dest.fill(0)

        try:
            # To parse as format that contains both data types
            self.from_profpay_json(prof, dest_prof=dest, verify=False)
        except ValueError:
            # Only remaining format is straight dictionary
            super().from_prof_json(prof, dest=dest, verify=False)

        assert not verify or self.is_profile(dest), \
            "\"{}\" is not a valid profile".format(prof)
        return dest

    def from_payoff_json(self, pays, dest=None, verify=True):
        """Read a set of payoffs from json

        Parameters
        ----------
        pays : json
            A description of a set of payoffs in a number of formats
        dest : ndarray, optional
            If supplied, ``dest`` will be written to instead of allocating a
            new array.
        """
        if dest is None:
            dest = np.empty(self.num_strats, float)
        dest.fill(0)

        try:
            # To parse as format that contains both data types
            self.from_profpay_json(pays, dest_pays=dest, verify=verify)
        except ValueError:
            # Only remaining format is straight dictionary
            super().from_payoff_json(pays, dest=dest)

        return dest

    def from_profpay_json(self, prof, dest_prof=None, dest_pays=None,
                          verify=True):
        """Read json as a profile and a payoff"""
        if dest_prof is None:
            dest_prof = np.empty(self.num_strats, int)
        if dest_pays is None:
            dest_pays = np.empty(self.num_strats, float)
        dest_prof.fill(0)
        dest_pays.fill(0)

        # observations but no data
        if not prof.get('observations', True):
            for symgrp in prof['symmetry_groups']:
                _, role, strat, count, __ = _unpack_symgrp(**symgrp)
                index = self.role_strat_index(role, strat)
                dest_prof[index] = count
                dest_pays[index] = np.nan

        # summary format
        elif 'observations' not in prof and 'symmetry_groups' in prof:
            for symgrp in prof['symmetry_groups']:
                _, role, strat, count, pay = _unpack_symgrp(**symgrp)
                index = self.role_strat_index(role, strat)
                dest_prof[index] = count
                dest_pays[index] = pay

        # observations format
        elif ('observations' in prof
              and 'symmetry_groups' in prof['observations'][0]):
            ids = {}
            for symgrp in prof['symmetry_groups']:
                i, role, strat, count, _ = _unpack_symgrp(**symgrp)
                index = self.role_strat_index(role, strat)
                ids[i] = index
                dest_prof[index] = count

            counts = np.zeros(self.num_strats, int)
            for j, obs in enumerate(prof['observations']):
                for symgrp in obs['symmetry_groups']:
                    i, pay = _unpack_obs(**symgrp)
                    k = ids[i]
                    counts[k] += 1
                    dest_pays[k] += (pay - dest_pays[k]) / counts[k]

        # full format
        elif 'observations' in prof:
            ids = {}
            for symgrp in prof['symmetry_groups']:
                i, role, strat, count, _ = _unpack_symgrp(**symgrp)
                index = self.role_strat_index(role, strat)
                ids[i] = index
                dest_prof[index] = count

            counts = np.zeros(self.num_strats, int)
            for j, obs in enumerate(prof['observations']):
                for player in obs['players']:
                    i, pay = _unpack_player(**player)
                    k = ids[i]
                    counts[k] += 1
                    dest_pays[k] += (pay - dest_pays[k]) / counts[k]

        # observation from simulation
        elif 'players' in prof:
            for player in prof['players']:
                role, strat, pay = _unpack_obs_player(**player)
                ind = self.role_strat_index(role, strat)
                dest_prof[ind] += 1
                dest_pays[ind] += (pay - dest_pays[ind]) / dest_prof[ind]

        # dict payoff
        elif all(not isinstance(v, abc.Mapping) for v in prof.values()):
            for role, strats in prof.items():
                for strat, count, pays in strats:
                    index = self.role_strat_index(role, strat)
                    dest_prof[index] = count
                    dest_pays[index] = _mean(pays)

        # error
        else:
            raise ValueError("unknown format")

        assert not verify or self.is_profile(dest_prof), \
            "\"{}\" does not define a valid profile".format(prof)
        return dest_prof, dest_pays

    def to_profpay_json(self, payoffs, prof):
        """Format a profile and payoffs as json"""
        return {role: [(strat, int(count), float(pay)) for strat, count, pay
                       in zip(strats, counts, pays) if count > 0]
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       np.split(prof, self.role_starts[1:]),
                       np.split(payoffs, self.role_starts[1:]))}

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_profiles,
                     self.num_complete_profiles))

    def __eq__(self, other):
        return (super().__eq__(other) and
                # Identical profiles
                self.num_profiles == other.num_profiles and
                self.num_complete_profiles == other.num_complete_profiles and
                # Identical payoffs
                not np.setxor1d(utils.axis_to_elem(self._profiles),
                                utils.axis_to_elem(other._profiles)).size and
                # Identical payoffs
                all(np.allclose(pay, other.get_payoffs(prof), equal_nan=True)
                    for prof, pay in zip(self._profiles, self._payoffs)))

    def to_json(self):
        """Fromat a Game as json"""
        res = super().to_json()
        res['profiles'] = [self.to_profpay_json(pay, prof) for prof, pay
                           in zip(self._profiles, self._payoffs)]
        res['type'] = 'game.1'
        return res

    def __repr__(self):
        return '{old}, {data:d} / {total:d})'.format(
            old=super().__repr__()[:-1],
            data=self.num_profiles,
            total=self.num_all_profiles)

    def __str__(self):
        """Fromat basegame as a printable string"""
        return '{}\npayoff data for {:d} out of {:d} profiles'.format(
            super().__str__(), self.num_profiles, self.num_all_profiles)


def game(num_role_players, num_role_strats, profiles, payoffs):
    """Create a game with default names

    Parameters
    ----------
    num_role_players : ndarray-like, int,
        The number of players per role.
    num_role_strats : ndarray-like, int,
        The number of strategies per role.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    payoffs : ndarray-like, float, (num_profiles, num_strats)
        The payoffs for the game.
    """
    return game_replace(rsgame.emptygame(num_role_players, num_role_strats),
                        profiles, payoffs)


def game_names(role_names, num_role_players, strat_names, profiles, payoffs):
    """Create a game with specified names

    Parameters
    ----------
    role_names : [str]
        The name for each role.
    num_role_players : ndarray-like, int,
        The number of players per role.
    strat_names : [[str]]
        The name for each strategy per role.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    payoffs : ndarray-like, float, (num_profiles, num_strats)
        The payoffs for the game.
    """
    return game_replace(
        rsgame.emptygame_names(role_names, num_role_players, strat_names),
        profiles, payoffs)


def game_json(json):
    """Read a Game from json

    This takes a game in any valid payoff format (i.e. output by this or by
    EGTA Online), and converts it into a Game. If several payoff exist, the
    mean is taken. This means that loading a game using this method, and
    loading it as a sample game produce different results, as the sample game
    will truncate extra payoffs for an individual profile, while this will take
    the minimum.  Note, that there is no legitimate way to get a game with that
    structure, but it is possible to write the json.
    """
    base = game_copy(rsgame.emptygame_json(json))
    profiles = json.get('profiles', ())
    if not profiles:
        return base

    num_profs = len(profiles)
    profs = np.empty((num_profs, base.num_strats), int)
    pays = np.empty((num_profs, base.num_strats), float)
    for profj, prof, pay in zip(profiles, profs, pays):
        base.from_profpay_json(profj, prof, pay)
    return game_replace(base, profs, pays)


def game_copy(copy_game):
    """Copy structure and payoffs from an existing game

    Parameters
    ----------
    copy_game : RsGame
        Game to copy data from. This will create a copy with the games profiles
        and payoffs.
    """
    return Game(copy_game.role_names, copy_game.strat_names,
                copy_game.num_role_players, copy_game.profiles(),
                copy_game.payoffs())


def game_replace(copy_game, profiles, payoffs):
    """Copy structure from an existing game with new data

    Parameters
    ----------
    copy_game : Game
        Game to copy structure out of. Structure includes role names, strategy
        names, and the number of players.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    payoffs : ndarray-like, float, (num_profiles, num_strats)
        The payoffs for the game.
    """
    profiles = np.asarray(profiles, int)
    payoffs = np.asarray(payoffs, float)

    assert profiles.shape == payoffs.shape, \
        "profiles and payoffs must be the same shape {} {}".format(
            profiles.shape, payoffs.shape)
    assert profiles.shape[1:] == (copy_game.num_strats,), \
        "profiles must have proper end shape : expected {} but was {}" \
        .format((copy_game.num_strats,), profiles.shape[1:])
    assert np.all(profiles >= 0), "profiles was negative"
    assert np.all(
        np.add.reduceat(profiles, copy_game.role_starts, 1) ==
        copy_game.num_role_players), \
        "not all profiles equaled player total"
    assert not np.any((payoffs != 0) & (profiles == 0)), \
        "there were nonzero payoffs for strategies without players"
    assert not np.all(np.isnan(payoffs) | (profiles == 0), 1).any(), \
        "a profile can't have etirely nan payoffs"
    assert profiles.shape[0] == np.unique(utils.axis_to_elem(profiles)).size, \
        "there can't be any duplicate profiles"

    return Game(copy_game.role_names, copy_game.strat_names,
                copy_game.num_role_players, profiles, payoffs)


class SampleGame(Game):
    """A Role Symmetric Game that has multiple samples per profile

    This behaves the same as a normal Game object, except that it has methods
    for accessing several payoffs per profile. It also has a `resample` method
    which returns a Game with bootstrapped payoffs instead of mean payoffs,
    allowing for easy bootstrapping.

    Parameters
    ----------
    role_names : (str,)
        The name of each role.
    strat_names : ((str,),)
        The name of each strategy for each role.
    num_role_players : ndarray, int
        The number of players per role.
    profiles : ndarray
        The profiles for the game.
    sample_payoffs : (ndarray,)
        The sample payoffs for the game. Each element of the tuple is a set of
        payoff samples grouped by number of samples and parallel with profiles.
        The dimension of each element should be (num_payoffs, num_samples,
        num_strats), where num_payoffs is the number of samples for that number
        of observations. The number of samples for each element of the tuple
        must be distinct, and an element with zero samples is disallowed, it
        should be omitted instead. All requirements for valid payoffs also
        apply.
    """

    def __init__(self, role_names, strat_names, num_role_players, profiles,
                 sample_payoffs):
        super().__init__(
            role_names, strat_names, num_role_players, profiles,
            np.concatenate([s.mean(1) for s in sample_payoffs])
            if sample_payoffs else np.empty((0, profiles.shape[1])))

        self._sample_payoffs = sample_payoffs
        for spay in self._sample_payoffs:
            spay.setflags(write=False)

        self.num_sample_profs = np.fromiter(
            (x.shape[0] for x in sample_payoffs),
            int, len(sample_payoffs))
        self.num_sample_profs.setflags(write=False)
        self.sample_starts = np.insert(
            self.num_sample_profs[:-1].cumsum(), 0, 0)
        self.sample_starts.setflags(write=False)
        self.num_samples = np.fromiter(
            (v.shape[1] for v in sample_payoffs),
            int, len(sample_payoffs))
        self.num_samples.setflags(write=False)

        self._sample_profile_map = None

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns the minimum payoff for each role"""
        mins = np.empty(self.num_strats)
        mins.fill(np.nan)
        for profs, spays in zip(
                np.split(self._profiles, self.sample_starts[1:]),
                self._sample_payoffs):
            sample_mins = np.fmin.reduce(
                np.where(profs[:, None] > 0, spays, np.nan), (0, 1))
            np.fmin(mins, sample_mins, mins)
        mins.setflags(write=False)
        return mins

    @utils.memoize
    def max_strat_payoffs(self):
        """Returns the maximum payoff for each role"""
        maxs = np.empty(self.num_strats)
        maxs.fill(np.nan)
        for profs, spays in zip(
                np.split(self._profiles, self.sample_starts[1:]),
                self._sample_payoffs):
            sample_maxs = np.fmax.reduce(
                np.where(profs[:, None] > 0, spays, np.nan), (0, 1))
            np.fmax(maxs, sample_maxs, maxs)
        maxs.setflags(write=False)
        return maxs

    def sample_payoffs(self):
        """Get the underlying sample payoffs"""
        return self._sample_payoffs

    def resample(self, num_resamples=None, *, independent_profile=False,
                 independent_role=False, independent_strategy=False):
        """Fetch a game with bootstrap sampled payoffs

        Arguments
        ---------
        num_resamples : int
            The number of resamples to take for each realized payoff. By
            default this is equal to the number of observations for that
            profile, yielding proper bootstrap sampling.
        independent_profile : bool
            If true, sample each profile independently. In general, only
            profiles with a different number of observations will be resampled
            independently.
        independent_role : bool
            If true, sample each role independently. Within a profile, the
            payoffs for each role will be drawn independently.
        independent_strategy : bool
            IF true, sample each strategy independently. Within a profile, the
            payoffs for each strategy will be drawn independently. This
            supersceeds `independent_role`.

        Notes
        -----
        Each of the `independent_` arguments will increase the time to do a
        resample, but may produce better results as it will remove correlations
        between payoffs.
        """
        dim2 = (self.num_strats if independent_strategy
                else self.num_roles if independent_role
                else 1)
        payoffs = np.empty_like(self._payoffs)
        for obs, pays in zip(self._sample_payoffs,
                             np.split(payoffs, self.sample_starts[1:])):
            obs = np.rollaxis(obs, 1, 3)
            num_samples = obs.shape[2]
            num_obs_resamples = (num_samples if num_resamples is None
                                 else num_resamples)
            dim1 = obs.shape[0] if independent_profile else 1
            sample = rand.multinomial(num_obs_resamples,
                                      np.ones(num_samples) / num_samples,
                                      (dim1, dim2))
            if independent_role and not independent_strategy:
                sample = sample.repeat(self.num_role_strats, 1)
            np.copyto(pays, np.mean(obs * sample, 2) *
                      (num_samples / num_obs_resamples))
        return Game(self.role_names, self.strat_names, self.num_role_players,
                    self._profiles, payoffs)

    def get_sample_payoffs(self, profile):
        """Get sample payoffs associated with a profile

        This returns an array of shape (num_observations, num_role_strats). If
        profile has no data, num_observations will be 0."""
        if self._sample_profile_map is None:
            self._sample_profile_map = dict(zip(
                map(utils.hash_array, self._profiles),
                itertools.chain.from_iterable(self._sample_payoffs)))
        profile = np.asarray(profile, int)
        assert self.is_profile(profile)
        hashed = utils.hash_array(profile)
        if hashed not in self._sample_profile_map:
            return np.empty((0, self.num_strats), float)
        else:
            return self._sample_profile_map[hashed]

    def flat_profiles(self):
        """Profiles in parallel with flat_payoffs"""
        return self._profiles.repeat(
            self.num_samples.repeat(self.num_sample_profs), 0)

    def flat_payoffs(self):
        """All sample payoffs linearly concatenated together"""
        return np.concatenate([
            pay.reshape((-1, self.num_strats))
            for pay in self._sample_payoffs])

    def normalize(self):
        """Return a normalized SampleGame"""
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0)] = 1
        scale = scale.repeat(self.num_role_strats)
        offset = self.min_role_payoffs().repeat(self.num_role_strats)
        spayoffs = tuple((pays - offset) / scale
                         for pays in self._sample_payoffs)
        for profs, spays in zip(
                np.split(self._profiles, self.sample_starts[1:]), spayoffs):
            spays *= 0 < profs[:, None]
        return samplegame_replace(self, self._profiles, spayoffs)

    def subgame(self, subgame_mask):
        """Remove possible strategies from consideration"""
        subgame_mask = np.asarray(subgame_mask, bool)
        base = rsgame.emptygame_copy(self).subgame(subgame_mask)
        prof_mask = ~np.any(self._profiles * ~subgame_mask, 1)
        profiles = self._profiles[prof_mask][:, subgame_mask]
        sample_payoffs = tuple(
            pays[pmask][..., subgame_mask] for pays, pmask
            in zip(self._sample_payoffs,
                   np.split(prof_mask, self.sample_starts[1:]))
            if pmask.any())
        return SampleGame(base.role_names, base.strat_names,
                          base.num_role_players, profiles, sample_payoffs)

    def from_samplepay_json(self, prof, dest=None):
        """Read a set of payoff samples

        Parameters
        ----------
        prof : json
            A description of a set of profiles and their payoffs. There are
            several formats that are acceptable, they're all output by egta.
        dest : ndarray, options
            If supplied, ``dest`` will be written to instead of allocting a new
            array. This may be hard to use as you need to know how many
            observations are in the json.
        """
        try:
            # samplepay format with profile too
            _, dest = self.from_profsamplepay_json(prof, dest_samplepay=dest)

        except ValueError:
            # Must be {role: {strat: [pay]}}
            num = max(max(len(p) if isinstance(p, abc.Iterable) else 1
                          for p in pays.values())
                      for pays in prof.values())

            if dest is None:
                dest = np.empty((num, self.num_strats), float)
            else:
                assert dest.shape[0] >= num, \
                    "dest_samplepay not large enough for observations"
            dest.fill(0)

            for role, strats in prof.items():
                for strat, pay in strats.items():
                    dest[:, self.role_strat_index(role, strat)] = pay

        return dest

    def to_samplepay_json(self, samplepay, prof=None):
        """Format sample payoffs as json

        If prof is specified, the resulting json will omit payoffs for
        strategies that aren't played.
        """
        if prof is None:
            prof = np.broadcast_to(1, self.num_strats)
        return {role: {strat: list(map(float, pay)) for strat, count, pay
                       in zip(strats, counts, pays.T) if count > 0}
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       np.split(prof, self.role_starts[1:]),
                       np.split(samplepay, self.role_starts[1:], 1))}

    def from_profsamplepay_json(self, prof, dest_prof=None,
                                dest_samplepay=None):
        """Convert json into a profile and an observation"""
        if dest_prof is None:
            dest_prof = np.empty(self.num_strats, int)
        dest_prof.fill(0)

        def get_pay(num):
            dest = dest_samplepay
            if dest is None:
                dest = np.empty((num, self.num_strats), float)
            else:
                assert dest.shape[0] >= num, \
                    "dest_samplepay not large enough for observations"
            dest.fill(0)
            return dest

        # summary format
        if 'observations' not in prof and 'symmetry_groups' in prof:
            dest = get_pay(1)
            for symgrp in prof['symmetry_groups']:
                _, role, strat, count, pay = _unpack_symgrp(**symgrp)
                index = self.role_strat_index(role, strat)
                dest_prof[index] = count
                dest[0, index] = pay

        # observations format
        elif ('observations' in prof
              and 'symmetry_groups' in prof['observations'][0]):
            dest = get_pay(len(prof['observations']))
            ids = {}
            for symgrp in prof['symmetry_groups']:
                i, role, strat, count, _ = _unpack_symgrp(**symgrp)
                index = self.role_strat_index(role, strat)
                ids[i] = index
                dest_prof[index] = count

            for j, obs in enumerate(prof['observations']):
                for symgrp in obs['symmetry_groups']:
                    i, pay = _unpack_obs(**symgrp)
                    dest[j, ids[i]] = pay

        # full format
        elif 'observations' in prof:
            dest = get_pay(len(prof['observations']))
            ids = {}
            for symgrp in prof['symmetry_groups']:
                i, role, strat, count, _ = _unpack_symgrp(**symgrp)
                index = self.role_strat_index(role, strat)
                ids[i] = index
                dest_prof[index] = count

            counts = np.empty(self.num_strats, int)
            for j, obs in enumerate(prof['observations']):
                counts.fill(0)
                for player in obs['players']:
                    i, pay = _unpack_player(**player)
                    k = ids[i]
                    counts[k] += 1
                    dest[j, k] += (pay - dest[j, k]) / counts[k]
                assert np.all(counts == dest_prof), \
                    "full format didn't have payoffs for the correct number of players"  # noqa

        # profile payoff
        elif all(not isinstance(v, abc.Mapping) for v in prof.values()):
            num = max(max(len(p) if isinstance(p, abc.Iterable) else 1
                          for _, __, p in sg)
                      for sg in prof.values())
            dest = get_pay(num)
            for role, strats in prof.items():
                for strat, count, pays in strats:
                    index = self.role_strat_index(role, strat)
                    dest_prof[index] = count
                    dest[:, index] = pays

        # unrecognized
        else:
            raise ValueError("unrecognized format")

        return dest_prof, dest

    def to_profsamplepay_json(self, samplepay, prof):
        """Convery profile and observations to prof obs output"""
        return {role: [(strat, int(count), list(map(float, pay)))
                       for strat, count, pay
                       in zip(strats, counts, pays.T) if count > 0]
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       np.split(prof, self.role_starts[1:]),
                       np.split(samplepay, self.role_starts[1:], 1))}

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), tuple(sorted(self.num_samples))))

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            # Identical sample payoffs
            all(_sample_payoffs_equal(pay, other.get_sample_payoffs(prof))
                for prof, pay in zip(
                    self._profiles,
                    itertools.chain.from_iterable(self._sample_payoffs))))

    def to_json(self):
        """Fromat a SampleGame as json"""
        res = super().to_json()
        res['profiles'] = [
            self.to_profsamplepay_json(pay, prof) for prof, pay
            in zip(self._profiles,
                   itertools.chain.from_iterable(self._sample_payoffs))]
        res['type'] = 'samplegame.1'
        return res

    def __repr__(self):
        samples = self.num_samples
        if samples.size == 0:
            sample_str = '0'
        elif samples.size == 1:
            sample_str = str(samples[0])
        else:
            sample_str = '{:d} - {:d}'.format(samples.min(), samples.max())
        return '{}, {})'.format(super().__repr__()[:-1], sample_str)

    def __str__(self):
        if self.num_samples.size == 0:
            sampstr = 'no observations'
        elif self.num_samples.size == 1:
            samps = self.num_samples[0]
            sampstr = '{:d} observation{} per profile'.format(
                samps, '' if samps == 1 else 's')
        else:
            sampstr = '{:d} to {:d} observations per profile'.format(
                self.num_samples.min(), self.num_samples.max())
        return '{}\n{}'.format(super().__str__(), sampstr)


def _sample_payoffs_equal(p1, p2):
    """Returns true if two sample payoffs are almost equal"""
    # XXX Pathological payoffs will make this fail, but we're testing for
    # equality, so that's not really an issue, as strict permutations will
    # still be valid.
    return (p1.shape[0] == p2.shape[0] and
            np.allclose(p1[np.lexsort(p1.T)], p2[np.lexsort(p2.T)],
                        equal_nan=True))


def samplegame(num_role_players, num_role_strats, profiles,
               sample_payoffs):
    """Create a SampleGame with default names.

    Parameters
    ----------
    num_role_players : ndarray-like, int
        The number of players per role.
    num_role_strats : ndarray-like, int
        The number of strategies per role.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    sample_payoffs : [ndarray-like, float]
        The sample payoffs for the game.
    """
    return samplegame_replace(
        rsgame.emptygame(num_role_players, num_role_strats),
        profiles, sample_payoffs)


def samplegame_names(role_names, num_role_players, strat_names, profiles,
                     sample_payoffs):
    """Create a SampleGame with specified names

    Parameters
    ----------
    role_names : [str]
        The name of each role.
    num_role_players : ndarray
        The number of players for each role.
    strat_names : [[str]]
        The name of each strategy.
    profiles : ndarray
        The profiles for the game.
    sample_payoffs : [ndarray]
        The sample payoffs for the game."""
    return samplegame_replace(
        rsgame.emptygame_names(role_names, num_role_players, strat_names),
        profiles, sample_payoffs)


def samplegame_json(json):
    """Read a SampleGame from json

    This will read any valid payoff game as a sample game. Invalid games will
    produce an empty sample game."""
    base = samplegame_copy(rsgame.emptygame_json(json))
    profiles = json.get('profiles', ())
    if not profiles:
        return base

    sample_map = {}
    for profile in profiles:
        prof, spay = base.from_profsamplepay_json(profile)
        num_samps = spay.shape[0]
        profls, payls = sample_map.setdefault(num_samps, ([], []))
        profls.append(prof[None])
        payls.append(spay[None])

    values = [v for _, v in sorted(sample_map.items())]
    profiles = np.concatenate(list(itertools.chain.from_iterable(
        prof for prof, _ in values)))
    sample_payoffs = tuple(np.concatenate(spay) for _, spay in values)
    return samplegame_replace(base, profiles, sample_payoffs)


def samplegame_copy(copy_game):
    """Copy a SampleGame from another game

    If game defined sample_payoffs, this will be created with those, otherwise
    it will create a game with one sample per payoff.

    Parameters
    ----------
    copy_game : RsGame
        Game to copy data from.
    """
    if hasattr(copy_game, 'sample_payoffs'):
        sample_payoffs = copy_game.sample_payoffs()
    elif not copy_game.is_empty():
        sample_payoffs = (copy_game.payoffs()[:, None],)
    else:
        sample_payoffs = ()
    return SampleGame(copy_game.role_names, copy_game.strat_names,
                      copy_game.num_role_players, copy_game.profiles(),
                      sample_payoffs)


def samplegame_replace(copy_game, profiles, sample_payoffs):
    """Replace payoff data for an existing game

    Parameters
    ----------
    copy_game : BaseGame, optional
        Game to copy information out of.
    profiles : ndarray-like, int, (num_profiles, num_strats)
        The profiles for the game.
    sample_payoffs : [ndarray-like, float]
        The sample payoffs for the game.
    """
    profiles = np.asarray(profiles, int)
    sample_payoffs = tuple(np.asarray(sp) for sp in sample_payoffs)

    assert profiles.shape[1:] == (copy_game.num_strats,), \
        "profiles must have proper end shape : expected {} but was {}" \
        .format((copy_game.num_strats,), profiles.shape[1:])
    assert np.all(profiles >= 0), "profiles were negative"
    assert np.all(
        np.add.reduceat(profiles, copy_game.role_starts, 1) ==
        copy_game.num_role_players), \
        "not all profiles equaled player total"
    assert profiles.shape[0] == np.unique(utils.axis_to_elem(profiles)).size, \
        "there can't be any duplicate profiles"
    assert profiles.shape[0] == sum(sp.shape[0] for sp in sample_payoffs), \
        "profiles and sample_payoffs must have the same number of 'profiles'"
    assert all(sp.shape[2] == copy_game.num_strats for sp in sample_payoffs), \
        "all sample payoffs must have the appropriate number of strategies"
    assert not any(pays.size == 0 for pays in sample_payoffs), \
        "sample_payoffs can't be empty"
    assert len({s.shape[1] for s in sample_payoffs}) == len(sample_payoffs), \
        "Each set of observations must have a unique number or be merged"

    for profs, spays in zip(np.split(profiles, list(itertools.accumulate(
            sp.shape[0] for sp in sample_payoffs[:-1]))), sample_payoffs):
        assert not np.any((spays != 0) & (profs == 0)[:, None]), \
            "some sample payoffs were nonzero for invalid payoffs"
        assert not np.all(np.isnan(spays) | (profs == 0)[:, None], 2).any(), \
            "an observation can't have entirely nan payoffs"
        assert np.all(np.isnan(spays).all(1) | ~np.isnan(spays).any()), \
            "for a given strategy, all payoffs must be nan or non"

    return SampleGame(copy_game.role_names, copy_game.strat_names,
                      copy_game.num_role_players, profiles, sample_payoffs)


# ---------
# Utilities
# ---------


_TINY = np.finfo(float).tiny


def _mean(vals):
    if isinstance(vals, abc.Iterable):
        count = 0
        mean = 0
        for v in vals:
            count += 1
            mean += (v - mean) / count
        return mean if count > 0 else float('nan')
    else:
        return vals


def _unpack_symgrp(role, strategy, count, payoff=None, id=None, **_):
    return id, role, strategy, count, payoff


def _unpack_obs(id, payoff, **_):
    return id, payoff


def _unpack_player(sid, p, **_):
    return sid, p


def _unpack_obs_player(role, strategy, payoff, **_):
    return role, strategy, payoff
