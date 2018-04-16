"""Module for games with potentially sparse payoff data"""
# pylint: disable=too-many-lines
import contextlib
import itertools
from collections import abc

import numpy as np
import numpy.random as rand
import scipy.special as sps

from gameanalysis import rsgame
from gameanalysis import utils


# TODO For real games, there does seem to be a memory benefit (3-10x) for using
# sparse matrices. This is likely due to the fact that for real games we
# prioritize low support sampling. scipy sparse isn't a great api for this
# usage, but maybe there are things we can do to make this more feasible. Any
# implementation should probably still be based around scipy sparse, so we
# should check speed too before doing anything drastic.
# However, it is worth noting that the density of a complete profile or payoff
# matrix is \frac{\sum_r \frac{s_r n_r}{s_r + n_r - 1}}{\sum_r s_r}. This means
# that the density goes asymptotically to 1 as the number of players increases,
# but to 0 as the strategies goes to infinity, however, strategies are
# generally more fixed, and players are more limiting. Also, for a single role
# game, the number of strategies would have to be more than 3x the number of
# players to get a benefit, which is infeasible in most circumstances. What
# this ultimately implies is that there's not an asymptotic argument to support
# sparsity, so it should probably be done on a case by case basis.


class _Game(rsgame._RsGame): # pylint: disable=protected-access
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

    def __init__( # pylint: disable=too-many-arguments
            self, role_names, strat_names, num_role_players, profiles,
            payoffs):
        super().__init__(role_names, strat_names, num_role_players)
        self._profiles = profiles
        self._profiles.setflags(write=False)
        self._payoffs = payoffs
        self._payoffs.setflags(write=False)
        self._num_profiles = profiles.shape[0]

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
        self._num_complete_profiles = len(self._complete_profiles)

    @property
    def num_profiles(self):
        return self._num_profiles

    @property
    def num_complete_profiles(self):
        return self._num_complete_profiles

    def profiles(self):
        return self._profiles.view()

    def payoffs(self):
        return self._payoffs.view()

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns the minimum payoff for each role"""
        if not self.num_profiles:
            pays = np.full(self.num_strats, np.nan)
        else:
            pays = np.fmin.reduce(np.where(
                self._profiles > 0, self._payoffs, np.nan), 0)
        pays.setflags(write=False)
        return pays

    @utils.memoize
    def max_strat_payoffs(self):
        """Returns the maximum payoff for each role"""
        if not self.num_profiles:
            pays = np.full(self.num_strats, np.nan)
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
        utils.check(self.is_profile(profiles).all(), 'profiles must be valid')
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

    def deviation_payoffs( # pylint: disable=too-many-statements,too-many-branches,too-many-locals,arguments-differ
            self, mixture, *, jacobian=False, ignore_incomplete=False, **_):
        """Computes the expected value of deviating

        More specifically, this is the expected payoff of playing each pure
        strategy played against all opponents playing mix.

        Parameters
        ----------
        mixture : ndarray
            The mix all other players are using
        jacobian : bool
            If true, the second returned argument will be the jacobian of the
            deviation payoffs with respect to the mixture. The first axis is
            the deviating strategy, the second axis is the strategy in the mix
            the derivative is taken with respect to. For this to be calculated
            correctly, the game must be complete. Thus if the game is not
            complete, this will be all nan.
        ignore_incomplete : bool, optional
            If True, a "best estimate" will be returned for incomplete data.
            This means that instead of marking a payoff where all deviations
            aren't known as nan, the probability will be renormalized by the
            mass that is known, creating a biased estimate based of the data
            that is present.
        """
        mixture = np.asarray(mixture, float)
        supp = mixture > 0
        nan_mask = np.empty_like(mixture, dtype=bool)

        # Fill out mask where we don't have data
        if ignore_incomplete or self.is_complete():
            nan_mask.fill(False)
        elif self.is_empty():
            nan_mask.fill(True)
        else:
            # These calculations are approximate, but for games we can do
            # anything with, the size is bounded, and so numeric methods are
            # actually exact.
            strats = np.add.reduceat(supp, self.role_starts)
            devs = self._profiles[:, ~supp]
            num_supp = utils.game_size(self.num_role_players, strats).prod()
            dev_players = self.num_role_players - \
                np.eye(self.num_roles, dtype=int)
            role_num_dev = utils.game_size(dev_players, strats).prod(1)
            num_dev = role_num_dev.repeat(self.num_role_strats)[~supp]

            nan_mask[supp] = np.all(devs == 0, 1).sum() < num_supp
            nan_mask[~supp] = devs[devs.sum(1) == 1].sum(0) < num_dev

        # Compute values
        if not nan_mask.all():
            # zero_prob effectively makes 0^0=1 and 0/0=0.
            zmix = mixture + self.zero_prob.repeat(self.num_role_strats)
            log_mix = np.log(zmix)
            prof_prob = self._profiles.dot(log_mix)[:, None]
            with np.errstate(under='ignore'):
                # Ignore underflow caused when profile probability is not
                # representable in floating point.
                probs = np.exp(prof_prob + self._dev_reps - log_mix)

            if ignore_incomplete:
                # mask out nans
                mask = np.isnan(self._payoffs)
                payoffs = np.where(mask, 0, self._payoffs)
                probs[mask] = 0
            else:
                payoffs = self._payoffs

            # Mask out nans
            zprob = self.zero_prob.dot(self.num_role_players)
            # TODO This threshold causes large errors in the jacobian when we
            # look at sparse mixtures. This should probably be addressed, but
            # it's unclear how without making this significantly slower.
            nan_pays = np.where(probs > zprob, payoffs, 0)
            devs = np.einsum('ij,ij->j', probs, nan_pays)
            devs[nan_mask] = np.nan

        else:
            devs = np.full(self.num_strats, np.nan)

        if ignore_incomplete:
            tprobs = probs.sum(0)
            tsupp = tprobs > 0
            devs[tsupp] /= tprobs[tsupp]
            devs[~tsupp] = np.nan

        if not jacobian:
            return devs

        if ignore_incomplete or not nan_mask.all():
            dev_profs = (self._profiles[:, None] -
                         np.eye(self.num_strats, dtype=int))
            dev_jac = np.einsum(
                'ij,ij,ijk->jk', probs, nan_pays, dev_profs) / zmix
            if ignore_incomplete:
                dev_jac -= (np.einsum('ij,ijk->jk', probs, dev_profs) *
                            devs[:, None] / zmix)
                dev_jac[tsupp] /= tprobs[tsupp, None]
                dev_jac[~tsupp] = np.nan
            # TODO This is a little conservative and could be relaxed but would
            # require extra computation
            if not self.is_complete():
                dev_jac[nan_mask | ~supp] = np.nan
        else:
            dev_jac = np.full((self.num_strats,) * 2, np.nan)

        return devs, dev_jac

    def restrict(self, restriction):
        """Remove possible strategies from consideration"""
        restriction = np.asarray(restriction, bool)
        base = rsgame.empty_copy(self).restrict(restriction)
        prof_mask = ~np.any(self._profiles * ~restriction, 1)
        profiles = self._profiles[prof_mask][:, restriction]
        payoffs = self._payoffs[prof_mask][:, restriction]
        return _Game(
            base.role_names, base.strat_names, base.num_role_players, profiles,
            payoffs)

    def _add_constant(self, constant):
        with np.errstate(invalid='ignore'):
            new_pays = self._payoffs + np.broadcast_to(
                constant, self.num_roles).repeat(self.num_role_strats)
        new_pays[self._profiles == 0] = 0
        return _Game(
            self.role_names, self.strat_names, self.num_role_players,
            self._profiles, new_pays)

    def _multiply_constant(self, constant):
        with np.errstate(invalid='ignore'):
            new_pays = self._payoffs * np.broadcast_to(
                constant, self.num_roles).repeat(self.num_role_strats)
        return _Game(
            self.role_names, self.strat_names, self.num_role_players,
            self._profiles, new_pays)

    def _add_game(self, othr):
        with np.errstate(invalid='ignore'):
            new_pays = self._payoffs + othr.get_payoffs(self._profiles)
        mask = np.any((~np.isnan(new_pays)) & (self._profiles > 0), 1)
        return _Game(
            self.role_names, self.strat_names, self.num_role_players,
            self._profiles[mask], new_pays[mask])

    def __contains__(self, profile):
        """Returns true if all data for that profile exists"""
        return (utils.hash_array(np.asarray(profile, int))
                in self._complete_profiles)

    def profile_from_json(self, prof, dest=None, *, verify=True):
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
        else:
            utils.check(
                dest.dtype.kind == 'i', 'dest dtype must be integral')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num_strats')
        dest.fill(0)

        try:
            # To parse as format that contains both data types
            self.profpay_from_json(prof, dest_prof=dest, verify=False)
        except ValueError:
            # Only remaining format is straight dictionary
            super().profile_from_json(prof, dest=dest, verify=False)

        utils.check(
            not verify or self.is_profile(dest),
            '"{}" is not a valid profile', prof)
        return dest

    def profile_to_assignment(self, prof):
        """Convert a profile to an assignment string"""
        return {
            role: list(itertools.chain.from_iterable(
                itertools.repeat(strat, val.item()) for strat, val
                in zip(strats, counts)))
            for counts, role, strats
            in zip(np.split(prof, self.role_starts[1:]),
                   self.role_names, self.strat_names)
            if np.any(counts > 0)}

    def payoff_from_json(self, pays, dest=None, *, verify=True): # pylint: disable=arguments-differ
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
        else:
            utils.check(
                dest.dtype.kind == 'f', 'dest dtype must be floating')
            utils.check(
                dest.shape == (self.num_strats,),
                'dest shape must be num strats')
        dest.fill(0)

        try:
            # To parse as format that contains both data types
            self.profpay_from_json(pays, dest_pays=dest, verify=verify)
        except ValueError:
            # Only remaining format is straight dictionary
            super().payoff_from_json(pays, dest=dest)

        return dest

    def profpay_from_json(
            self, prof, dest_prof=None, dest_pays=None, *, verify=True):
        """Read json as a profile and a payoff"""
        if dest_prof is None:
            dest_prof = np.empty(self.num_strats, int)
        if dest_pays is None:
            dest_pays = np.empty(self.num_strats, float)
        dest_prof.fill(0)
        dest_pays.fill(0)

        # observations but no data
        if not prof.get('observations', True):
            self._profpay_from_json_empty_obs(prof, dest_prof, dest_pays)
        # summary format
        elif 'observations' not in prof and 'symmetry_groups' in prof:
            self._profpay_from_json_summ(prof, dest_prof, dest_pays)
        # observations format
        elif ('observations' in prof
              and 'symmetry_groups' in prof['observations'][0]):
            self._profpay_from_json_obs(prof, dest_prof, dest_pays)
        # full format
        elif 'observations' in prof:
            self._profpay_from_json_full(prof, dest_prof, dest_pays)
        # observation from simulation
        elif 'players' in prof:
            self._profpay_from_json_observation(prof, dest_prof, dest_pays)
        # dict payoff
        elif all(not isinstance(v, abc.Mapping) for v in prof.values()):
            self._profpay_from_json_dict(prof, dest_prof, dest_pays)
        # error
        else:
            raise ValueError('unknown format')

        utils.check(
            not verify or self.is_profile(dest_prof),
            '"{}" does not define a valid profile', prof)
        return dest_prof, dest_pays

    def _profpay_from_json_empty_obs(self, prof, dest_prof, dest_pays):
        """Get profile and payoff from empty observations format"""
        for symgrp in prof['symmetry_groups']:
            _, role, strat, count, _ = _unpack_symgrp(**symgrp)
            index = self.role_strat_index(role, strat)
            dest_prof[index] = count
            dest_pays[index] = np.nan

    def _profpay_from_json_summ(self, prof, dest_prof, dest_pays):
        """Get profile and payoff from summary format"""
        for symgrp in prof['symmetry_groups']:
            _, role, strat, count, pay = _unpack_symgrp(**symgrp)
            index = self.role_strat_index(role, strat)
            dest_prof[index] = count
            dest_pays[index] = pay

    def _profpay_from_json_obs(self, prof, dest_prof, dest_pays): # pylint: disable=too-many-locals
        """Get profile and payoff from observations format"""
        ids = {}
        for symgrp in prof['symmetry_groups']:
            i, role, strat, count, _ = _unpack_symgrp(**symgrp)
            index = self.role_strat_index(role, strat)
            ids[i] = index
            dest_prof[index] = count

        for j, obs in enumerate(prof['observations'], 1):
            for symgrp in obs['symmetry_groups']:
                i, pay = _unpack_obs(**symgrp)
                k = ids[i]
                dest_pays[k] += (pay - dest_pays[k]) / j

    def _profpay_from_json_full(self, prof, dest_prof, dest_pays): # pylint: disable=too-many-locals
        """Get profile and payoff from full format"""
        ids = {}
        for symgrp in prof['symmetry_groups']:
            i, role, strat, count, _ = _unpack_symgrp(**symgrp)
            index = self.role_strat_index(role, strat)
            ids[i] = index
            dest_prof[index] = count

        counts = np.zeros(self.num_strats, int)
        for obs in prof['observations']:
            for player in obs['players']:
                i, pay = _unpack_player(**player)
                k = ids[i]
                counts[k] += 1
                dest_pays[k] += (pay - dest_pays[k]) / counts[k]

    def _profpay_from_json_observation(self, prof, dest_prof, dest_pays):
        """Get profile and payoff from observation format"""
        for player in prof['players']:
            role, strat, pay = _unpack_obs_player(**player)
            ind = self.role_strat_index(role, strat)
            dest_prof[ind] += 1
            dest_pays[ind] += (pay - dest_pays[ind]) / dest_prof[ind]

    def _profpay_from_json_dict(self, prof, dest_prof, dest_pays):
        """Get profile and payoff from dict format"""
        for role, strats in prof.items():
            for strat, count, pays in strats:
                index = self.role_strat_index(role, strat)
                dest_prof[index] = count
                dest_pays[index] = _mean(pays)

    def profpay_to_json(self, payoffs, prof):
        """Format a profile and payoffs as json"""
        return {role: [(strat, int(count), float(pay)) for strat, count, pay
                       in zip(strats, counts, pays) if count > 0]
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       np.split(prof, self.role_starts[1:]),
                       np.split(payoffs, self.role_starts[1:]))}

    @utils.memoize
    def __hash__(self):
        return hash((super().__hash__(), self.num_complete_profiles,
                     np.sort(utils.axis_to_elem(self._profiles)).tobytes()))

    def __eq__(self, othr):
        return (super().__eq__(othr) and
                # Identical profiles
                self.num_profiles == othr.num_profiles and
                self.num_complete_profiles == othr.num_complete_profiles and
                self._eq_payoffs(othr))

    def _eq_payoffs(self, othr):
        """Identical profiles and payoffs conditioned on all else equal"""
        # pylint: disable-msg=protected-access
        sord = np.argsort(utils.axis_to_elem(self._profiles))
        oord = np.argsort(utils.axis_to_elem(othr._profiles))
        return (np.all(self._profiles[sord] == othr._profiles[oord]) and
                np.allclose(self._payoffs[sord], othr._payoffs[oord],
                            equal_nan=True))

    def to_json(self):
        """Fromat a Game as json"""
        res = super().to_json()
        res['profiles'] = [self.profpay_to_json(pay, prof) for prof, pay
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
    profiles : ndarray-like, int
        The profiles for the game, with shape (num_profiles, num_strats).
    payoffs : ndarray-like, float
        The payoffs for the game, with shape (num_profiles, num_strats).
    """
    return game_replace(rsgame.empty(num_role_players, num_role_strats),
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
    profiles : ndarray-like, int
        The profiles for the game, with shape (num_profiles, num_strats).
    payoffs : ndarray-like, float
        The payoffs for the game, with shape (num_profiles, num_strats).
    """
    return game_replace(
        rsgame.empty_names(role_names, num_role_players, strat_names),
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
    base = game_copy(rsgame.empty_json(json))
    profiles = json.get('profiles', ())
    if not profiles:
        return base

    num_profs = len(profiles)
    profs = np.empty((num_profs, base.num_strats), int)
    pays = np.empty((num_profs, base.num_strats), float)
    for profj, prof, pay in zip(profiles, profs, pays):
        base.profpay_from_json(profj, prof, pay)
    return game_replace(base, profs, pays)


def game_copy(copy_game):
    """Copy structure and payoffs from an existing game

    Parameters
    ----------
    copy_game : RsGame
        Game to copy data from. This will create a copy with the games profiles
        and payoffs.
    """
    return _Game(
        copy_game.role_names, copy_game.strat_names,
        copy_game.num_role_players, copy_game.profiles(), copy_game.payoffs())


def game_replace(copy_game, profiles, payoffs):
    """Copy structure from an existing game with new data

    Parameters
    ----------
    copy_game : Game
        Game to copy structure out of. Structure includes role names, strategy
        names, and the number of players.
    profiles : ndarray-like, int
        The profiles for the game, with shape (num_profiles, num_strats).
    payoffs : ndarray-like, float
        The payoffs for the game, with shape (num_profiles, num_strats).
    """
    profiles = np.asarray(profiles, int)
    payoffs = np.asarray(payoffs, float)

    utils.check(
        profiles.shape == payoffs.shape,
        'profiles and payoffs must be the same shape {} {}',
        profiles.shape, payoffs.shape)
    utils.check(
        profiles.shape[1:] == (copy_game.num_strats,),
        'profiles must have proper end shape : expected {} but was {}',
        (copy_game.num_strats,), profiles.shape[1:])
    utils.check(np.all(profiles >= 0), 'profiles was negative')
    utils.check(
        np.all(
            np.add.reduceat(profiles, copy_game.role_starts, 1) ==
            copy_game.num_role_players),
        'not all profiles equaled player total')
    utils.check(
        not np.any((payoffs != 0) & (profiles == 0)),
        'there were nonzero payoffs for strategies without players')
    utils.check(
        not np.all(np.isnan(payoffs) | (profiles == 0), 1).any(),
        "a profile can't have entirely nan payoffs")
    utils.check(
        profiles.shape[0] == np.unique(utils.axis_to_elem(profiles)).size,
        "there can't be any duplicate profiles")

    return _Game(
        copy_game.role_names, copy_game.strat_names,
        copy_game.num_role_players, profiles, payoffs)


class _SampleGame(_Game):
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

    def __init__( # pylint: disable=too-many-arguments
            self, role_names, strat_names, num_role_players, profiles,
            sample_payoffs):
        super().__init__(
            role_names, strat_names, num_role_players, profiles,
            np.concatenate([s.mean(1) for s in sample_payoffs])
            if sample_payoffs else np.empty((0, profiles.shape[1])))

        self._sample_payoffs = sample_payoffs
        for spay in self._sample_payoffs:
            spay.setflags(write=False)

        self.num_sample_profs = np.fromiter(  # pragma: no branch
            (x.shape[0] for x in sample_payoffs),
            int, len(sample_payoffs))
        self.num_sample_profs.setflags(write=False)
        self.sample_starts = np.insert(
            self.num_sample_profs[:-1].cumsum(), 0, 0)
        self.sample_starts.setflags(write=False)
        self.num_samples = np.fromiter(  # pragma: no branch
            (v.shape[1] for v in sample_payoffs),
            int, len(sample_payoffs))
        self.num_samples.setflags(write=False)

        self._sample_profile_map = None

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns the minimum payoff for each role"""
        mins = np.full(self.num_strats, np.nan)
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
        maxs = np.full(self.num_strats, np.nan)
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
        return _Game(
            self.role_names, self.strat_names, self.num_role_players,
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
        utils.check(self.is_profile(profile), 'must pass a valid profile')
        hashed = utils.hash_array(profile)
        if hashed not in self._sample_profile_map: # pylint: disable=no-else-return
            return np.empty((0, self.num_strats), float)
        else:
            return self._sample_profile_map[hashed]

    def flat_profiles(self):
        """Profiles in parallel with flat_payoffs"""
        if self.is_empty(): # pylint: disable=no-else-return
            return np.empty((0, self.num_strats), int)
        else:
            return self._profiles.repeat(
                self.num_samples.repeat(self.num_sample_profs), 0)

    def flat_payoffs(self):
        """All sample payoffs linearly concatenated together"""
        if self.is_empty(): # pylint: disable=no-else-return
            return np.empty((0, self.num_strats))
        else:
            return np.concatenate([
                pay.reshape((-1, self.num_strats))
                for pay in self._sample_payoffs])

    def _add_constant(self, constant):
        off = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        with np.errstate(invalid='ignore'):
            new_pays = tuple(
                (profs > 0)[:, None] * (pays + off)
                for profs, pays in zip(
                    np.split(self._profiles, self.sample_starts[1:]),
                    self._sample_payoffs))
        return _SampleGame(
            self.role_names, self.strat_names, self.num_role_players,
            self._profiles, new_pays)

    def _multiply_constant(self, constant):
        mult = np.broadcast_to(constant, self.num_roles).repeat(
            self.num_role_strats)
        with np.errstate(invalid='ignore'):
            new_pays = tuple(pays * mult for pays in self._sample_payoffs)
        return _SampleGame(
            self.role_names, self.strat_names, self.num_role_players,
            self._profiles, new_pays)

    def restrict(self, restriction):
        """Remove possible strategies from consideration"""
        restriction = np.asarray(restriction, bool)
        base = rsgame.empty_copy(self).restrict(restriction)
        prof_mask = ~np.any(self._profiles * ~restriction, 1)
        profiles = self._profiles[prof_mask][:, restriction]
        sample_payoffs = tuple(
            pays[pmask][..., restriction] for pays, pmask
            in zip(self._sample_payoffs,
                   np.split(prof_mask, self.sample_starts[1:]))
            if pmask.any())
        return _SampleGame(
            base.role_names, base.strat_names, base.num_role_players, profiles,
            sample_payoffs)

    def samplepay_from_json(self, prof, dest=None):
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
        with contextlib.suppress(ValueError):
            # samplepay format with profile too
            _, dest = self.profsamplepay_from_json(prof, dest_samplepay=dest)
            return dest

        with contextlib.suppress(ValueError, AttributeError):
            # Must be {role: {strat: [pay]}}
            num = max(max(len(p) if isinstance(p, abc.Iterable) else 1
                          for p in pays.values())
                      for pays in prof.values())

            if dest is None:
                dest = np.empty((num, self.num_strats), float)
            else:
                utils.check(
                    dest.dtype.kind == 'f', 'dest dtype must be floating')
                utils.check(
                    dest.shape == (num, self.num_strats),
                    'dest_samplepay not large enough for observations')
            dest.fill(0)

            for role, strats in prof.items():
                for strat, pay in strats.items():
                    dest[:, self.role_strat_index(role, strat)] = pay
            return dest

        raise ValueError('unknown format')

    def samplepay_to_json(self, samplepay):
        """Format sample payoffs as json"""
        # In a really weird degenerate case, if all payoffs are 0, we'll write
        # out an empty dictionary, which loses information about the number of
        # samples. In that case we arbitrarily write out the first strategy
        # with zero payoffs.
        samplepay = np.asarray(samplepay, float)
        if np.all(samplepay == 0):
            return {self.role_names[0]: {
                self.strat_names[0][0]: [0] * samplepay.shape[0]}}

        return {role: {strat: pay.tolist() for strat, pay
                       in zip(strats, pays)
                       if np.any(pay != 0)}
                for role, strats, pays
                in zip(self.role_names, self.strat_names,
                       np.split(samplepay.T, self.role_starts[1:]))
                if np.any(pays != 0)}

    def profsamplepay_from_json(
            self, prof, dest_prof=None, dest_samplepay=None):
        """Convert json into a profile and an observation"""
        if dest_prof is None:
            dest_prof = np.empty(self.num_strats, int)
        dest_prof.fill(0)

        # summary format
        if 'observations' not in prof and 'symmetry_groups' in prof:
            return self._profsamplepay_from_json_summ(
                prof, dest_prof, dest_samplepay)
        # observations format
        elif ('observations' in prof
              and 'symmetry_groups' in prof['observations'][0]):
            return self._profsamplepay_from_json_obs(
                prof, dest_prof, dest_samplepay)
        # full format
        elif 'observations' in prof:
            return self._profsamplepay_from_json_full(
                prof, dest_prof, dest_samplepay)
        # profile payoff
        elif all(not isinstance(v, abc.Mapping) for v in prof.values()):
            return self._profsamplepay_from_json_prof(
                prof, dest_prof, dest_samplepay)
        # unrecognized
        else:
            raise ValueError('unrecognized format')

    def _get_spay_dest(self, dest, num):
        """Get payoff dest for number of samples"""
        if dest is None:
            return np.zeros((num, self.num_strats), float)
        utils.check(
            dest.shape == (num, self.num_strats),
            'dest_samplepay not large enough for observations')
        dest.fill(0)
        return dest

    def _profsamplepay_from_json_summ(self, prof, dest_prof, dest):
        """Get profile and sample payoff for summary format"""
        dest = self._get_spay_dest(dest, 1)
        for symgrp in prof['symmetry_groups']:
            _, role, strat, count, pay = _unpack_symgrp(**symgrp)
            index = self.role_strat_index(role, strat)
            dest_prof[index] = count
            dest[0, index] = pay
        return dest_prof, dest

    def _profsamplepay_from_json_obs(self, prof, dest_prof, dest): # pylint: disable=too-many-locals
        """Get profile and sample payoff for observation format"""
        dest = self._get_spay_dest(dest, len(prof['observations']))
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
        return dest_prof, dest

    def _profsamplepay_from_json_full(self, prof, dest_prof, dest): # pylint: disable=too-many-locals
        """Get profile and sample payoff for full format"""
        dest = self._get_spay_dest(dest, len(prof['observations']))
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
            utils.check(
                np.all(counts == dest_prof),
                "full format didn't have payoffs for the correct number "
                'of players')
        return dest_prof, dest

    def _profsamplepay_from_json_prof(self, prof, dest_prof, dest):
        """Get profile and sample payoff for profile format"""
        num = max(max(len(p) if isinstance(p, abc.Iterable) else 1
                      for _, __, p in sg)
                  for sg in prof.values())
        dest = self._get_spay_dest(dest, num)
        for role, strats in prof.items():
            for strat, count, pays in strats:
                index = self.role_strat_index(role, strat)
                dest_prof[index] = count
                dest[:, index] = pays
        return dest_prof, dest

    def profsamplepay_to_json(self, samplepay, prof):
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
            self.profsamplepay_to_json(pay, prof) for prof, pay
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
        samples = self.num_sample_profs.dot(self.num_samples)
        if self.num_samples.size == 0:
            sampstr = 'no observations'
        elif self.num_samples.size == 1:
            samps = self.num_samples[0]
            sampstr = '{:d} observation{} per profile'.format(
                samps, '' if samps == 1 else 's')
        else:
            sampstr = '{:d} to {:d} observations per profile'.format(
                self.num_samples.min(), self.num_samples.max())
        return '{}\n{} payoff sample{}\n{}'.format(
            super().__str__(), 'no' if samples == 0 else samples,
            '' if samples == 1 else 's', sampstr)


def _sample_payoffs_equal(pay1, pay2):
    """Returns true if two sample payoffs are almost equal"""
    return pay1.shape[0] == pay2.shape[0] and utils.allclose_perm(
        pay1, pay2, equal_nan=True)


def samplegame(num_role_players, num_role_strats, profiles,
               sample_payoffs):
    """Create a SampleGame with default names

    Parameters
    ----------
    num_role_players : ndarray-like, int
        The number of players per role.
    num_role_strats : ndarray-like, int
        The number of strategies per role.
    profiles : ndarray-like, int
        The profiles for the game, with shape (num_profiles, num_strats).
    sample_payoffs : [ndarray-like, float]
        The sample payoffs for the game.
    """
    return samplegame_replace(
        rsgame.empty(num_role_players, num_role_strats),
        profiles, sample_payoffs)


def samplegame_flat(num_role_players, num_role_strats, profiles, payoffs):
    """Create a SampleGame with default names and flat profiles

    Parameters
    ----------
    num_role_players : ndarray-like, int
        The number of players per role.
    num_role_strats : ndarray-like, int
        The number of strategies per role.
    profiles : ndarray-like, int
        The profiles for the game, potentially with duplicates, with shape
        (num_sample_profiles, num_strats).
    payoffs : ndarray-like, float
        The sample payoffs for the game, in parallel with the profiles they're
        samples from, with shape (num_sample_profiles, num_strats).
    """
    return samplegame_replace_flat(
        rsgame.empty(num_role_players, num_role_strats), profiles, payoffs)


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
        rsgame.empty_names(role_names, num_role_players, strat_names),
        profiles, sample_payoffs)


def samplegame_names_flat(role_names, num_role_players, strat_names, profiles,
                          payoffs):
    """Create a SampleGame with specified names and flat payoffs

    Parameters
    ----------
    role_names : [str]
        The name of each role.
    num_role_players : ndarray
        The number of players for each role.
    strat_names : [[str]]
        The name of each strategy.
    profiles : ndarray-like, int
        The profiles for the game, potentially with duplicates,
        (num_sample_profiles, num_strats).
    payoffs : ndarray-like, float
        The sample payoffs for the game, in parallel with the profiles they're
        samples from, (num_sample_profiles, num_strats).
    """
    return samplegame_replace_flat(
        rsgame.empty_names(role_names, num_role_players, strat_names),
        profiles, payoffs)


def samplegame_json(json):
    """Read a SampleGame from json

    This will read any valid payoff game as a sample game. Invalid games will
    produce an empty sample game."""
    base = samplegame_copy(rsgame.empty_json(json))
    profiles = json.get('profiles', ())
    if not profiles:
        return base

    sample_map = {}
    for profile in profiles:
        prof, spay = base.profsamplepay_from_json(profile)
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
    return _SampleGame(
        copy_game.role_names, copy_game.strat_names,
        copy_game.num_role_players, copy_game.profiles(), sample_payoffs)


def samplegame_replace_flat(copy_game, profiles, payoffs): # pylint: disable=too-many-locals
    """Replace sample payoff data for an existing game

    Parameters
    ----------
    copy_game : BaseGame, optional
        Game to copy information out of.
    profiles : ndarray-like, int
        The profiles for the game, potentially with duplicates, with shape
        (num_sample_profiles, num_strats).
    payoffs : ndarray-like, float
        The sample payoffs for the game, in parallel with the profiles they're
        samples from, with shape (num_sample_profiles, num_strats).
    """
    profiles = np.asarray(profiles, int)
    payoffs = np.asarray(payoffs, float)
    _, ind, inv, counts = np.unique(
        utils.axis_to_elem(profiles), return_index=True, return_inverse=True,
        return_counts=True)
    countso = counts.argsort()
    countsoi = np.empty(counts.size, int)
    countsoi[countso] = np.arange(counts.size)
    cinv = countsoi[inv]
    cinvo = cinv.argsort()
    cinvs = cinv[cinvo]
    payo = (np.insert(np.cumsum(1 - np.diff(cinvs)), 0, 0) + cinvs)[cinvo]
    num_samps, ccounts = np.unique(counts[countso], return_counts=True)
    splits = (num_samps * ccounts)[:-1].cumsum()

    profs = profiles[ind[countso]]
    pays = [pay.reshape((n, c, -1)) for pay, n, c
            in zip(np.split(payoffs[payo], splits), ccounts, num_samps)]
    return samplegame_replace(copy_game, profs, pays)


def samplegame_replace(copy_game, profiles, sample_payoffs):
    """Replace sample payoff data for an existing game

    Parameters
    ----------
    copy_game : BaseGame, optional
        Game to copy information out of.
    profiles : ndarray-like, int
        The profiles for the game, with shape (num_profiles, num_strats).
    sample_payoffs : [ndarray-like, float]
        The sample payoffs for the game.
    """
    profiles = np.asarray(profiles, int)
    sample_payoffs = tuple(np.asarray(sp) for sp in sample_payoffs)

    utils.check(
        profiles.shape[1:] == (copy_game.num_strats,),
        'profiles must have proper end shape : expected {} but was {}',
        (copy_game.num_strats,), profiles.shape[1:])
    utils.check(np.all(profiles >= 0), 'profiles were negative')
    utils.check(
        np.all(
            np.add.reduceat(profiles, copy_game.role_starts, 1) ==
            copy_game.num_role_players),
        'not all profiles equaled player total')
    utils.check(
        profiles.shape[0] == np.unique(utils.axis_to_elem(profiles)).size,
        "there can't be any duplicate profiles")
    utils.check(
        profiles.shape[0] == sum(sp.shape[0] for sp in sample_payoffs),
        'profiles and sample_payoffs must have the same number of "profiles"')
    utils.check(
        all(sp.shape[2] == copy_game.num_strats for sp in sample_payoffs),
        'all sample payoffs must have the appropriate number of strategies')
    utils.check(
        not any(pays.size == 0 for pays in sample_payoffs),
        "sample_payoffs can't be empty")
    utils.check(
        len({s.shape[1] for s in sample_payoffs}) == len(sample_payoffs),
        'each set of observations must have a unique number or be merged')

    for profs, spays in zip(np.split(profiles, list(itertools.accumulate(
            sp.shape[0] for sp in sample_payoffs[:-1]))), sample_payoffs):
        utils.check(
            not np.any((spays != 0) & (profs == 0)[:, None]),
            'some sample payoffs were nonzero for invalid payoffs')
        utils.check(
            not np.all(np.isnan(spays) | (profs == 0)[:, None], 2).any(),
            "an observation can't have entirely nan payoffs")
        utils.check(
            np.all(np.isnan(spays).all(1) | ~np.isnan(spays).any()),
            'for a given strategy, all payoffs must be nan or non')

    return _SampleGame(
        copy_game.role_names, copy_game.strat_names,
        copy_game.num_role_players, profiles, sample_payoffs)


# ---------
# Utilities
# ---------


def _mean(vals):
    """Streaming mean of some values"""
    if not isinstance(vals, abc.Iterable):
        return vals

    count = 0
    mean = 0
    for val in vals:
        count += 1
        mean += (val - mean) / count
    return mean if count > 0 else float('nan')


def _unpack_symgrp(role, strategy, count, payoff=None, id=None, **_): # pylint: disable=invalid-name,redefined-builtin
    """Unpack a symmetry group"""
    return id, role, strategy, count, payoff


def _unpack_obs(id, payoff, **_): # pylint: disable=invalid-name,redefined-builtin
    """Unpack an observation"""
    return id, payoff


def _unpack_player(sid, p, **_): # pylint: disable=invalid-name
    """Unpack a player"""
    return sid, p


def _unpack_obs_player(role, strategy, payoff, **_):
    """Unpack an observation player"""
    return role, strategy, payoff
