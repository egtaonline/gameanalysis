"""Utility module that contains code for parsing legacy game formats"""
import itertools
import warnings
from collections import abc

import numpy as np

from gameanalysis import rsgame
from gameanalysis import utils


class GameSerializer(rsgame._StratArray):
    """An object with utilities for serializing a game with names

    Parameters
    ----------
    roles : [role]
        A list of ordered roles. This must be included with ``strategies``.
    strategies : [[strategy]]
        A list of lists of ordered strategies for each role. This must be
        included with ``roles``.
    """

    def __init__(self, role_names, strat_names):
        self.role_names = tuple(role_names)
        self.strat_names = tuple(map(tuple, strat_names))

        super().__init__(np.fromiter(map(len, self.strat_names), int,
                                     len(self.strat_names)))
        if not all(map(utils.is_sorted, self.strat_names)):
            warnings.warn("If strategies aren't sorted, some functions "
                          "won't work as intended")
        self._named_role_index = {r: i for i, r in enumerate(self.role_names)}
        role_strats = itertools.chain.from_iterable(
            ((r, s) for s in strats) for r, strats
            in zip(self.role_names, self.strat_names))
        self._role_strat_index = {(r, s): i for i, (r, s)
                                  in enumerate(role_strats)}
        self._hash = hash((self.role_names, self.strat_names))

    def role_index(self, role):
        """Return the index of a role by name or strat index"""
        return self._named_role_index[role]

    def role_strat_index(self, role, strat):
        """Return the index of a role strat pair"""
        return self._role_strat_index[role, strat]

    def strat_name(self, role_strat_index):
        """Get the strategy name from a full index"""
        role_index = self.role_indices[role_strat_index]
        return self.strat_names[role_index][role_strat_index -
                                            self.role_starts[role_index]]

    def from_prof_json(self, prof, dest=None, dtype=int):
        """Read a profile from json

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
        dtype : dtype, optional
            The dtype of the returned array
        """
        if dest is None:
            dest = np.empty(self.num_role_strats, dtype)
        dest.fill(0)

        # standard egta symmetry groups
        if 'symmetry_groups' in prof:
            for symgrp in prof['symmetry_groups']:
                _, role, strat, count, _ = _unpack_symgrp(**symgrp)
                dest[self.role_strat_index(role, strat)] = count

        # dict profile
        elif all(isinstance(v, abc.Mapping) for v in prof.values()):
            for role, strats in prof.items():
                for strat, count in strats.items():
                    dest[self.role_strat_index(role, strat)] = count

        # payoff profile
        else:
            for role, strats in prof.items():
                for strat, count, _ in strats:
                    dest[self.role_strat_index(role, strat)] = count

        return dest

    def to_prof_json(self, prof):
        """Convert a profile to json"""
        return {role: {strat: count.item() for strat, count
                       in zip(strats, counts) if count > 0}
                for counts, role, strats
                in zip(self.role_split(prof),
                       self.role_names, self.strat_names)}

    def from_payoff_json(self, prof, dest=None):
        """Read a set of payoffs from json

        Parameters
        ----------
        prof : json
            A description of a set of payoffs in a number of formats
        dest : ndarray, optional
            If supplied, ``dest`` will be written to instead of allocating a
            new array.
        """
        if dest is None:
            dest = np.empty(self.num_role_strats, float)
        dest.fill(0)

        # observations but no data
        if not prof.get('observations', True):
            for symgrp in prof['symmetry_groups']:
                _, role, strat, *__ = _unpack_symgrp(**symgrp)
                dest[self.role_strat_index(role, strat)] = np.nan
            return dest

        # summary format
        if 'observations' not in prof and 'symmetry_groups' in prof:
            for symgrp in prof['symmetry_groups']:
                _, role, strat, __, pay = _unpack_symgrp(**symgrp)
                dest[self.role_strat_index(role, strat)] = pay

        # observations format
        elif ('observations' in prof
              and 'symmetry_groups' in prof['observations'][0]):
            ids = {i: self.role_strat_index(r, s) for i, r, s, *_
                   in (_unpack_symgrp(**sg) for sg in prof['symmetry_groups'])}
            counts = np.zeros(self.num_role_strats, int)
            for j, obs in enumerate(prof['observations']):
                for symgrp in obs['symmetry_groups']:
                    i, pay = _unpack_obs(**symgrp)
                    k = ids[i]
                    counts[k] += 1
                    dest[k] += (pay - dest[k]) / counts[k]

        # full format
        elif 'observations' in prof:
            ids = {i: self.role_strat_index(r, s) for i, r, s, *_
                   in (_unpack_symgrp(**sg) for sg in prof['symmetry_groups'])}
            counts = np.zeros(self.num_role_strats, int)
            for j, obs in enumerate(prof['observations']):
                for player in obs['players']:
                    i, pay = _unpack_player(**player)
                    k = ids[i]
                    counts[k] += 1
                    dest[k] += (pay - dest[k]) / counts[k]

        # dict payoff
        elif all(isinstance(v, abc.Mapping) for v in prof.values()):
            for role, strats in prof.items():
                for strat, pay in strats.items():
                    dest[self.role_strat_index(role, strat)] = _mean(pay)

        # profile payoff
        else:
            for role, strats in prof.items():
                for strat, _, pays in strats:
                    dest[self.role_strat_index(role, strat)] = _mean(pays)

        return dest

    def to_payoff_json(self, prof, payoffs):
        """Format a profile and payoffs as json"""
        return {role: {strat: pay.mean() for strat, count, pay
                       in zip(strats, counts, pays) if count > 0}
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       self.role_split(prof),
                       self.role_split(payoffs))}

    def from_profpay_json(self, prof, dest_prof=None, dest_pays=None,
                          dtype=int):
        """Read json as a profile and a payoff"""
        return (self.from_prof_json(prof, dest_prof, dtype),
                self.from_payoff_json(prof, dest_pays))

    def to_profpay_json(self, prof, payoffs):
        """Format a profile and payoffs as json"""
        return {role: [(strat, int(count), float(pay)) for strat, count, pay
                       in zip(strats, counts, pays) if count > 0]
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       self.role_split(prof),
                       self.role_split(payoffs))}

    def from_obs_json(self, prof, dest=None):
        """Read a set of payoff observations

        An observation is a set of payoffs.

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
        def set_dest(dest, num):
            if dest is None:
                dest = np.empty((num, self.num_role_strats), float)
            dest.fill(0)
            return dest

        # summary format
        if 'observations' not in prof and 'symmetry_groups' in prof:
            dest = set_dest(dest, 1)
            for symgrp in prof['symmetry_groups']:
                _, role, strat, __, pay = _unpack_symgrp(**symgrp)
                dest[0, self.role_strat_index(role, strat)] = pay

        # observations format
        elif ('observations' in prof
              and 'symmetry_groups' in prof['observations'][0]):
            dest = set_dest(dest, len(prof['observations']))
            ids = {i: self.role_strat_index(r, s) for i, r, s, *_
                   in (_unpack_symgrp(**sg) for sg in prof['symmetry_groups'])}
            for j, obs in enumerate(prof['observations']):
                for symgrp in obs['symmetry_groups']:
                    i, pay = _unpack_obs(**symgrp)
                    dest[j, ids[i]] = pay

        # full format
        elif 'observations' in prof:
            dest = set_dest(dest, len(prof['observations']))
            ids = {i: self.role_strat_index(r, s) for i, r, s, *_
                   in (_unpack_symgrp(**sg) for sg in prof['symmetry_groups'])}
            counts = np.empty(self.num_role_strats, int)
            for j, obs in enumerate(prof['observations']):
                counts.fill(0)
                for player in obs['players']:
                    i, pay = _unpack_player(**player)
                    k = ids[i]
                    counts[k] += 1
                    dest[j, k] += (pay - dest[j, k]) / counts[k]

        # dict payoff
        elif all(isinstance(v, abc.Mapping) for v in prof.values()):
            val = next(iter(next(iter(prof.values())).values()))
            num = len(val) if isinstance(val, abc.Iterable) else 1
            dest = set_dest(dest, num)
            for role, strats in prof.items():
                for strat, pay in strats.items():
                    dest[:, self.role_strat_index(role, strat)] = pay

        # profile payoff
        else:
            val = next(iter(prof.values()))[0][2]
            num = len(val) if isinstance(val, abc.Iterable) else 1
            dest = set_dest(dest, num)
            for role, strats in prof.items():
                for strat, _, pays in strats:
                    dest[:, self.role_strat_index(role, strat)] = pays

        return dest

    def to_obs_json(self, prof, obs):
        """Format a profile and payoffs as json"""
        return {role: {strat: list(map(float, pay)) for strat, count, pay
                       in zip(strats, counts, pays.T) if count > 0}
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       self.role_split(prof),
                       self.role_split(obs))}

    def from_profobs_json(self, prof, dest_prof=None, dest_obs=None,
                          dtype=int):
        """Convert json into a profile and an observation"""
        return (self.from_prof_json(prof, dest_prof, dtype),
                self.from_obs_json(prof, dest_obs))

    def to_profobs_json(self, prof, obs):
        """Convery profile and observations to prof obs output"""
        return {role: [(strat, int(count), list(map(float, pay)))
                       for strat, count, pay
                       in zip(strats, counts, pays.T) if count > 0]
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       self.role_split(prof),
                       self.role_split(obs))}

    def from_prof_string(self, prof_string, dest=None):
        """Read a profile from a string"""
        if dest is None:
            dest = np.empty(self.num_role_strats, int)
        dest.fill(0)
        for role_str in prof_string.split('; '):
            role, strats = role_str.split(': ', 1)
            for strat_str in strats.split(', '):
                count, strat = strat_str.split(' ', 1)
                dest[self.role_strat_index(role, strat)] = count
        return dest

    def to_prof_string(self, prof):
        """Convert a profile to a string"""
        return '; '.join(
            '{}: {}'.format(role, ', '.join(
                '{:d} {}'.format(count, strat)
                for strat, count in zip(strats, counts) if count > 0))
            for role, strats, counts
            in zip(self.role_names, self.strat_names,
                   self.role_split(prof)))

    def to_prof_printstring(self, prof):
        """Convert a profile to a printable string"""
        prof = np.asarray(prof)
        if np.issubdtype(prof.dtype, int):
            def format_strat(s, p):
                return '\t{}: {:d}\n'.format(s, p)
        elif np.issubdtype(prof.dtype, float):
            def format_strat(s, p):
                return '\t{}: {:>7.2%}\n'.format(s, p)
        else:  # boolean
            def format_strat(s, p):
                return '\t{}\n'.format(s)

        return ''.join(
            '{}:\n{}'.format(role, ''.join(format_strat(s, p)
                                           for p, s in zip(probs, strats)
                                           if p > 0))
            for probs, role, strats
            in zip(self.role_split(prof), self.role_names, self.strat_names)
            if probs.any()
        ).expandtabs(4)

    def from_role_json(self, role_json, dest=None, dtype=float):
        """Format role data as array"""
        if dest is None:
            dest = np.empty(self.num_roles, dtype)
        for role, val in role_json.items():
            dest[self.role_index(role)] = val
        return dest

    def to_role_json(self, role_info):
        """Format role data as json"""
        return {role: info.item() for role, info
                in zip(self.role_names, np.asarray(role_info))}

    def to_deviation_payoff_json(self, profile, payoffs):
        """Format a profile and deviation payoffs as json"""
        supp = np.asarray(profile, bool)
        role_supp = self.role_reduce(supp)
        splits = ((self.num_strategies - 1) * role_supp)[:-1].cumsum()
        print(role_supp, splits, payoffs)
        return {r: {s: {d: p.item() for p, d
                        in zip(dps, (d for d in ses if d != s))}  # noqa
                    for dps, s in zip(np.split(ps, sp.sum()),
                                      (s for s, m in zip(ses, sp) if m))}  # noqa
                for r, ses, ps, sp in zip(self.role_names, self.strat_names,
                                          np.split(payoffs, splits),
                                          self.role_split(supp))}

    def _get_num_players(self, game):
        num_players = np.empty(self.num_roles, int)
        if 'roles' in game:
            for role in game['roles']:
                num_players[self.role_index(role['name'])] = role['count']

        elif 'players' in game:
            for role, count in game['players'].items():
                num_players[self.role_index(role)] = count

        else:
            raise ValueError("Unknown game format: {}".format(game))

        return num_players

    def from_basegame_json(self, game):
        num_players = self._get_num_players(game)
        return rsgame.basegame(num_players, self.num_strategies)

    def to_basegame_json(self, game):
        return {
            'players': dict(zip(self.role_names, map(int, game.num_players))),
            'strategies': dict(zip(self.role_names,
                                   map(list, self.strat_names)))
        }

    def to_basegame_printstring(self, game):
        return (('BaseGame:\n\tRoles: {}\n\tPlayers:\n\t\t{}\n\t'
                 'Strategies:\n\t\t{}').format(
                     ', '.join(self.role_names),
                     '\n\t\t'.join(
                         '{:d}x {}'.format(count, role)
                         for role, count
                         in sorted(zip(self.role_names, game.num_players))),
                     '\n\t\t'.join(
                         '{}:\n\t\t\t{}'.format(role, '\n\t\t\t'.join(strats))
                         for role, strats
                         in sorted(zip(self.role_names, self.strat_names)))
        )).expandtabs(4)

    def from_game_json(self, game):
        num_players = self._get_num_players(game)
        profile_list = game.get('profiles', ())
        num_profs = len(profile_list)
        profiles = np.empty((num_profs, self.num_role_strats), int)
        payoffs = np.empty((num_profs, self.num_role_strats), float)
        for profj, prof, pay in zip(profile_list, profiles, payoffs):
            self.from_profpay_json(profj, prof, pay)
        return rsgame.game(num_players, self.num_strategies, profiles, payoffs)

    def to_game_json(self, game):
        res = self.to_basegame_json(game)
        if isinstance(game, rsgame.Game):
            res['profiles'] = [self.to_profpay_json(*pp) for pp
                               in zip(game.profiles, game.payoffs)]
        else:
            res['profiles'] = []
        return res

    def to_game_printstring(self, game):
        num_profs = game.num_profiles if isinstance(
            game, rsgame.Game) else 0
        return ('{base}\npayoff data for {data:d} out of {total:d} '
                'profiles').format(base=self.to_basegame_printstring(game)[4:],
                                   data=num_profs, total=game.num_all_profiles)

    def from_samplegame_json(self, game):
        num_players = self._get_num_players(game)
        profile_list = game.get('profiles', ())

        sample_map = {}
        for profile in profile_list:
            prof, obs = self.from_profobs_json(profile)
            num_samps = obs.shape[0]
            profls, payls = sample_map.setdefault(num_samps, ([], []))
            profls.append(prof[None])
            payls.append(obs.T[None])

        if sample_map:
            values = [v for _, v in sorted(sample_map.items())]
            profiles = np.concatenate(list(itertools.chain.from_iterable(
                prof for prof, _ in values)))
            sample_payoffs = [np.concatenate(obs) for _, obs in values]
        else:  # No data
            profiles = np.empty((0, self.num_role_strats), dtype=int)
            sample_payoffs = []

        return rsgame.samplegame(num_players, self.num_strategies, profiles,
                                 sample_payoffs)

    def to_samplegame_json(self, game):
        res = self.to_basegame_json(game)
        if isinstance(game, rsgame.SampleGame):
            profiles = game.profiles
            spayoffs = game.sample_payoffs
        elif isinstance(game, rsgame.Game):
            profiles = game.profiles
            spayoffs = [game.payoffs[..., None]]
        else:
            profiles = ()
            spayoffs = ()

        res['profiles'] = [self.to_profobs_json(prof, pay.T) for prof, pay
                           in zip(profiles,
                                  itertools.chain.from_iterable(spayoffs))]
        return res

    def to_samplegame_printstring(self, game):
        str_ = 'Sample' + self.to_game_printstring(game)
        if isinstance(game, rsgame.SampleGame):
            samples = game.num_samples
        elif isinstance(game, rsgame.Game):
            samples = np.ones(1, int)
        else:
            samples = np.empty(0, int)
        if samples.size == 0:
            return str_ + '\nno observations'
        elif samples.size == 1:
            return '{}\n{:d} observation{} per profile'.format(
                str_, samples[0], '' if samples[0] == 1 else 's')
        else:
            return '{}\n{:d} to {:d} observations per profile'.format(
                str_, samples.min(), samples.max())

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.role_names,
                                   self.strat_names)

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.role_names == other.role_names and
                self.strat_names == other.strat_names)


def gameserializer(role_names, strat_names):
    """Static constrictor for GameSerializer

    Parameters
    ----------
    role_names : [str]
    strat_names : [[str]]"""
    return GameSerializer(role_names, strat_names)


def gameserializer_json(json):
    """Read a GameSerializer from json

    Parameters
    ----------
    json : json
        A json representation of a basic game with names. Must either be
        {roles: [{name: <role>, strategies: [<strat>]}]}, or {strategies:
        {<role>: [<strat>]}}."""
    if 'roles' in json:
        desc = json['roles']
        role_names = [j['name'] for j in desc]
        strat_names = [j['strategies'] for j in desc]
    elif 'strategies' in json:
        desc = sorted(json['strategies'].items())
        role_names = [r for r, _ in desc]
        strat_names = [s for _, s in desc]
    else:
        raise ValueError("unparsable json")
    return GameSerializer(role_names, strat_names)


def read_basegame(json):
    """Read a BaseGame and GameSerializer from json"""
    serial = gameserializer_json(json)
    return serial.from_basegame_json(json), serial


def read_game(json):
    """Read a Game and GameSerializer from json"""
    serial = gameserializer_json(json)
    return serial.from_game_json(json), serial


def read_samplegame(json):
    serial = gameserializer_json(json)
    return serial.from_samplegame_json(json), serial


# Convenient unpacking of dictionaries
def _unpack_symgrp(role, strategy, count, payoff=None, id=None, **_):
    return id, role, strategy, count, payoff


def _unpack_obs(id, payoff, **_):
    return id, payoff


def _unpack_player(sid, p, **_):
    return sid, p


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
