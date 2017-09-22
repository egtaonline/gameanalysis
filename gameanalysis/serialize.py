"""Utility module that contains code for parsing legacy game formats"""
import itertools
import warnings
from collections import abc

import numpy as np

from gameanalysis import rsgame
from gameanalysis import utils


class _BaseSerializer(rsgame._StratArray):
    """An object with utilities for serializing objects with names

    Parameters
    ----------
    roles : [role]
        A list of ordered roles. It is probably best if these are in
        lexicographic order.
    strategies : [[strategy]]
        A list of lists of ordered strategies for each role. This must be
        included with ``roles``. For some algorithms to work as desired, these
        strategies should be in lexicographic order.
    """

    def __init__(self, role_names, strat_names):
        super().__init__(np.fromiter(map(len, strat_names), int,
                                     len(strat_names)))
        self.role_names = role_names
        self.strat_names = strat_names

        if not all(map(utils.is_sorted, self.strat_names)):
            warnings.warn("If strategies aren't sorted, some functions "
                          "won't work as intended")
        self._named_role_index = {r: i for i, r in enumerate(self.role_names)}
        role_strats = itertools.chain.from_iterable(
            ((r, s) for s in strats) for r, strats
            in zip(self.role_names, self.strat_names))
        self._role_strat_index = {(r, s): i for i, (r, s)
                                  in enumerate(role_strats)}
        role_strat_devs = itertools.chain.from_iterable(
            itertools.chain.from_iterable(
                ((r, s, d) for d in strats if d != s)
                for s in strats)
            for r, strats
            in zip(self.role_names, self.strat_names))
        self._role_strat_dev_index = {(r, s, d): i for i, (r, s, d)
                                      in enumerate(role_strat_devs)}
        self._hash = hash((self.role_names, self.strat_names))

    def role_index(self, role):
        """Return the index of a role by name or strat index"""
        return self._named_role_index[role]

    def role_strat_index(self, role, strat):
        """Return the index of a role strat pair"""
        return self._role_strat_index[role, strat]

    def role_strat_dev_index(self, role, strat, dev):
        return self._role_strat_dev_index[role, strat, dev]

    def strat_name(self, role_strat_index):
        """Get the strategy name from a full index"""
        role_index = self.role_indices[role_strat_index]
        return self.strat_names[role_index][role_strat_index -
                                            self.role_starts[role_index]]

    def from_prof_json(self, prof, dest=None):
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
            self.from_profpay_json(prof, dest_prof=dest)

        except ValueError:
            # Only remaining format is straight dictionary
            for role, strats in prof.items():
                for strat, count in strats.items():
                    dest[self.role_strat_index(role, strat)] = count

        return dest

    def to_prof_json(self, prof):
        """Convert a profile array to json"""
        return {role: {strat: count.item() for strat, count
                       in zip(strats, counts) if count > 0}
                for counts, role, strats
                in zip(np.split(prof, self.role_starts[1:]),
                       self.role_names, self.strat_names)}

    def from_mix_json(self, mix, dest=None, verify=True):
        """Read a json mixture into an array"""
        if dest is None:
            dest = np.empty(self.num_strats, float)
        dest.fill(0)

        for role, strats in mix.items():
            for strat, prob in strats.items():
                dest[self.role_strat_index(role, strat)] = prob

        assert not verify or self.is_mixture(dest), \
            "\"{}\" does not define a valid mixture".format(mix)
        return dest

    def to_mix_json(self, mix):
        """Convert a mixture array to json"""
        return self.to_prof_json(mix)

    def from_subgame_json(self, subg, dest=None, verify=True):
        """Read a json subgame into an array"""
        if dest is None:
            dest = np.empty(self.num_strats, bool)
        dest.fill(False)

        for role, strats in subg.items():
            for strat in strats:
                dest[self.role_strat_index(role, strat)] = True

        assert not verify or self.is_subgame(dest), \
            "\"{}\" does not define a valid subgame".format(subg)
        return dest

    def to_subgame_json(self, subg):
        """Convert a subgame array to json"""
        return {role: [strat for strat, inc in zip(strats, mask) if inc]
                for mask, role, strats
                in zip(np.split(subg, self.role_starts[1:]),
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
            dest = np.empty(self.num_strats, float)
        dest.fill(0)

        try:
            # To parse as format that contains both data types
            self.from_profpay_json(prof, dest_pays=dest)

        except ValueError:
            # Only remaining format is straight dictionary
            for role, strats in prof.items():
                for strat, pay in strats.items():
                    dest[self.role_strat_index(role, strat)] = _mean(pay)

        return dest

    def to_payoff_json(self, payoffs, prof=None):
        """Format payoffs as json

        If an optional profile is specified, the json will be sparsified to
        only strategies with at least one player.

        Parameters
        ----------
        payoffs : ndarray
            The payoffs to serialize.
        prof : ndarray, optional
            The profile the payoffs correspond to, specifying it allows the
            written json to omit strategies that aren't played.
        """
        if prof is None:
            prof = np.broadcast_to(True, self.num_strats)
        return {role: {strat: pay.mean() for strat, count, pay
                       in zip(strats, counts, pays) if count > 0}
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       np.split(prof, self.role_starts[1:]),
                       np.split(payoffs, self.role_starts[1:]))}

    def from_profpay_json(self, prof, dest_prof=None, dest_pays=None):
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

        return dest_prof, dest_pays

    def to_profpay_json(self, payoffs, prof):
        """Format a profile and payoffs as json"""
        return {role: [(strat, int(count), float(pay)) for strat, count, pay
                       in zip(strats, counts, pays) if count > 0]
                for role, strats, counts, pays
                in zip(self.role_names, self.strat_names,
                       np.split(prof, self.role_starts[1:]),
                       np.split(payoffs, self.role_starts[1:]))}

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

    def from_prof_str(self, prof_str, dest=None):
        """Read a profile from a string"""
        if dest is None:
            dest = np.empty(self.num_strats, int)
        dest.fill(0)
        for role_str in prof_str.split('; '):
            role, strats = role_str.split(': ', 1)
            for strat_str in strats.split(', '):
                count, strat = strat_str.split(' ', 1)
                dest[self.role_strat_index(role, strat)] = count
        return dest

    def to_prof_str(self, prof):
        """Convert a profile to a string"""
        return '; '.join(
            '{}: {}'.format(role, ', '.join(
                '{:d} {}'.format(count, strat)
                for strat, count in zip(strats, counts) if count > 0))
            for role, strats, counts
            in zip(self.role_names, self.strat_names,
                   np.split(prof, self.role_starts[1:])))

    def to_prof_printstr(self, prof):
        """Convert a profile to a printable string"""
        return ''.join(
            '{}:\n{}'.format(role, ''.join(
                '    {}: {:d}\n'.format(s, c)
                for c, s in zip(counts, strats)
                if c > 0))
            for counts, role, strats
            in zip(np.split(np.asarray(prof), self.role_starts[1:]),
                   self.role_names, self.strat_names))

    def to_mix_printstr(self, mix):
        """Convert a mixture to a printable string"""
        return ''.join(
            '{}:\n{}'.format(role, ''.join(
                '    {}: {:>7.2%}\n'.format(s, p)
                for p, s in zip(probs, strats)
                if p > 0))
            for probs, role, strats
            in zip(np.split(np.asarray(mix), self.role_starts[1:]),
                   self.role_names, self.strat_names))

    def to_subgame_printstr(self, subg):
        """Convert a subgame to a printable string"""
        return ''.join(
            '{}:\n{}'.format(role, ''.join(
                '    {}\n'.format(s)
                for m, s in zip(mask, strats)
                if m))
            for mask, role, strats
            in zip(np.split(np.asarray(subg), self.role_starts[1:]),
                   self.role_names, self.strat_names))

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

    def from_dev_payoff_json(self, deviations, dest=None):
        if dest is None:
            dest = np.empty(self.num_devs)
        dest.fill(0)

        for role, strats in deviations.items():
            for strat, devs in strats.items():
                for dev, val in devs.items():
                    dest[self.role_strat_dev_index(role, strat, dev)] = val

        return dest

    def to_dev_payoff_json(self, payoffs, profile=None):
        """Format a profile and deviation payoffs as json"""
        payoffs = np.asarray(payoffs, float)
        supp = (np.ones(self.num_strats, bool) if profile is None
                else np.asarray(profile, bool))
        return {r: {s: {d: float(pay) for pay, d
                        in zip(spays, (d for d in ses if d != s))}  # noqa
                    for spays, s, su
                    in zip(np.split(rpay, n), ses, sup)
                    if su}
                for r, ses, n, rpay, sup
                in zip(self.role_names, self.strat_names,
                       self.num_role_strats,
                       np.split(payoffs, self.dev_role_starts[1:]),
                       np.split(supp, self.role_starts[1:]))}

    def from_json(self, game):
        """Read a BaseGame from json"""
        num_role_players = np.empty(self.num_roles, int)
        if 'roles' in game:
            for role in game['roles']:
                num_role_players[self.role_index(role['name'])] = role['count']
        elif 'players' in game:
            for role, count in game['players'].items():
                num_role_players[self.role_index(role)] = count
        else:
            raise ValueError("Unknown game format: {}".format(game))
        return rsgame.emptygame(num_role_players, self.num_role_strats)

    def to_json(self, game):
        """Format basegame as json"""
        return {
            'players': dict(zip(self.role_names,
                                map(int, game.num_role_players))),
            'strategies': dict(zip(self.role_names,
                                   map(list, self.strat_names)))
        }

    def to_printstr(self, game):
        """Fromat basegame as a printable string"""
        return (
            ('{}:\n    Roles: {}\n    Players:\n        {}\n    '
             'Strategies:\n        {}').format(
                 game.__class__.__name__,
                 ', '.join(self.role_names),
                 '\n        '.join(
                     '{:d}x {}'.format(count, role)
                     for role, count
                     in zip(self.role_names, game.num_role_players)),
                 '\n        '.join(
                     '{}:\n            {}'.format(
                         role, '\n            '.join(strats))
                     for role, strats
                     in zip(self.role_names, self.strat_names))))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.role_names,
                                   self.strat_names)

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.role_names == other.role_names and
                self.strat_names == other.strat_names)


class GameSerializer(_BaseSerializer):
    def __init__(self, role_names, strat_names):
        super().__init__(role_names, strat_names)

    def from_json(self, game):
        """Read a Game from json"""
        base = super().from_json(game)
        profile_list = game.get('profiles', ())
        num_profs = len(profile_list)
        profiles = np.empty((num_profs, self.num_strats), int)
        payoffs = np.empty((num_profs, self.num_strats), float)
        for profj, prof, pay in zip(profile_list, profiles, payoffs):
            self.from_profpay_json(profj, prof, pay)
        return rsgame.game_replace(base, profiles, payoffs)

    def to_json(self, game):
        """Fromat a Game as json"""
        res = super().to_json(game)
        res['profiles'] = [self.to_profpay_json(pay, prof) for prof, pay
                           in zip(game.profiles, game.payoffs)]
        res['type'] = 'game.1'
        return res

    def to_printstr(self, game):
        """Format game as a printable string"""
        return '{}\npayoff data for {:d} out of {:d} profiles'.format(
            super().to_printstr(game), game.num_profiles,
            game.num_all_profiles)

    def subserial(self, subgame_mask):
        """Restrict possible strategies"""
        subgame_mask = np.asarray(subgame_mask, bool)
        assert self.is_subgame(subgame_mask), \
            "subgame_mask must be valid"
        strat_names = [[s for s, m in zip(strats, mask) if m]
                       for strats, mask
                       in zip(self.strat_names,
                              np.split(subgame_mask, self.role_starts[1:]))]
        return gameserializer(self.role_names, strat_names)


def gameserializer(role_names, strat_names):
    """Static constrictor for GameSerializer

    Parameters
    ----------
    role_names : [str]
        The name of each role.
    strat_names : [[str]]
        The name of each strategy for each role.
    """
    return GameSerializer(tuple(role_names), tuple(map(tuple, strat_names)))


def gameserializer_json(json):
    """Read a GameSerializer from json

    Parameters
    ----------
    json : json
        A json representation of a basic game with names. Must either be
        {roles: [{name: <role>, strategies: [<strat>]}]}, or {strategies:
        {<role>: [<strat>]}}.
    """
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
    return gameserializer(role_names, strat_names)


def gameserializer_copy(serial):
    """Copy a serializer into a GameSerializer"""
    return GameSerializer(serial.role_names, serial.strat_names)


def read_game(json):
    """Read a Game and its serializer from json"""
    serial = gameserializer_json(json)
    return serial.from_json(json), serial


class SampleGameSerializer(_BaseSerializer):
    def __init__(self, role_names, strat_names):
        super().__init__(role_names, strat_names)

    def from_json(self, game):
        """Read a SampleGame from json"""
        base = super().from_json(game)
        profile_list = game.get('profiles', ())

        sample_map = {}
        for profile in profile_list:
            prof, spay = self.from_profsamplepay_json(profile)
            num_samps = spay.shape[0]
            profls, payls = sample_map.setdefault(num_samps, ([], []))
            profls.append(prof[None])
            payls.append(spay.T[None])

        if sample_map:
            values = [v for _, v in sorted(sample_map.items())]
            profiles = np.concatenate(list(itertools.chain.from_iterable(
                prof for prof, _ in values)))
            sample_payoffs = [np.concatenate(spay) for _, spay in values]
            return rsgame.samplegame_replace(base, profiles, sample_payoffs)
        else:  # No data
            return rsgame.samplegame_copy(base)

    def to_json(self, game):
        """Fromat a SampleGame as json"""
        game = rsgame.samplegame_copy(game)  # relatively free
        res = super().to_json(game)
        res['profiles'] = [
            self.to_profsamplepay_json(pay.T, prof) for prof, pay
            in zip(game.profiles,
                   itertools.chain.from_iterable(game.sample_payoffs))]
        res['type'] = 'samplegame.1'
        return res

    def to_printstr(self, game):
        """Format a SampleGame as a printable string"""
        game = rsgame.samplegame_copy(game)  # relatively free
        if game.num_samples.size == 0:
            sampstr = 'no observations'
        elif game.num_samples.size == 1:
            samps = game.num_samples[0]
            sampstr = '{:d} observation{} per profile'.format(
                samps, '' if samps == 1 else 's')
        else:
            sampstr = '{:d} to {:d} observations per profile'.format(
                game.num_samples.min(), game.num_samples.max())
        return '{}\npayoff data for {:d} out of {:d} profiles\n{}'.format(
            super().to_printstr(game),
            game.num_profiles,
            game.num_all_profiles,
            sampstr,
        )

    def subserial(self, subgame_mask):
        """Restrict possible strategies"""
        subgame_mask = np.asarray(subgame_mask, bool)
        assert self.is_subgame(subgame_mask), \
            "subgame_mask must be valid"
        strat_names = [[s for s, m in zip(strats, mask) if m]
                       for strats, mask
                       in zip(self.strat_names,
                              np.split(subgame_mask, self.role_starts[1:]))]
        return samplegameserializer(self.role_names, strat_names)


def samplegameserializer(role_names, strat_names):
    """Static constrictor for GameSerializer

    Parameters
    ----------
    role_names : [str]
        The name of each role.
    strat_names : [[str]]
        The name of each strategy for each role.
    """
    return SampleGameSerializer(tuple(role_names),
                                tuple(map(tuple, strat_names)))


def samplegameserializer_json(json):
    """Read a GameSerializer from json

    Parameters
    ----------
    json : json
        A json representation of a basic game with names. Must either be
        {roles: [{name: <role>, strategies: [<strat>]}]}, or {strategies:
        {<role>: [<strat>]}}."""
    return samplegameserializer_copy(gameserializer_json(json))


def samplegameserializer_copy(serial):
    """Copy a SampleGameSerializer from a serialzier"""
    return SampleGameSerializer(serial.role_names, serial.strat_names)


def read_samplegame(json):
    """Read a SampleGame and its associated serialzier from json"""
    serial = samplegameserializer_json(json)
    return serial.from_json(json), serial


# Convenient unpacking of dictionaries
_nan = float('nan')


def _unpack_symgrp(role, strategy, count, payoff=None, id=None, **_):
    return id, role, strategy, count, payoff


def _unpack_obs(id, payoff, **_):
    return id, payoff


def _unpack_player(sid, p, **_):
    return sid, p


def _unpack_obs_player(role, strategy, payoff, **_):
    return role, strategy, payoff


def _mean(vals):
    if isinstance(vals, abc.Iterable):
        count = 0
        mean = 0
        for v in vals:
            count += 1
            mean += (v - mean) / count
        return mean if count > 0 else _nan
    else:
        return vals
