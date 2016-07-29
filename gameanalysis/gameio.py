"""Utility module that contains code for parsing legacy game formats"""
import collections
import itertools
import warnings
from collections import abc

import numpy as np

from gameanalysis import rsgame
from gameanalysis import utils


class GameSerializer(object):
    """An object with utilities for serializing a game with names

    Parameters 1
    ------------
    strategies : {role: [strategy]}
        A dictionary mapping role to strategies. The resulting serializer is
        the sorted version of all inputs.

    Parameters 2
    ------------
    role : [role]
        A list of ordered roles.
    strategies : [[strategies]]
        A list of lists of ordered strategies for each role.
    """
    def __init__(self, *args):
        if len(args) == 1:
            role_strats = sorted(args[0].items())
            self.role_names = tuple(r for r, _ in role_strats)
            self.strat_names = tuple(tuple(sorted(s)) for _, s in role_strats)
        elif len(args) == 2:
            self.role_names = tuple(args[0])
            self.strat_names = tuple(map(tuple, args[1]))
            if any(any(a > b for a, b in zip(s[:-1], s[1:]))
                   for s in self.strat_names):
                warnings.warn("If strategies aren't sorted, some functions "
                              "won't work as intended")
        self.num_strategies = np.fromiter(map(len, self.strat_names), int,
                                          len(self.role_names))
        self.num_roles = self.num_strategies.size
        self.num_role_strats = self.num_strategies.sum()
        self.role_starts = self.num_strategies[:-1].cumsum()
        self._role_index = {r: i for i, r in enumerate(self.role_names)}
        role_strats = itertools.chain.from_iterable(
            ((r, s) for s in strats) for r, strats
            in zip(self.role_names, self.strat_names))
        self._role_strat_index = {(r, s): i for i, (r, s)
                                  in enumerate(role_strats)}
        self._hash = hash((self.role_names, self.strat_names))

    def role_split(self, array, axis=-1):
        return np.split(array, self.role_starts, axis)

    def role_index(self, role):
        """Return the index of a role"""
        return self._role_index[role]

    def strat_name(self, role_strat_index):
        """Get the strategy name from a full index"""
        role_index = np.searchsorted(self.role_starts, role_strat_index,
                                     'right') - 1
        return self.strat_names[role_index][role_strat_index -
                                            self.role_starts[role_index]]

    def role_strat_index(self, role, strat):
        """Return the index of a role strat pair"""
        return self._role_strat_index[role, strat]

    def to_prof_json(self, prof, filter_zeros=True):
        """Convert a profile to json"""
        return {role: {strat: count.item() for strat, count
                       in zip(strats, counts)
                       if not filter_zeros or count > 0}
                for counts, role, strats
                in zip(self.role_split(prof),
                       self.role_names, self.strat_names)}

    def to_prof_symgrp(self, prof):
        """Convert a profile to a symmetry group"""
        return list(itertools.chain.from_iterable(
            (
                {'role': role, 'strategy': strat, 'count': int(count)}
                for strat, count in zip(strats, counts))
            for role, strats, counts
            in zip(self.role_names, self.strat_names,
                   self.role_split(prof))))

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
        if np.issubdtype(prof.dtype, int):
            format_strat = lambda s, p: '\t{}: {:d}\n'.format(s, p)
        elif np.issubdtype(prof.dtype, float):
            format_strat = lambda s, p: '\t{}: {:>7.2%}\n'.format(s, p)
        else:  # boolean
            format_strat = lambda s, p: '\t{}\n'.format(s)

        return ''.join(
            '{}:\n{}'.format(role, ''.join(format_strat(s, p)
                                           for p, s in zip(probs, strats)
                                           if p > 0))
            for probs, role, strats
            in zip(self.role_split(prof), self.role_names, self.strat_names)
            if probs.any()
        ).expandtabs(4)

    def from_prof(self, prof):
        """Read a profile from an auto-detected format"""
        if isinstance(prof, str):
            return self.from_prof_string(prof)
        elif isinstance(prof, abc.Mapping):
            return self.from_prof_json(prof)
        elif isinstance(prof, abc.Iterable):
            return self.from_prof_symgrp(prof)
        else:
            raise ValueError('Unrecognized auto style for input: {}'
                             .format(prof))

    def from_prof_json(self, dictionary):
        """Read a profile from json"""
        prof = [False] * self.num_role_strats
        for role, strats in dictionary.items():
            for strat, count in strats.items():
                prof[self._role_strat_index[role, strat]] = count
        return np.array(prof)

    def from_prof_symgrp(self, symgrps):
        """Read a profile from symmetry groups"""
        prof = np.zeros(self.num_role_strats, int)
        for sym_group in symgrps:
            role = sym_group['role']
            strat = sym_group['strategy']
            count = sym_group['count']
            prof[self._role_strat_index[role, strat]] = count
        return prof

    def from_payoff_symgrp(self, symgrps):
        """Read a set of payoffs from symmetry groups"""
        payoffs = np.zeros(self.num_role_strats)
        for sym_group in symgrps:
            role = sym_group['role']
            strat = sym_group['strategy']
            payoff = sym_group['payoff']
            payoffs[self._role_strat_index[role, strat]] = payoff
        return payoffs

    def from_prof_string(self, prof_string):
        """Read a profile from a string"""
        prof = np.zeros(self.num_role_strats, int)
        for role_str in prof_string.split('; '):
            role, strats = role_str.split(': ', 1)
            for strat_str in strats.split(', '):
                count, strat = strat_str.split(' ', 1)
                prof[self._role_strat_index[role, strat]] = count
        return prof

    def to_role_json(self, role_info):
        """Format role data as json"""
        return {role: info.item() for role, info
                in zip(self.role_names, role_info)}

    def from_role_json(self, role_json):
        prof = [False] * self.num_roles
        for role, count in role_json.items():
            prof[self._role_index[role]] = count
        return np.array(prof)

    def to_payoff_json(self, profile, payoffs):
        """Format a profile and payoffs as json"""
        return {role: {strat: pay for strat, pay, count
                       in zip(strats, pays, counts) if count > 0}
                for role, strats, pays, counts
                in zip(self.role_names, self.strat_names,
                       self.role_split(payoffs),
                       self.role_split(profile))}

    def to_deviation_payoff_json(self, profile, payoffs):
        """Format a profile and deviation payoffs as json"""
        supp = profile > 0
        role_supp = np.add.reduceat(supp, np.insert(self.role_starts, 0, 0))
        splits = np.repeat(self.num_strategies - 1, role_supp)[:-1].cumsum()
        return {r: {s: {d: p.item() for p, d
                        in zip(dps, (d for d in ses if d != s))}  # noqa
                    for dps, s in zip(np.split(ps, sp.sum()),
                                      (s for s, m in zip(ses, sp) if m))}  # noqa
                for r, ses, ps, sp in zip(self.role_names, self.strat_names,
                                          np.split(payoffs, splits),
                                          self.role_split(supp))}

    def to_json(self, game):
        """Convert to a json serializable format"""
        assert np.all(self.num_strategies == game.num_strategies), \
            "Number of strategies per role didn't match"
        json = {'players': dict(zip(self.role_names,
                                    map(int, game.num_players))),
                'strategies': dict(zip(self.role_names, self.strat_names))}

        if isinstance(game, rsgame.SampleGame):
            json['profiles'] = [
                {
                    role: [(strat, int(count), list(map(float, pay)))
                           for strat, count, pay
                           in zip(strats, counts, pays)
                           if count > 0]
                    for counts, pays, role, strats
                    in zip(self.role_split(prof),
                           self.role_split(payoffs, 0),
                           self.role_names,
                           self.strat_names)}
                for prof, payoffs
                in zip(game.profiles,
                       itertools.chain.from_iterable(game.sample_payoffs))]

        elif isinstance(game, rsgame.Game):
            json['profiles'] = [
                {
                    role: [(strat, int(count), float(pay))
                           for strat, count, pay
                           in zip(strats, counts, pays)
                           if count > 0]
                    for counts, pays, role, strats
                    in zip(self.role_split(prof),
                           self.role_split(payoffs),
                           self.role_names,
                           self.strat_names)}
                for prof, payoffs in zip(game.profiles, game.payoffs)]

        return json

    def to_str(self, game):
        strg = ('{}:\n\tRoles: {}\n\tPlayers:\n\t\t{}\n\tStrategies:\n\t\t{}\n'
                .format(
                    game.__class__.__name__,
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
        if isinstance(game, rsgame.Game):
            strg += ('payoff data for {data:d} out of {total:d} '
                     'profiles').format(data=game.num_profiles,
                                        total=game.num_all_profiles)
        if isinstance(game, rsgame.SampleGame):
            samples = game.num_samples
            if samples.size == 0:
                sample_str = '\nno observations'
            elif samples.size == 1:
                sample_str = '\n{:d} observations per profile'.format(
                    samples[0])
            else:
                sample_str = '\n{:d} to {:d} observations per profile'.format(
                    samples.min(), samples.max())
            strg += sample_str

        return strg

    def __hash__(self):
        return self._hash


def read_base_game(json):
    players, strats, _, _ = _game_from_json(json)
    conv = GameSerializer(strats)
    num_players = np.fromiter((players[r] for r in conv.role_names), int,
                              conv.num_roles)
    return rsgame.BaseGame(num_players, conv.num_strategies), conv


def read_game(json):
    """Constructor for Game"""
    # From json
    players, strats, payoff_data, num_profs = _game_from_json(json)
    conv = GameSerializer(strats)
    num_players = np.fromiter((players[r] for r in conv.role_names), int,
                              conv.num_roles)

    profiles = np.zeros((num_profs, conv.num_role_strats), int)
    payoffs = np.zeros((num_profs, conv.num_role_strats), float)

    p = 0  # profile index
    for profile_data in payoff_data:
        if any(any(len(p[2]) == 0 or any(x is None for x in p[2])
                   for p in sym_grp) for sym_grp in profile_data.values()):
            warnings.warn('Encountered null payoff data in profile: {}'
                          .format(profile_data))
            continue  # Invalid data, but can continue

        for role, strategy_data in profile_data.items():
            for strategy, count, pays in strategy_data:
                i = conv.role_strat_index(role, strategy)
                assert profiles[p, i] == 0, (
                    'Duplicate role strategy pair ({}, {})'
                    .format(role, strategy))
                profiles[p, i] = count
                payoffs[p, i] = np.average(pays)

        p += 1  # profile added

    # The slice at the end truncates any null data
    return (rsgame.Game(num_players, conv.num_strategies, profiles, payoffs),
            conv)


def read_sample_game(json):
    # From json
    players, strats, payoff_data, num_profs = _game_from_json(json)
    conv = GameSerializer(strats)
    num_players = np.fromiter((players[r] for r in conv.role_names), int,
                              conv.num_roles)

    sample_map = {}
    for profile_data in payoff_data:
        if any(any(p is None or len(p) == 0 for _, __, p in sym_grps)
               for sym_grps in profile_data.values()):
            warnings.warn('Encountered null payoff data in profile: {}'
                          .format(profile_data))
            continue  # Invalid data, but can continue

        num_samples = min(min(len(payoffs[2]) for payoffs in sym_grps)
                          for sym_grps in profile_data.values())
        profile = np.zeros(conv.num_role_strats, dtype=int)
        spayoffs = np.zeros((conv.num_role_strats, num_samples))

        for role, strategy_data in profile_data.items():
            for strategy, count, payoffs in strategy_data:
                i = conv.role_strat_index(role, strategy)
                assert profile[i] == 0, (
                    'Duplicate role strategy pair ({}, {})'
                    .format(role, strategy))
                if len(payoffs) > num_samples:
                    warnings.warn("Truncating observation data")

                profile[i] = count
                spayoffs[i] = payoffs[:num_samples]

        lst_profs, lst_pays = sample_map.setdefault(num_samples, ([], []))
        lst_profs.append(profile[None])
        lst_pays.append(spayoffs[None])

    # Join data together
    profs = []
    for prof, _ in sample_map.values():
        profs.extend(profs)
    if profs:
        profiles = np.concatenate(profs)
        sample_payoffs = [np.concatenate(p) for _, p
                          in sample_map.values() if p]
    else:  # No data
        profiles = np.empty((0, conv.num_role_strats), dtype=int)
        sample_payoffs = []

    # The slice at the end truncates any null data
    return (rsgame.SampleGame(num_players, conv.num_strategies, profiles,
                              sample_payoffs), conv)


def _game_from_json(json_):
    """Returns constructor arguments for a game from parsed json"""
    if 'players' in json_ and 'strategies' in json_:
        return _ga_game_from_json(json_)
    elif not json_['profiles']:
        return _roles_from_json(json_)
    elif ('symmetry_groups' in json_['profiles'][0] and
          'observations' in json_['profiles'][0]):
        return _new_game_from_json(json_, _samples_profile_v4_from_json)
    elif 'symmetry_groups' in json_['profiles'][0]:
        return _new_game_from_json(json_, _profile_v3_from_json)
    elif 'observations' in json_['profiles'][0]:
        return _new_game_from_json(json_, _samples_profile_v3_from_json)
    elif 'strategy_array' in json_['roles'][0]:
        return _old_game_from_json(json_)
    elif 'strategies' in json_['roles'][0]:
        return _new_game_from_json(json_, _profile_v2_from_json)
    else:
        raise IOError(utils.one_line('invalid game JSON: {}'.format(json_),
                                     71))


def _ga_game_from_json(json_):
    """Returns parameters necessary for constructing a game analysis game when
    starting is proper output

    """
    profiles = json_.get('profiles', ())

    # Fix shorthand payoffs
    for profile in profiles:
        for role, sym_grps in profile.items():
            for sym_grp in sym_grps:
                if not isinstance(sym_grp[2], abc.Sized):
                    sym_grp[2] = [sym_grp[2]]

    return (json_['players'],
            json_['strategies'],
            profiles,
            len(profiles))


def _roles_from_json(json_):
    """Load json that has a roles field instead"""
    roles = json_['roles']
    players = {r['name']: int(r['count']) for r in roles}
    strategies = {r['name']: r['strategies'] for r in roles}
    return (players, strategies, (), 0)


def _new_game_from_json(json_, profile_reader):
    """Interprets a new style game"""
    players, strategies, _, _ = _roles_from_json(json_)
    return (players,
            strategies,
            (profile_reader(prof) for prof in json_['profiles']),
            len(json_['profiles']))


def _old_game_from_json(json_):
    players = {r['name']: int(r['count']) for r in json_['roles']}
    strategies = {r['name']: r['strategy_array'] for r in json_['roles']}
    roles = list(players.keys())

    def profiles():
        for prof_dict in json_['profiles']:
            profile = {r: [] for r in roles}
            counts = {}
            for role_str in prof_dict['proto_string'].split('; '):
                role, strategy_str = role_str.split(': ')
                counts[role] = collections.Counter(strategy_str.split(', '))
            for role_dict in prof_dict['roles']:
                role = role_dict['name']
                role_counts = counts[role]
                for strat_dict in role_dict['strategies']:
                    strat = strat_dict['name']
                    profile[role].append((strat,
                                          role_counts[strat],
                                          [float(strat_dict['payoff'])]))
            yield profile
    return (players, strategies, profiles(), len(json_['profiles']))


def _profile_v2_from_json(prof_json):
    """Interprets a version 2 profile"""
    profile = {}
    for role_dict in prof_json['roles']:
        role = role_dict['name']
        profile_data = []
        for strat_dict in role_dict['strategies']:
            profile_data.append((strat_dict['name'],
                                 int(strat_dict['count']),
                                 [float(strat_dict['payoff'])]))
        profile[role] = profile_data
    return profile


def _profile_v3_from_json(prof_json):
    """Interprets a version 3 profile"""
    prof = {}
    for sym_grp in prof_json['symmetry_groups']:
        strat_data = prof.setdefault(sym_grp['role'], [])
        payoff = sym_grp['payoff']
        if not isinstance(payoff, abc.Sized):
            payoff = [payoff]
        strat_data.append((sym_grp['strategy'],
                           sym_grp['count'],
                           payoff))
    return prof


def _samples_profile_v3_from_json(prof_json):
    """Interprets a version 3 profile with sample data"""
    prof = {}
    for obs in prof_json['observations']:
        for sym_grp in obs['symmetry_groups']:
            role = sym_grp['role']
            strat = sym_grp['strategy']
            count = sym_grp['count']
            payoff = sym_grp['payoff']

            strat_counts = prof.setdefault(role, {})
            payoffs = strat_counts.setdefault((strat, count), [])
            payoffs.append(payoff)
    return {role: {(strat, count, payoff) for (strat, count), payoff
                   in strats.items()}
            for role, strats in prof.items()}


def _samples_profile_v4_from_json(prof_json):
    """Interprets a version 4 sample profile"""
    prof = {}
    grp_ids = {sg['id']: sg for sg in prof_json['symmetry_groups']}
    for obs in prof_json['observations']:
        for sg in obs['symmetry_groups']:
            sym_grp = grp_ids[sg['id']]
            role = sym_grp['role']
            strat = sym_grp['strategy']
            count = sym_grp['count']
            payoff = sg['payoff']

            strat_counts = prof.setdefault(role, {})
            payoffs = strat_counts.setdefault((strat, count), [])
            payoffs.append(payoff)

    return {role: [(strat, count, payoff) for (strat, count), payoff
                   in strats.items()]
            for role, strats in prof.items()}


# def to_nfg_asym(game, output):
#     output.write('NFG 1 R "asymmetric"\n{ ')
#     output.write(' '.join(('"' + str(r) + '"' for r in game.roles)))
#     output.write(' } { ')
#     output.write(' '.join(map(str, game.numStrategies)))
#     output.write(' }\n\n')
#     prof = rsgame.PureProfile({role: {next(iter(strats)): 1}
#                                for role, strats in game.strategies.items()})
#     last_prof = Profile({r: {game.strategies[r][-1]: 1} for r in game.roles})
#     while prof != last_prof:
#         # prof_strat = {r: prof[r].keys()[0] for r in game.roles}
#         output += _nfg_payoffs(game, prof) + ' '
#         prof = _increment_profile(game, prof)
#     output += _nfg_payoffs(game, last_prof) + '\n'
#     return output


# def _nfg_payoffs(game, prof):
#     return ' '.join(str(game.get_payoff(prof, role, next(iter(strats.keys())))) # noqa
#                     for role, strats in prof.items())
