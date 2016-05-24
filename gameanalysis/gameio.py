"""Utility module that contains code for parsing legacy game formats"""
import collections
from collections import abc

from gameanalysis import utils


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
            profiles)


def _roles_from_json(json_):
    """Load json that has a roles field instead"""
    roles = json_['roles']
    players = {r['name']: int(r['count']) for r in roles}
    strategies = {r['name']: r['strategies'] for r in roles}
    return (players, strategies, ())


def _new_game_from_json(json_, profile_reader):
    """Interprets a new style game"""
    players, strategies, _ = _roles_from_json(json_)
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


# def read_GA_profile(profileJSON):
#     try:
#         return Profile(profileJSON['data'])
#     except KeyError:
#         return Profile(profileJSON)


# def read_NE(data):
#     prob_strs = data.split(',')[1:]
#     probs = []
#     for s in prob_strs:
#         try:
#             num, denom = s.split('/')
#             probs.append(float(num) / float(denom))
#         except ValueError:
#             probs.append(float(s))
#     return harray(probs)


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


# def _increment_profile(game, prof):
#     for role in game.roles:
#         strat = prof[role].keys()[0]
#         i = game.index(role, strat)
#         if i < game.numStrategies[game.index(role)] - 1:
#             return prof.deviate(role, strat, game.strategies[role][i+1])
#         prof = prof.deviate(role, strat, game.strategies[role][0])
#     return prof
