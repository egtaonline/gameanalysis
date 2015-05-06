import sys
import argparse
import json

from functools import partial

from HashableClasses import harray
from BasicFunctions import one_line
from RoleSymmetricGame import (Game, PayoffData, Profile, SampleGame)


def parse_json(data):
    '''Convert loaded json data (list or dict) into GameAnalysis classes.'''
    if isinstance(data, list):
        return map(parse_json, data)
    elif isinstance(data, dict):
        if 'object' in data:  # game downloaded from EGATS
            return parse_json(json.loads(data['object']))
        elif 'profiles' in data:
            return parse_game_json(data)
        elif 'symmetry_groups' in data:
            return read_v3_profile(data)
        elif 'observations' in data:
            if 'players' in data['observations']['symmetry_groups']:
                raise NotImplementedError
            else:
                return read_v3_samples_profile(data)
        elif 'sample_count' in data:
            return read_v2_profile(data)
        elif 'type' in data and data['type'] == 'GA_Profile':
            return read_GA_profile(data)
        else:
            return {k: parse_json(v) for k, v in data.items()}
    else:
        return data


def parse_game_json(gameJSON):
    try:
        if 'players' in gameJSON and 'strategies' in gameJSON:
            return read_GA_game(gameJSON)
        elif len(gameJSON['profiles']) == 0:
            return Game(*parse_roles(gameJSON['roles']))
        elif 'symmetry_groups' in gameJSON['profiles'][0] and \
             'observations' in gameJSON['profiles'][0]:
            return read_game_JSON_v4(gameJSON)
        elif 'symmetry_groups' in gameJSON['profiles'][0]:
            return read_game_JSON_v3(gameJSON)
        elif 'observations' in gameJSON['profiles'][0]:
            if 'players' in gameJSON['profiles'][0]['observations'][0]['symmetry_groups'][0]:
                raise NotImplementedError
            else:
                return read_game_JSON_v3_samples(gameJSON)
        elif 'strategy_array' in gameJSON['roles'][0]:
            return read_game_JSON_old(gameJSON)
        elif 'strategies' in gameJSON['roles'][0]:
            return read_game_JSON_v2(gameJSON)
        else:
            raise IOError(one_line('invalid game JSON: ' + str(gameJSON), 71))
    except TypeError:
        raise TypeError('Null payoff encountered')


def read_GA_game(gameJSON):
    if len(gameJSON['profiles']) > 0 and isinstance(gameJSON['profiles'][0]
                                                    .values()[0][0][2], list):
        game_type = SampleGame
    else:
        game_type = Game
    return game_type(gameJSON['players'].keys(), gameJSON['players'],
                     gameJSON['strategies'],
                     [{r: [PayoffData(*scv) for scv in p[r]]for r in p}
                      for p in gameJSON['profiles']])


def read_game_JSON_new(gameJSON, game_type, profile_reader):
    roles, players, strategies = parse_roles(gameJSON['roles'])
    profiles = []
    if 'profiles' in gameJSON:
        for profileJSON in gameJSON['profiles']:
            profiles.append(profile_reader(profileJSON))
    return game_type(roles, players, strategies, profiles)


def parse_roles(rolesJSON):
    players = {r['name']: int(r['count']) for r in rolesJSON}
    strategies = {r['name']: r['strategies'] for r in rolesJSON}
    roles = list(players.keys())
    return roles, players, strategies


def read_game_JSON_old(json_data):
    players = {r['name']: int(r['count']) for r in json_data['roles']}
    strategies = {r['name']: r['strategy_array'] for r in json_data['roles']}
    roles = list(players.keys())
    profiles = []
    for profileDict in json_data['profiles']:
        profile = {r: [] for r in roles}
        prof_strat = {}
        for role_str in profileDict['proto_string'].split('; '):
            role, strategy_str = role_str.split(': ')
            prof_strat[role] = strategy_str.split(', ')
        for roleDict in profileDict['roles']:
            role = roleDict['name']
            for strategyDict in roleDict['strategies']:
                s = strategyDict['name']
                profile[role].append(PayoffData(str(s),
                                                int(prof_strat[role].count(s)),
                                                float(strategyDict['payoff'])))
        profiles.append(profile)
    return Game(roles, players, strategies, profiles)


def read_v2_profile(profileJSON):
    profile = {r['name']: [] for r in profileJSON['roles']}
    for roleDict in profileJSON['roles']:
        role = roleDict['name']
        for strategyDict in roleDict['strategies']:
            profile[role].append(PayoffData(str(strategyDict['name']),
                                            int(strategyDict['count']),
                                            float(strategyDict['payoff'])))
    return profile


def read_v3_profile(profileJSON):
    prof = {}
    for sym_grp in profileJSON['symmetry_groups']:
        if sym_grp['role'] not in prof:
            prof[sym_grp['role']] = []
        prof[sym_grp['role']].append(PayoffData(sym_grp['strategy'],
                                                sym_grp['count'],
                                                sym_grp['payoff']))
    return prof


def read_v3_samples_profile(profileJSON):
    prof = {}
    for obs in profileJSON['observations']:
        for sym_grp in obs['symmetry_groups']:
            role = sym_grp['role']
            if role not in prof:
                prof[role] = {}
            strat = sym_grp['strategy']
            count = sym_grp['count']
            value = sym_grp['payoff']
            if (strat, count) not in prof[role]:
                prof[role][(strat, count)] = []
            prof[role][(strat, count)].append(value)
    return {r: [PayoffData(sc[0], sc[1], v) for sc, v in prof[r].items()]
            for r in prof}


def read_v4_samples_profile(profileJSON):
    prof = {}
    grp_ids = {s['id']: s for s in profileJSON['symmetry_groups']}
    for obs in profileJSON['observations']:
        for s in obs['symmetry_groups']:
            sym_grp = grp_ids[s['id']]
            role = sym_grp['role']
            if role not in prof:
                prof[role] = {}
            strat = sym_grp['strategy']
            count = sym_grp['count']
            value = s['payoff']
            if (strat, count) not in prof[role]:
                prof[role][(strat, count)] = []
            prof[role][(strat, count)].append(value)
    return {r: [PayoffData(sc[0], sc[1], v) for sc, v in prof[r].items()]
            for r in prof}


def read_GA_profile(profileJSON):
    try:
        return Profile(profileJSON['data'])
    except KeyError:
        return Profile(profileJSON)


read_game_JSON_v4 = partial(read_game_JSON_new, game_type=SampleGame,
                            profile_reader=read_v4_samples_profile)
read_game_JSON_v3 = partial(read_game_JSON_new, game_type=Game,
                            profile_reader=read_v3_profile)
read_game_JSON_v3_samples = partial(read_game_JSON_new, game_type=SampleGame,
                                    profile_reader=read_v3_samples_profile)
read_game_JSON_v2 = partial(read_game_JSON_new, game_type=Game,
                            profile_reader=read_v2_profile)


def read_NE(data):
    prob_strs = data.split(',')[1:]
    probs = []
    for s in prob_strs:
        try:
            num, denom = s.split('/')
            probs.append(float(num) / float(denom))
        except ValueError:
            probs.append(float(s))
    return harray(probs)


def to_JSON_str(obj, indent=2):
    return json.dumps(to_JSON_obj(obj), sort_keys=True, indent=indent)


def to_JSON_obj(obj):
    if hasattr(obj, 'toJSON'):
        return obj.toJSON()
    if hasattr(obj, 'items'):
        if all([hasattr(k, 'toJSON') for k in obj.keys()]):
            return {to_JSON_obj(k): to_JSON_obj(v) for k, v in obj.items()}
        return {k: to_JSON_obj(v) for k, v in obj.items()}
    if hasattr(obj, '__iter__'):
        return map(to_JSON_obj, obj)
    return json.loads(json.dumps(obj))


def to_nfg_asym(game):
    output = 'NFG 1 R "asymmetric"\n{ '
    output += ' '.join(('"' + str(r) + '"' for r in game.roles)) + ' } { '
    output += ' '.join(map(str, game.numStrategies)) + ' }\n\n'
    prof = Profile({r: {game.strategies[r][0]: 1} for r in game.roles})
    last_prof = Profile({r: {game.strategies[r][-1]: 1} for r in game.roles})
    while prof != last_prof:
        # prof_strat = {r: prof[r].keys()[0] for r in game.roles}
        output += nfg_payoffs(game, prof) + ' '
        prof = increment_profile(game, prof)
    output += nfg_payoffs(game, last_prof) + '\n'
    return output


def nfg_payoffs(game, prof):
    return ' '.join((str(game.getPayoff(prof, r, prof[r].keys()[0]))
                     for r in game.roles))


def increment_profile(game, prof):
    for role in game.roles:
        strat = prof[role].keys()[0]
        i = game.index(role, strat)
        if i < game.numStrategies[game.index(role)] - 1:
            return prof.deviate(role, strat, game.strategies[role][i+1])
        prof = prof.deviate(role, strat, game.strategies[role][0])
    return prof


PARSER = argparse.ArgumentParser('Converts between game data formats')
PARSER.add_argument('--input', '-i', type=argparse.FileType('r'),
                    default=sys.stdin,
                    help='Input file. Defaults to stdin.')
PARSER.add_argument('--output', '-o', type=argparse.FileType('w'),
                    default=sys.stdout,
                    help='Output file. Defaults to stdout.')
PARSER.add_argument('--format', '-f', choices=('json', 'nfg'),
                    default='json', help='Output format.')


def main():
    args = PARSER.parse_args()
    if args.format == 'json':
        print to_JSON_str(args.input)
    elif args.format == 'nfg':
        print to_nfg_asym(args.input)


if __name__ == '__main__':
    main()
