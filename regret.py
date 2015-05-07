#!/usr/bin/env python3
import itertools
import argparse
import numpy as np

import funcs
import rsgame


def regret(game, prof, role=None, strategy=None, deviation=None, bound=True):
    if is_pure_profile(prof):
        return profile_regret(game, prof, role, strategy, deviation, bound)
    elif is_mixture_array(prof):
        return mixture_regret(game, prof, role, deviation, bound)
    elif is_mixed_profile(prof):
        return mixture_regret(game, game.toArray(prof), role, deviation, bound)
    elif is_profile_array(prof):
        return profile_regret(game, game.toProfile(prof), role, strategy, \
                deviation, bound)
    raise TypeError(one_line('unrecognized profile type: ' + str(prof), 69))


def pure_strategy_regret(game, prof, role=None, strategy=None, deviation=None):
    '''Returns the regret of a pure strategy profile in a game.

    Role, strategy, and deviation may be specified to limit the scope of the
    regret. Regret of a role is undefined if it has only one strategy. If data
    for a deviation profile doesn't exist, this will return infinity.

    '''
    if role is None:
        return max(pure_strategy_regret(game, prof, r, strategy, deviation)
                   for r, ses in game.strategies.items()
                   if len(ses) > 1)
    elif strategy is None:
        return max(pure_strategy_regret(game, prof, role, s, deviation)
                   for s in prof[role])
    elif deviation is None:
        if len(game.strategies[role]) == 1:
            raise ValueError(
                'Can\'t calculate regret of a role with only one strategy')
        return max(pure_strategy_regret(game, prof, role, strategy, d)
                   for d in set(game.strategies[role]) - {strategy})
    else:
        dev_prof = prof.deviate(role, strategy, deviation)
        if dev_prof in game:
            return (game.getPayoff(dev_prof, role, deviation) -
                    game.getPayoff(prof, role, strategy))
        else:
            return np.inf


def mixture_regret(game, mix, role=None, deviation=None):
    '''Return the regret of a mixture profile

    Role and deviation may be specified to limit the scope of the regret. If
    data for a deviation doesn't exist, the regret is infinite.

    '''
    if role is None:
        return max(mixture_regret(game, mix, r, deviation)
                   for r, ses in game.strategies.items()
                   if len(ses) > 1)
    elif deviation is None:
        return max(mixture_regret(game, mix, role, d)
                   for d in game.strategies[role])
    elif any(p not in game for p
             in mixture_neighbors(game, mix, role, deviation)):
        return np.inf
    else:
        strategy_EVs = game.expectedValues(mix)
        role_EVs = (strategy_EVs * mix).sum(1)
        r = game.index(role)
        d = game.index(role, deviation)
        return float(strategy_EVs[r,d] - role_EVs[r])

def neighbors(game, p, *args, **kwargs):
    if isinstance(p, Profile):
        return profile_neighbors(game, p, *args, **kwargs)
    elif isinstance(p, np.ndarray):
        return mixture_neighbors(game, p, *args, **kwargs)
    raise TypeError('unrecognized argument type: ' + type(p).__name__)


def profile_neighbors(game, profile, role=None, strategy=None,
                      deviation=None):
    if role is None:
        return list(chain(*[profile_neighbors(game, profile, r, strategy, \
                deviation) for r in game.roles]))
    if strategy is None:
        return list(chain(*[profile_neighbors(game, profile, role, s, \
                deviation) for s in profile[role]]))
    if deviation is None:
        return list(chain(*[profile_neighbors(game, profile, role, strategy, \
                d) for d in set(game.strategies[role]) - {strategy}]))
    return [profile.deviate(role, strategy, deviation)]


def mixture_neighbors(game, mix, role=None, deviation=None):
    n = set()
    for profile in feasible_profiles(game, mix):
        n.update(profile_neighbors(game, profile, role, deviation=deviation))
    return n


def feasible_profiles(game, mix, thresh=1e-3):
    return [Profile({r:{s:p[game.index(r)].count(s) for s in set(p[ \
            game.index(r)])} for r in game.roles}) for p in product(*[ \
            CwR(filter(lambda s: mix[game.index(r), game.index(r,s)] >= \
            thresh, game.strategies[r]), game.players[r]) for r \
            in game.roles])]


def symmetric_profile_regrets(game):
    assert game.is_symmetric(), 'Game must be symmetric'
    role = next(iter(game.strategies))
    return {s: regret(game, rsgame.Profile({role:{s:game.players[role]}})) for s \
            in game.strategies[role]}


def equilibrium_regrets(game, eq):
    '''NE regret for all roles and pure strategies.'''
    if is_mixed_profile(eq):
        eq = game.toArray(eq)
    if is_mixture_array(eq):
        return game.getExpectedPayoff(eq).reshape(eq.shape[0],1) - \
                game.expectedValues(eq)
    regrets = {}
    for role in game.roles:
        regrets[role] = {}
        for strategy in game.strategies[role]:
            regrets[role][strategy] = -regret(game, eq, deviation=strategy)
    return regrets


def equilibrium_regret(game, eq, role, mix):
    '''
    NE regret for a specific role and mixed strategy.
    '''
    regrets = equilibrium_regrets(game, eq)[game.index(role)]
    reg_arr = [regrets[game.index(role, s)] for s in game.strategies[role]]
    if isinstance(mix, dict):
        mix = np.array([mix[s] if s in mix else 0 for s in \
                game.strategies[role]])
    return (mix * reg_arr).sum()


def safety_value(game, role, strategy):
    sv = float('inf')
    for prof in game.knownProfiles():
        if strategy in prof[role]:
            sv = min(sv, game.getPayoff(prof, role, strategy))
    return sv


def social_welfare(game, profile, role=None):
    '''
    Sums values for a pure profile or expected values for a mixed profile.

    Restricts sum to specified role if role != None.
    '''
    if is_pure_profile(profile):
        values = (game.values[game[profile]] * game.counts[game[profile]])
    elif is_mixture_array(profile):
        players = np.array([game.players[r] for r in game.roles])
        values = (game.getExpectedPayoff(profile) * players)
    elif is_profile_array(profile):
        return social_welfare(game, game.toProfile(profile))
    elif is_mixed_profile(profile):
        return social_welfare(game, game.toArray(profile))
    else:
        raise TypeError('unrecognized profile type: %s' % profile)
    if role == None:
        return values.sum()
    else:
        return values[game.index(role)].sum()


def max_social_welfare(game, role=None):
    best_prof = None
    max_sw = -np.inf
    for prof in game.knownProfiles():
        sw = social_welfare(game, prof, role)
        if sw > max_sw:
            best_prof = prof
            max_sw = sw
    return best_prof, max_sw


PARSER = argparse.ArgumentParser(description='''Compute regret in input game(s)
of specified profiles.''')
PARSER.add_argument('profiles', type=str, help='''File with profiles from input
games for which regrets should be calculated.''')
PARSER.add_argument('--sw', action='store_true', help='''Calculate social
welfare instead of regret. Use keyword GLOBAL to calculate max social
welfare.''')
PARSER.add_argument('--ne', action='store_true', help='''Calculate 'NE regrets'
(regret a devitor would experience by switching to each other pure strategy)
for each profile instead of the profiles' regrets''')


def main():
    args = PARSER.parse_args()
    games = args.input

    #direct call to max_social_welfare()
    if args.profiles == 'GLOBAL' and args.SW:
        print to_JSON_str(max_social_welfare(games))
        return

    profiles = read(args.profiles)
    if not isinstance(profiles, list):
        profiles = [profiles]
    if not isinstance(games, list):
        games = [games] * len(profiles)
    regrets = []
    for g, prof_list in zip(games, profiles):
        if not isinstance(prof_list, list):
            prof_list = [prof_list]
        regrets.append([])
        for prof in prof_list:
            if args.SW:
                regrets[-1].append(social_welfare(g, prof))
            elif args.NE:
                eqr = equilibrium_regrets(g, prof)
                eqr_prof = {}
                for r in g.roles:
                    eqr_prof[r] = {}
                    for s in g.strategies[r]:
                        eqr_prof[r][s] = eqr[g.index(r),g.index(r,s)]
                regrets[-1].append(eqr_prof)
            else:
                regrets[-1].append(regret(g, prof))
    if len(regrets) > 1:
        print to_JSON_str(regrets)
    else:
        print to_JSON_str(regrets[0])


if __name__ == '__main__':
    main()
