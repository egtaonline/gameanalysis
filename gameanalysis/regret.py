import argparse
import numpy as np

from gameanalysis import subgame


def _pure_strategy_deviation_gains(game, prof):
    '''Like the non private version, but this returns generators of tuples'''
    prof = game.to_profile(prof)
    return ((role,
             ((strat,
               ((dev, game.get_payoff_default(prof.deviate(role, strat, dev),
                                              role, dev) - payoff)
                for dev in game.strategies[role]))
              for strat, payoff in strat_payoffs.items()))
            for role, strat_payoffs in game.get_payoffs(prof).items())


def pure_strategy_deviation_gains(game, prof):
    '''Returns a nested dict containing all of the gains from deviation

    The nested dict maps role to strategy to deviation to gain.

    '''
    return {role: {strat: dict(dev_gain) for strat, dev_gain in strat_gain}
            for role, strat_gain
            in _pure_strategy_deviation_gains(game, prof)}


def pure_strategy_regret(game, prof):
    '''Returns the regret of a pure strategy profile in a game'''
    return max(max(max(gain for _, gain in dev_gain)
                   for _, dev_gain in strat_gain)
               for _, strat_gain in _pure_strategy_deviation_gains(game, prof))


def _subgame_data(game, mix):
    '''Returns true if we have all support data'''
    sub = subgame.EmptySubgame(game, game.to_profile(mix).support())
    return all(prof in game for prof in sub.all_profiles())


def _deviation_data(game, mix):
    '''Returns a boolean array where True means we have data on mix deviations to
    that role strat

    '''
    mix = game.to_profile(mix)
    support = mix.support()
    has_data = {role: {strat: True for strat in strats}
                for role, strats in game.strategies.items()}
    sub = subgame.EmptySubgame(game, support)
    for prof, role, dev in sub.deviation_profiles():
        has_data[role][dev] &= prof in game
    return game.to_array(has_data)


def mixture_deviation_gains(game, mix, as_array=False):
    '''Returns all the gains from deviation from a mixed strategy

    Return type is a dict mapping role to deviation to gain. This is equivalent
    to what is sometimes called equilibrium regret.

    '''
    if _subgame_data(game, mix):
        mix = game.to_array(mix)
        strategy_evs = game.expected_values(mix)
        role_evs = (strategy_evs * mix).sum(1)
        # No data for specific deviations
        # This is set after role_evs to not get invalid data
        strategy_evs[~_deviation_data(game, mix) & game._mask] = np.nan
        gains = strategy_evs - role_evs[:, np.newaxis]
    else:  # No necessary data
        gains = np.empty_like(game._mask)
        gains.fill(np.nan)

    if as_array:
        gains[~game._mask] = 0
        return gains
    else:
        return {role: {strat: gains[r, s] for s, strat in enumerate(strats)}
                for r, (role, strats) in enumerate(game.strategies.items())}


def mixture_regret(game, mix):
    '''Return the regret of a mixture profile'''
    return mixture_deviation_gains(game, mix, as_array=True).max()


# def neighbors(game, p, *args, **kwargs):
#     if isinstance(p, Profile):
#         return profile_neighbors(game, p, *args, **kwargs)
#     elif isinstance(p, np.ndarray):
#         return mixture_neighbors(game, p, *args, **kwargs)
#     raise TypeError('unrecognized argument type: ' + type(p).__name__)


# def profile_neighbors(game, profile, role=None, strategy=None,
#                       deviation=None):
#     if role is None:
#         return list(chain(*[profile_neighbors(game, profile, r, strategy, \
#                 deviation) for r in game.roles]))
#     if strategy is None:
#         return list(chain(*[profile_neighbors(game, profile, role, s, \
#                 deviation) for s in profile[role]]))
#     if deviation is None:
#         return list(chain(*[profile_neighbors(game, profile, role, strategy, \
#                 d) for d in set(game.strategies[role]) - {strategy}]))
#     return [profile.deviate(role, strategy, deviation)]


# def mixture_neighbors(game, mix, role=None, deviation=None):
#     n = set()
#     for profile in feasible_profiles(game, mix):
#         n.update(profile_neighbors(game, profile, role, deviation=deviation))
#     return n


# def feasible_profiles(game, mix, thresh=1e-3):
#     return [Profile({r:{s:p[game.index(r)].count(s) for s in set(p[ \
#             game.index(r)])} for r in game.roles}) for p in product(*[ \
#             CwR(filter(lambda s: mix[game.index(r), game.index(r,s)] >= \
#             thresh, game.strategies[r]), game.players[r]) for r \
#             in game.roles])]


# def symmetric_profile_regrets(game):
#     assert game.is_symmetric(), 'Game must be symmetric'
#     role = next(iter(game.strategies))
#     return {s: regret(game, rsgame.Profile({role:{s:game.players[role]}})) for s \
#             in game.strategies[role]}


_PARSER = argparse.ArgumentParser(description='''Compute regret in input game
of specified profiles.''')
_PARSER.add_argument('profiles', type=str, help='''File with profiles from input
games for which regrets should be calculated.''')
_PARSER.add_argument('--sw', action='store_true', help='''Calculate social
welfare instead of regret. Use keyword GLOBAL to calculate max social
welfare.''')
_PARSER.add_argument('--ne', action='store_true', help='''Calculate 'NE regrets'
(regret a devitor would experience by switching to each other pure strategy)
for each profile instead of the profiles' regrets''')


def command(args, prog, print_help=False):
    _PARSER.prog = '%s %s' % (_PARSER.prog, prog)
    args = _PARSER.parse_args()
    games = args.input

    #direct call to max_social_welfare()
    # if args.profiles == 'GLOBAL' and args.SW:
    #     print to_JSON_str(max_social_welfare(games))
    #     return

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
    # if len(regrets) > 1:
    #     print to_JSON_str(regrets)
    # else:
    #     print to_JSON_str(regrets[0])


if __name__ == '__main__':
    main()
