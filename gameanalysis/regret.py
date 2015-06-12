"""A module for computing regret and social welfare of profiles"""
import sys
import argparse
import json
import numpy as np

from gameanalysis import subgame, rsgame


def pure_strategy_deviation_gains(game, prof):
    """Returns a nested dict containing all of the gains from deviation

    The nested dict maps role to strategy to deviation to gain. The profile
    must have data.

    """
    prof = game.to_profile(prof)
    return {role:
            {strat:
             {dev: game.get_payoff(
                 prof.deviate(role, strat, dev, default=np.nan),
                 role, dev) - payoff
              for dev in game.strategies[role]}
             for strat, payoff in strat_payoffs.items()}
            for role, strat_payoffs in game.get_payoffs(prof).items()}


def pure_strategy_regret(game, prof):
    """Returns the regret of a pure strategy profile in a game"""
    return max(max(max(gain for _, gain in dev_gain.items())
                   for _, dev_gain in strat_gain.items())
               for _, strat_gain
               in pure_strategy_deviation_gains(game, prof).items())


def _subgame_data(game, mix):
    """Returns true if we have all support data"""
    sub = subgame.EmptySubgame(game, game.to_profile(mix).support())
    return all(prof in game for prof in sub.all_profiles())


def _deviation_data(game, mix):
    """Returns a boolean array where True means we have data on mix deviations to
    that role strat

    """
    mix = game.to_profile(mix)
    support = mix.support()
    has_data = {role: {strat: True for strat in strats}
                for role, strats in game.strategies.items()}
    sub = subgame.EmptySubgame(game, support)
    for prof, role, dev in sub.deviation_profiles():
        has_data[role][dev] &= prof in game
    return game.to_array(has_data)


def mixture_deviation_gains(game, mix, as_array=False):
    """Returns all the gains from deviation from a mixed strategy

    Return type is a dict mapping role to deviation to gain. This is equivalent
    to what is sometimes called equilibrium regret.

    """
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
    """Return the regret of a mixture profile"""
    return mixture_deviation_gains(game, mix, as_array=True).max()


def pure_social_welfare(game, profile):
    """Returns the social welfare of a pure strategy profile in game"""
    indexable = game.as_profile(profile)
    array = game.as_array(profile, dtype=int)
    return np.sum(array * game.get_payoffs(indexable, as_array=True))


def mixed_social_welfare(game, mix):
    """Returns the social welfare of a mixed strategy profile"""
    return game.get_expected_payoff(mix, as_array=True).dot(
        game.players.values())


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

##########
# Parser #
##########

def _is_pure_profile(prof):
    """Returns true of the profile is pure"""
    # For an asymmetric game, this will always return false, but then it
    # shouldn't be an issue.
    return any(sum(strats.values()) > 1.5 for strats in prof.values())


_TYPE = {
    'regret': lambda prof: (pure_strategy_regret(prof)
                            if _is_pure_profile(prof)
                            else mixture_regret(prof)),
    'gains': lambda prof: (pure_strategy_deviation_gains(prof)
                           if _is_pure_profile(prof)
                           else mixture_deviation_gains(prof)),
    'ne': lambda prof: (pure_strategy_deviation_gains(prof)
                        if _is_pure_profile(prof)
                        else mixture_deviation_gains(prof)),
    'welfare': lambda prof: (pure_social_welfare(prof)
                             if _is_pure_profile(prof)
                             else mixed_social_welfare(prof))
}

_PARSER = argparse.ArgumentParser(add_help=False, description='''Compute regret
in input game of specified profiles.''')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='file', default=sys.stdout,
                     type=argparse.FileType('w'), help='''Output dominance
                     file. The contents depend on the format specified.
                     (default: stdout)''')
_PARSER.add_argument('profiles', type=argparse.FileType('r'), help='''File with
profiles from input games for which regrets should be calculated. This file
needs to be a json list. of profiles''')
_PARSER.add_argument('-t', '--type', metavar='type', default='regret',
                     choices=_TYPE, help='''What to return. regret: returns the
                     the regret of the profile; gains: returns a json object of
                     the deviators gains for every deviation; ne: return the
                     "nash equilibrium regrets", these are identical to gains;
                     welfare: returns the social welfare of the
                     profile. (default: %(default)s)''')
# _PARSER.add_argument('-m', '--max-welfare', action='store_true', help='''Ignore
# all other options, and instead return the maximum social welfare''')


def command(args, prog, print_help=False):
    _PARSER.prog = '{} {}'.format(_PARSER.prog, prog)
    if print_help:
        _PARSER.print_help()
        return
    args = _PARSER.parse_args(args)
    game = rsgame.Game.from_json(json.load(args.input))
    profiles = json.load(args.profiles)

    # Need to differentiate between mixed and pure
    regrets = [mixture_regret(game, prof) for prof in profiles]

    json.dump(regrets, args.output, defaults=lambda x: x.to_json())
    args.output.write('\n')
