import sys
import argparse
import json
import itertools

from gameanalysis import rsgame


def hierarchical_reduction(game, players):
    raise NotImplementedError
#     HR_game = type(game)(game.roles, players, game.strategies)
#     for reduced_profile in HR_game.allProfiles():
#         try:
#             full_profile = Profile({r:full_prof_sym(reduced_profile[r], \
#                     game.players[r]) for r in game.roles})
#             HR_game.addProfile({r:[PayoffData(s, reduced_profile[r][s], \
#                     game.getPayoffData(full_profile, r, s)) for s in \
#                     full_profile[r]] for r in full_profile})
#         except KeyError:
#             continue
#     return HR_game


# def full_prof_sym(HR_profile, N):
#     '''Returns the symmetric full game profile corresponding to the given
#     symmetric reduced game profile under hierarchical reduction.

#     In the event that N isn't divisible by n, we first assign by rounding
#     error and break ties in favor of more-played strategies. The final
#     tie-breaker is alphabetical order.

#     '''
#     if N < 2:
#         return HR_profile
#     n = sum(HR_profile.values())
#     full_profile = {s : (c * N / n) if n > 0 else N for s,c in \
#                     HR_profile.items()}
#     if sum(full_profile.values()) == N:
#         return full_profile

#     #deal with non-divisible strategy counts
#     rounding_error = {s : float(c * N) / n - full_profile[s] for \
#                         s,c in HR_profile.items()}
#     strat_order = sorted(HR_profile.keys())
#     strat_order.sort(key=HR_profile.get, reverse=True)
#     strat_order.sort(key=rounding_error.get, reverse=True)
#     for s in strat_order[:N - sum(full_profile.values())]:
#         full_profile[s] += 1
#     return full_profile


def full_prof_DPR(DPR_profile, role, strat, players):
    '''Returns the full game profile whose payoff determines that of strat in the
    reduced game profile.

    '''
    full_prof = {}
    for r in DPR_profile:
        if r == role:
            opp_prof = DPR_profile.asDict()[r]
            opp_prof[strat] -= 1
            full_prof[r] = full_prof_sym(opp_prof, players[r] - 1)
            full_prof[r][strat] += 1
        else:
            full_prof[r] = full_prof_sym(DPR_profile[r], players[r])
    return rsgame.PureProfile(full_prof)


def _profile_contributions(profile, players, reduced_players):
    '''Returns a generator of dpr profiles and the role-strategy pair that
    contributes to it

    '''
    # TODO Right now this is written only for exact DPR
    fracts = {role: count // reduced_players[role]
              for role, count in players.items()}
    for role, strats in profile.items():
        r_fracts = dict(fracts)
        # The conditional fixed the case when one player is reduced down to one
        # player, but it does not support reducing n > 1 players down to 1,
        # which requires a hierarchical reduction.
        r_fracts[role] = (1 if players[role] == reduced_players[role] == 1
                          else (players[role] - 1) // (reduced_players[role] - 1))
        for strat in strats:
            prof_copy = dict(profile)
            prof_copy[role] = dict(strats)
            prof_copy[role][strat] -= 1

            if all(all(cnt % r_fracts[r] == 0 for cnt in ses.values())
                   for r, ses in prof_copy.items()):
                # The inner loop needs to be a list, because it's evaluation
                # depends on the current value of r, and therefore can't be
                # lazily evaluated.
                red_prof = rsgame.PureProfile(
                    (r, [(s, (cnt // r_fracts[r]) +
                          (1 if r == role and s == strat else 0))
                         for s, cnt in ses.items()])
                    for r, ses in prof_copy.items())
                yield red_prof, role, strat


def deviation_preserving_reduction(game, players):
    '''Convert an input game to a reduced game with new players

    This version uses exact math, and so will fail if your player counts are
    not DPR reducible.

    '''
    # Map from profile to (role, strat) to a list of payoffs
    # The list is so we can keep multiple observations, but it's not clear how
    # well we can take advantage of this.
    profile_map = {}
    for prof in game:
        payoffs = game.get_payoffs(prof)
        for red_prof, role, strat in _profile_contributions(
                prof, game.players, players):
            (profile_map.setdefault(red_prof, {}).setdefault((role, strat), [])
             .append(payoffs[role][strat]))

    # What follows is a long, but documented generator expression.
    #
    # This stage builds the structure {role: sym_group}
    profs = ({role: {(strat, counts, tuple(payoff_map[(role, strat)]))
                     for strat, counts in strats.items()}
              for role, strats in prof.items()}
             # Profile map contains a profile key {role: {strat: count}} and a
             # payoff map {(role, strat): payoff}, all the info we need to
             # construct a profile
             for prof, payoff_map in profile_map.items()
             # This final stage turns the profile into (role, strat) tuples and
             # verifies that all are present in the payoff map before we
             # construct a profile. Otherwise we would not have data for all of
             # the strategies for a profile.
             if all(rs in payoff_map for rs in itertools.chain.from_iterable(
                 ((r, s) for s in ses) for r, ses in prof.items())))

    return rsgame.Game(players, game.strategies, profs)


def twins_reduction(game, _=None):
    players = {r: min(2, p) for r, p in game.players.items()}
    return deviation_preserving_reduction(game, players)


# def DPR_profiles(game, players={}):
#     '''Returns the profiles from game that contribute to the DPR game.'''
#     if not players:
#         players = {r: 2 for r in game.roles}
#     elif len(game.roles) == 1 and isinstance(players, int):
#         players = {game.roles[0]:players}
#     elif isinstance(players, list):
#         players = dict(zip(game.roles, players))
#     DPR_game = rsgame.Game(game.roles, players, game.strategies)
#     profiles = []
#     for DPR_prof in DPR_game.allProfiles():
#         for r in game.roles:
#             for s in DPR_prof[r]:
#                 full_prof = full_prof_DPR(DPR_prof, r, s, game.players)
#                 profiles.append(full_prof)
#     return profiles


def _parse_sorted(players, game):
    assert len(players) == len(game.strategies), \
        'Must input a reduced count for every role'
    return dict(zip(game.strategies, map(int, players)))


def _parse_inorder(players, game):
    assert len(players) == 2 * len(game.strategies), \
        'Must input a reduced count for every role'
    parsed = {}
    for i in range(0, len(players), 2):
        assert players[i] in game.strategies, \
            'role "%s" not found in game' % players[i]
        parsed[players[i]] = int(players[i + 1])
    return parsed


_PLAYERS = {
    True: _parse_sorted,
    False: _parse_inorder}

_REDUCTIONS = {
    'dpr': deviation_preserving_reduction,
    'hr': hierarchical_reduction,
    'tr': twins_reduction}

_PARSER = argparse.ArgumentParser(add_help=False, description='')
_PARSER.add_argument('--input', '-i', metavar='game-file', default=sys.stdin,
                     type=argparse.FileType('r'), help='''Input game file.
                     (default: stdin)''')
_PARSER.add_argument('--output', '-o', metavar='reduced-file', default=sys.stdout,
                     type=argparse.FileType('w'), help='''Output equilibria
                     file. This file will contain a json list of mixed
                     profiles. (default: stdout)''')
_PARSER.add_argument('--type', '-t', choices=_REDUCTIONS, default='dpr',
                     help='''Type of reduction to perform. (default:
                     %(default)s)''')
_PARSER.add_argument('--sorted-roles', '-s', action='store_true', help='''If
set, players should be a list of reduced counts for the role names in sorted
order.''')
_PARSER.add_argument('players', nargs='*', help='''Number of players in each
reduced-game role. This should be a list of role then counts e.g. "role1 4
role2 2"''')


def command(args, prog, print_help=False):
    _PARSER.prog = '%s %s' % (_PARSER.prog, prog)
    if print_help:
        _PARSER.print_help()
        return
    args = _PARSER.parse_args(args)
    game = rsgame.Game.from_json(json.load(args.input))
    players = _PLAYERS[args.sorted_roles](args.players, game)

    reduced = _REDUCTIONS[args.type](game, players)

    json.dump(reduced, args.output, default=lambda x: x.to_json())
    args.output.write('\n')
