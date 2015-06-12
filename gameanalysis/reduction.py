"""Module for computing player reductions"""
from gameanalysis import rsgame, subgame


def _hr_profiles(game, reduced_players):
    """Returns a generator over tuples of hr profiles and the corresponding profile
    for payoff data.

    The profile must be evenly divisible for the reduction.

    """
    fracts = {role: count // reduced_players[role]
              for role, count in game.players.items()}
    for profile in game:
        if all(all(cnt % fracts[r] == 0 for cnt in ses.values())
               for r, ses in profile.items()):
            red_prof = rsgame.PureProfile(
                (r, [(s, cnt // fracts[r])
                     for s, cnt in ses.items()])
                for r, ses in profile.items())
            yield red_prof, profile


def hierarchical_reduction(game, players):
    """Convert an input game to a reduced game with new players

    This version uses exact math, and so will fail if your player counts are
    not DPR reducible.

    """
    profiles = (red_prof.to_input_profile(game.get_payoffs(profile))
                for red_prof, profile in _hr_profiles(game, players))
    return rsgame.Game(players, game.strategies, profiles)


# def full_prof_sym(HR_profile, N):
#     """Returns the symmetric full game profile corresponding to the given
#     symmetric reduced game profile under hierarchical reduction.

#     In the event that N isn't divisible by n, we first assign by rounding
#     error and break ties in favor of more-played strategies. The final
#     tie-breaker is alphabetical order.

#     """
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


# def full_prof_DPR(DPR_profile, role, strat, players):
#     """Returns the full game profile whose payoff determines that of strat in the
#     reduced game profile.

#     """
#     full_prof = {}
#     for r in DPR_profile:
#         if r == role:
#             opp_prof = DPR_profile.asDict()[r]
#             opp_prof[strat] -= 1
#             full_prof[r] = full_prof_sym(opp_prof, players[r] - 1)
#             full_prof[r][strat] += 1
#         else:
#             full_prof[r] = full_prof_sym(DPR_profile[r], players[r])
#     return rsgame.PureProfile(full_prof)


def _dpr_profile_contributions(profile, players, reduced_players):
    """Returns a generator of dpr profiles and the role-strategy pair that
    contributes to it

    """
    # TODO Right now this is written only for exact DPR
    fracts = {role: count // reduced_players[role]
              for role, count in players.items()}
    for role, strats in profile.items():
        r_fracts = dict(fracts)
        # The conditional fixed the case when one player is reduced down to one
        # player, but it does not support reducing n > 1 players down to 1,
        # which requires a hierarchical reduction.
        r_fracts[role] = (1 if players[role] == reduced_players[role] == 1
                          else ((players[role] - 1)
                                // (reduced_players[role] - 1)))
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
    """Convert an input game to a reduced game with new players

    This version uses exact math, and so will fail if your player counts are
    not DPR reducible. It also means the minimum number of players to reduce a
    role to is 2, unless the role only has one player to begin with.

    """
    # Map from profile to role to strat to a list of payoffs
    # This allows us to incrementally build DPR profiles as we scan the data
    # The list is so we can keep multiple observations, but it's not clear how
    # well we can take advantage of this.
    profile_map = {}
    for prof in game:
        payoffs = game.get_payoffs(prof)
        for red_prof, role, strat in _dpr_profile_contributions(
                prof, game.players, players):
            (profile_map.setdefault(red_prof, {})
             .setdefault(role, {})
             .setdefault(strat, [])
             .append(payoffs[role][strat]))

    profiles = (prof.to_input_profile(payoff_map)
                for prof, payoff_map in profile_map.items()
                if (subgame.support_set(payoff_map)
                    == subgame.support_set(prof)))

    return rsgame.Game(players, game.strategies, profiles)


def twins_reduction(game, _=None):
    """Compute twins reduction on a game.

    This is identical to a DPR where every role is mapped to two players
    (except for roles with only a single player).

    """
    players = {r: min(2, p) for r, p in game.players.items()}
    return deviation_preserving_reduction(game, players)


# def DPR_profiles(game, players={}):
#     """Returns the profiles from game that contribute to the DPR game."""
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
