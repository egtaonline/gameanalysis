"""Module for computing player reductions"""
from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import profile


def _sym_hr_full_prof(hr_profile, full_players, reduced_players):
    """Expands symmetric hierarchal profile

    In the event that `full_players` isn't divisible by `reduced_players`, we
    first assign by rounding error and break ties in favor of more-played
    strategies. The final tie-breaker is alphabetical order."""
    full_profile = {s: c * full_players // reduced_players
                    for s, c in hr_profile.items()}
    unassigned = full_players - sum(full_profile.values())
    if unassigned == 0:
        return full_profile

    # Deal with non-divisible strategy counts
    rounding_error = {s: c * full_players / reduced_players - full_profile[s]
                      for s, c in hr_profile.items()}

    strat_order = sorted(
        hr_profile.keys(),
        # order by rounding error, more played, alphabetical
        key=lambda k: (-rounding_error[k], -hr_profile[k], k))

    for s in strat_order[:unassigned]:
        full_profile[s] += 1
    return full_profile


class Hierarchical(object):
    """Hierarchical reduction"""
    def __init__(self, full_players, reduced_players):
        self.full_players = full_players
        self.reduced_players = reduced_players

    def _hr_profiles(self, game):
        """Returns a generator over tuples of hr profiles and the corresponding
        profile for payoff data.

        The profile must be evenly divisible for the reduction."""
        fracts = {role: count // self.reduced_players[role]
                  for role, count in self.full_players.items()}
        for prof in game:
            if all(all(cnt % fracts[r] == 0 for cnt in ses.values())
                   for r, ses in prof.items()):
                red_prof = profile.Profile(
                    (r, [(s, cnt // fracts[r])
                         for s, cnt in ses.items()])
                    for r, ses in prof.items())
                yield red_prof, prof

    def reduce_game(self, game):
        """Convert an input game to a reduced game with new players

        This version uses exact math, and so will fail if your player counts
        are not DPR reducible.

        """
        assert game.players == self.full_players, \
            "The games players don't match up with this reduction"
        profiles = (red_prof.to_input_profile(game.get_payoffs(profile))
                    for red_prof, profile in self._hr_profiles(game))
        return rsgame.Game(self.reduced_players, game.strategies, profiles)


class DeviationPreserving(object):
    """Deviation preserving reduction"""

    def __init__(self, full_players, reduced_players):
        self.full_players = full_players
        self.reduced_players = reduced_players

    def expand_profile(self, dpr_profile):
        """Returns the full game profile whose payoff determines that of strat in the
        reduced game profile"""

        for role, strategies in dpr_profile.items():
            for strategy in strategies:
                full_profile = {}
                for r, strat_counts in dpr_profile.items():
                    if r == role:
                        opp_prof = dict(strat_counts)
                        opp_prof[strategy] -= 1
                        opp_prof = _sym_hr_full_prof(
                            opp_prof,
                            max(1, self.full_players[r] - 1),
                            max(1, self.reduced_players[r] - 1))
                        opp_prof[strategy] += 1

                    else:
                        opp_prof = _sym_hr_full_prof(strat_counts,
                                                     self.full_players[r],
                                                     self.reduced_players[r])

                    full_profile[r] = opp_prof
                yield profile.Profile(full_profile)

    def _profile_contributions(self, full_profile):
        """Returns a generator of dpr profiles and the role-strategy pair that
        contributes to it

        """
        # TODO Right now this is written only for exact DPR
        fracts = {role: count // self.reduced_players[role]
                  for role, count in self.full_players.items()}
        for role, strats in full_profile.items():
            r_fracts = dict(fracts)
            # The conditional fixed the case when one player is reduced down to
            # one player, but it does not support reducing n > 1 players down
            # to 1, which requires a hierarchical reduction.
            r_fracts[role] = (
                1 if self.full_players[role] == self.reduced_players[role] == 1
                else ((self.full_players[role] - 1)
                      // (self.reduced_players[role] - 1)))

            for strat in strats:
                prof_copy = dict(full_profile)
                prof_copy[role] = dict(strats)
                prof_copy[role][strat] -= 1

                if all(all(cnt % r_fracts[r] == 0 for cnt in ses.values())
                       for r, ses in prof_copy.items()):
                    # The inner loop needs to be a list, because it's
                    # evaluation depends on the current value of r, and
                    # therefore can't be lazily evaluated.
                    red_prof = profile.Profile(
                        (r, {s: (cnt // r_fracts[r]) +
                             (1 if r == role and s == strat else 0)
                             for s, cnt in ses.items()})
                        for r, ses in prof_copy.items())
                    yield red_prof, role, strat

    def reduce_profile(self, full_profile):
        """Returns dpr profiles that contribute to the full profile

        Return is in the form of a generator"""
        return (p[0] for p in self._profile_contributions(full_profile))

    def reduce_game(self, game):
        """Convert an input game to a reduced game with new players

        This version uses exact math, and so will fail if your player counts
        are not DPR reducible. It also means the minimum number of players to
        reduce a role to is 2, unless the role only has one player to begin
        with."""

        assert game.players == self.full_players, \
            "The games players don't match up with this reduction"

        # Map from profile to role to strat to a list of payoffs This allows us
        # to incrementally build DPR profiles as we scan the data The list is
        # so we can keep multiple observations, but it's not clear how well we
        # can take advantage of this.
        profile_map = {}
        for prof in game:
            payoffs = game.get_payoffs(prof)
            for red_prof, role, strat in self._profile_contributions(prof):
                (profile_map.setdefault(red_prof, {})
                 .setdefault(role, {})
                 .setdefault(strat, [])
                 .append(payoffs[role][strat]))

        profiles = (prof.to_input_profile(payoff_map)
                    for prof, payoff_map in profile_map.items()
                    if (subgame.support_set(payoff_map)
                        == subgame.support_set(prof)))

        return rsgame.Game(self.reduced_players, game.strategies, profiles)


class Twins(DeviationPreserving):
    def __init__(self, full_players):
        super().__init__(full_players,
                         {r: min(2, p) for r, p in full_players.items()})


class Identity(object):
    """Identity reduction (lack of reduction)"""
    def expand_profile(self, reduced_profile):
        """Returns full game profiles that contribute to reduced profile"""
        yield reduced_profile

    def reduce_profile(self, full_profile):
        """Returns reduced profiles that contribute to the full profile"""
        yield full_profile

    def reduce_game(self, game):
        """Convert an input game to a reduced game with new players"""
        return game
