"""Module for computing player reductions"""
import numpy as np

from gameanalysis import collect
from gameanalysis import profile


# TODO A lot of these methods use dictionaries which is fine for a single
# profile, but there are likely vectorized routines that will work faster for
# large groups of profiles. A game object may need to be passed into for
# efficient array representations.


def _expand_sym_profile(profile, full_players, reduced_players):
    """Expands symmetric hierarchical profile

    In the event that `full_players` isn't divisible by `reduced_players`, we
    first assign by rounding error and break ties in favor of more-played
    strategies. The final tie-breaker is alphabetical order.
    """
    strats, players = zip(*sorted(profile.items()))
    expanded_players = _expand_sym_array_profile(np.array(players, dtype=int),
                                                 full_players, reduced_players)
    return dict(zip(strats, map(int, expanded_players)))


def _expand_sym_array_profile(profile, full_players, reduced_players):
    """Expands an array profile, order of strategies must be sorted

    In the event that `full_players` isn't divisible by `reduced_players`, we
    first assign by rounding error and break ties in favor of more-played
    strategies. The final tie-breaker is alphabetical order."""
    assert profile.sum() == reduced_players
    # Maximum prevents divide by zero error; equivalent to + eps
    expand_prof = profile * full_players // np.maximum(reduced_players, 1)
    unassigned = full_players - expand_prof.sum()
    if unassigned == 0:
        return expand_prof

    error = profile * full_players / reduced_players - expand_prof
    inds = np.lexsort((np.arange(profile.size), -profile, -error))
    expand_prof[inds[:unassigned]] += 1
    return expand_prof


def _reduce_sym_profile(profile, full_players, reduced_players):
    """Reduce a symmetric hierarchical profile

    This returns none if there is no profile, and the reduced profile
    otherwise. This maintains the invariant that _reduce_sym_prof .
    _expand_sym_prof is the identity. The reverse is also the identity if a
    reduced profile exists."""
    strats, players = zip(*sorted(profile.items()))
    reduced_players = _reduce_sym_array_profile(np.array(players, dtype=int),
                                                full_players, reduced_players)
    return (dict(zip(strats, map(int, reduced_players)))
            if reduced_players is not None else None)


def _reduce_sym_array_profile(profile, full_players, reduced_players):
    """Same as reduce sym array profile but for arrays"""
    assert profile.sum() == full_players
    red_prof = np.ceil(profile * reduced_players / full_players).astype(int)
    overassigned = red_prof.sum() - reduced_players

    # See if standard rounding works
    if overassigned == 0:
        expanded = _expand_sym_array_profile(red_prof, full_players,
                                             reduced_players)
        return red_prof if np.all(profile == expanded) else None

    # Rounding doesn't work, so we need to try and find a consistent set of
    # strategies that we added a tie breaker to in order to in order to get the
    # reduction

    # What if every strategy was tie broken
    alternate = np.ceil((profile - 1) * reduced_players / full_players)\
        .astype(int)

    # The strategies that could have been tie broken
    diff = np.nonzero(alternate != red_prof)[0]

    if diff.size < overassigned:
        # Clearly impossible
        return None

    # Here we create a vector of all of the untiebroken strategies, and the
    # strategies that might have been tie broken. We analyze the sorting
    # criteria for expanding the profile, and see if a consistent selection of
    # tie broken strategies exists.
    error = np.concatenate([
        red_prof * full_players / reduced_players - profile,
        alternate[diff] * full_players / reduced_players - profile[diff] + 1])
    key = (np.concatenate((np.arange(red_prof.size), diff)),
           -np.concatenate((red_prof, alternate[diff])),
           -error)
    lexinds = np.lexsort(key)
    # True are tie broken strategies
    group = (lexinds // red_prof.size).astype(bool)
    # These are the strategy indexes
    indexes = lexinds % red_prof.size
    incrimented = np.nonzero(group)[0][:overassigned]
    # These are the indexes that could be tie broken
    inc_inds = diff[indexes[incrimented]]
    start = incrimented[-1] + 1
    # This checks that inc_inds is a valid selection
    valid = (np.setdiff1d(indexes[start:][~group[start:]], inc_inds, True).size
             == red_prof.size - overassigned)
    red_prof[inc_inds] = alternate[inc_inds]

    return red_prof if valid else None


class Hierarchical(object):
    """Hierarchical reduction"""
    def __init__(self, full_players, reduced_players):
        self.full_players = collect.frozendict(full_players)
        self.reduced_players = collect.frozendict(reduced_players)

        assert all(c > 0 for c in self.reduced_players.values()), \
            "All counts must be greater than zero"
        assert all(self.full_players[r] >= c for r, c
                   in self.reduced_players.items()), \
            "Can't reduce to a greater number of players"

    def reduce_game(self, game):
        """Convert an input game to a reduced game with new players

        This version uses exact math, and so will fail if your player counts
        are not DPR reducible.

        """
        assert game.players == self.full_players, \
            "The games players don't match up with this reduction"
        # Would need to be converted into a list anyways, so we just get it
        # out of the way. This prevents warning
        profiles = []
        for prof, payoffs in game.sample_profile_payoffs():
            red_prof = {role: _reduce_sym_profile(strats,
                                                  self.full_players[role],
                                                  self.reduced_players[role])
                        for role, strats in prof.items()}
            if not any(strats is None for strats in red_prof.values()):
                profiles.append(profile.Profile(red_prof)
                                .to_input_profile(payoffs))
        return game.from_payoff_format(self.reduced_players,
                                       game.strategies, profiles)

    def __repr__(self):
        return '{name}({arg1}, {arg2})'.format(
            name=self.__class__.__name__,
            arg1=self.full_players,
            arg2=self.reduced_players)


class DeviationPreserving(object):
    """Deviation preserving reduction"""

    def __init__(self, full_players, reduced_players):
        self.full_players = collect.frozendict(full_players)
        self.reduced_players = collect.frozendict(reduced_players)

        assert all(c > 0 for c in self.reduced_players.values()), \
            "All counts must be greater than zero"
        assert all(self.full_players[r] >= c for r, c
                   in self.reduced_players.items()), \
            "Can't reduce to a greater number of players"
        assert all(self.full_players[r] == 1 or c > 1 for r, c
                   in self.reduced_players.items()), \
            "Can't dpr to 1 unless that's the full player count"

    def expand_profile(self, dpr_profile):
        """Returns the full game profile whose payoff determines that of strat in the
        reduced game profile"""
        for dev_role, dev_strategies in dpr_profile.items():
            for dev_strategy in dev_strategies:
                full_profile = {}
                for role, strat_counts in dpr_profile.items():
                    if role == dev_role:
                        opp_prof = dict(strat_counts)
                        opp_prof[dev_strategy] -= 1
                        opp_prof = _expand_sym_profile(
                            opp_prof,
                            self.full_players[role] - 1,
                            self.reduced_players[role] - 1)
                        opp_prof[dev_strategy] += 1

                    else:
                        opp_prof = _expand_sym_profile(
                            strat_counts, self.full_players[role],
                            self.reduced_players[role])

                    full_profile[role] = opp_prof
                yield profile.Profile(full_profile)

    def _profile_contributions(self, full_prof):
        """Returns a generator of dpr profiles and the role-strategy pair that
        contributes to it"""
        # TODO This can be made more efficient because it will never return 2
        # or more profiles unless they are "pure"
        hierarchical_prof = {
            role: _reduce_sym_profile(prof, self.full_players[role],
                                      self.reduced_players[role])
            for role, prof in full_prof.items()}

        for role, strats in full_prof.items():
            full = self.full_players[role] - 1
            red = self.reduced_players[role] - 1

            if red == 0:
                # One player in role
                if all(p is not None for p in hierarchical_prof.values()):
                    strat = next(iter(strats))
                    yield profile.Profile(hierarchical_prof), role, strat

            else:
                for strat in strats:
                    toreduce = dict(strats)
                    toreduce[strat] -= 1
                    reduced = _reduce_sym_profile(toreduce, full, red)
                    prof = hierarchical_prof.copy()
                    prof[role] = reduced
                    if all(p is not None for p in prof.values()):
                        reduced[strat] += 1
                        yield profile.Profile(prof), role, strat

    def reduce_profile(self, full_profile):
        """Returns dpr profiles that contribute to the full profile

        Return is in the form of a generator"""
        return (p[0] for p in self._profile_contributions(full_profile))

    def reduce_game(self, game):
        """Convert an input game to a reduced game with new players

        This version uses exact math, and so will fail if your player counts
        are not DPR reducible. It also means the minimum number of players to
        reduce a role to is 2, unless the role only has one player to begin
        with.
        """
        assert game.players == self.full_players, \
            ("The games players don't match up with this reduction "
             "Game: {game} Reduction: {reduction}").format(
                 game=game.players, reduction=self.full_players)

        # Map from profile to role to strat to a list of payoffs This allows us
        # to incrementally build DPR profiles as we scan the data The list is
        # so we can keep multiple observations, but it's not clear how well we
        # can take advantage of this.
        profile_map = {}
        for prof, payoffs in game.sample_profile_payoffs():
            for red_prof, role, strat in self._profile_contributions(prof):
                (profile_map.setdefault(red_prof, {})
                 .setdefault(role, {})
                 .setdefault(strat, [])
                 .extend(payoffs[role][strat]))

        # This could be a generator, but it'd be turned into a list anyways.
        # Better to make this explicit.
        profiles = [prof.to_input_profile(payoff_map)
                    for prof, payoff_map in profile_map.items()
                    if (profile.support_set(payoff_map)
                        == profile.support_set(prof))]

        # Return type is the same type as the original game
        return game.from_payoff_format(self.reduced_players,
                                       game.strategies, profiles)

    def __repr__(self):
        return '{name}({arg1}, {arg2})'.format(
            name=self.__class__.__name__,
            arg1=self.full_players,
            arg2=self.reduced_players)


class Twins(DeviationPreserving):
    def __init__(self, full_players):
        super().__init__(full_players,
                         {r: min(2, p) for r, p in full_players.items()})

    def __repr__(self):
        return '{name}({arg1})'.format(
            name=self.__class__.__name__,
            arg1=self.full_players)


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

    def __repr__(self):
        return self.__class__.__name__ + '()'
