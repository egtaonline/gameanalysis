import functools

import numpy as np

from gameanalysis import gamereader
from gameanalysis import rsgame
from gameanalysis import utils


class MergeGame(rsgame.RsGame):
    """A Game representing the `t` merger between two other games

    Payoffs in this game are simply the weighted fraction of payoffs from game1
    and game2 such that the interpolation is smooth."""

    def __init__(self, game1, game2, t):
        super().__init__(
            game1.role_names, game1.strat_names, game1.num_role_players)
        self._game1 = game1
        self._game2 = game2
        self.t = t

    @property
    @utils.memoize
    def num_complete_profiles(self):
        if self._game1.is_complete() and self._game2.is_complete():
            return self.num_all_profiles
        else:
            profs1 = frozenset(
                utils.hash_array(prof) for prof, pay
                in zip(self._game1.profiles(), self._game1.payoffs())
                if not np.isnan(pay).any())
            profs2 = frozenset(
                utils.hash_array(prof) for prof, pay
                in zip(self._game2.profiles(), self._game2.payoffs())
                if not np.isnan(pay).any())
            return len(profs1.intersection(profs2))

    @property
    @utils.memoize
    def num_profiles(self):
        if self._game1.is_complete() and self._game2.is_complete():
            return self.num_all_profiles
        else:
            return self.profiles().shape[0]

    @functools.lru_cache(maxsize=1)
    def profiles(self):
        if self._game1.is_complete() and self._game2.is_complete():
            return self.all_profiles()
        else:
            profs1 = utils.axis_to_elem(self._game1.profiles())
            profs2 = utils.axis_to_elem(self._game2.profiles())
            return utils.elem_to_axis(np.intersect1d(profs1, profs2), int)

    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        return self.get_payoffs(self.profiles())

    def deviation_payoffs(self, mix, *, jacobian=False, **kw):
        if jacobian:
            d1, j1 = self._game1.deviation_payoffs(mix, jacobian=True, **kw)
            d2, j2 = self._game2.deviation_payoffs(mix, jacobian=True, **kw)
            return ((1 - self.t) * d1 + self.t * d2,
                    (1 - self.t) * j1 + self.t * j2)
        else:
            return ((1 - self.t) * self._game1.deviation_payoffs(mix, **kw) +
                    self.t * self._game2.deviation_payoffs(mix, **kw))

    def get_payoffs(self, profile):
        return ((1 - self.t) * self._game1.get_payoffs(profile) +
                self.t * self._game2.get_payoffs(profile))

    @utils.memoize
    def max_strat_payoffs(self):
        return ((1 - self.t) * self._game1.max_strat_payoffs() +
                self.t * self._game2.max_strat_payoffs())

    @utils.memoize
    def min_strat_payoffs(self):
        return ((1 - self.t) * self._game1.min_strat_payoffs() +
                self.t * self._game2.min_strat_payoffs())

    def normalize(self):
        return MergeGame(
            self._game1.normalize(), self._game2.normalize(), self.t)

    def restrict(self, rest):
        return MergeGame(
            self._game1.restrict(rest), self._game2.restrict(rest), self.t)

    def to_json(self):
        base = super().to_json()
        base['game_1'] = self._game1.to_json()
        base['game_2'] = self._game2.to_json()
        base['t'] = self.t
        base['type'] = 'merge.1'
        return base

    def __contains__(self, profile):
        return profile in self._game1 and profile in self._game2

    def __eq__(self, other):
        return (super().__eq__(other) and
                ((self._game1 == other._game1 and
                  self._game2 == other._game2 and
                  np.isclose(self.t, other.t)) or
                 (self._game1 == other._game2 and
                  self._game2 == other._game1 and
                  np.isclose(self.t, 1 - other.t))))

    def __hash__(self):
        return hash(frozenset((self._game1, self._game2)))

    def __repr__(self):
        return '{}, {:d} / {:d})'.format(
            super().__repr__()[:-1], self.num_profiles, self.num_all_profiles)


# TODO This creates a lazy merge. Once games implement + and *, this can be
# made lazy_merge and a full merge can be implemented.
def merge(game1, game2, t):
    """Merge two games by a `t` fraction

    Parameters
    ----------
    game1 : RsGame
        The first game to merge.
    game2 : RsGame
        The second game to merge.
    t : float
        The fraction to merge the games. 0 corresponds to a copy of `game1`, 1
        corresponds to `game2`, and somewhere between corresponds to the linear
        interpolation between them.
    """
    assert 0 <= t <= 1, "t must be in [0, 1] but was {:f}".format(t)
    assert rsgame.emptygame_copy(game1) == rsgame.emptygame_copy(game2), \
        "games must have identical structure"
    return MergeGame(game1, game2, t)


def merge_json(jgame):
    """Read a merged game from json"""
    base = rsgame.emptygame_json(jgame)
    game1 = gamereader.loadj(jgame['game_1'])
    game2 = gamereader.loadj(jgame['game_2'])
    assert base == rsgame.emptygame_copy(game1), \
        "game structure didn't match each merged game"
    return merge(game1, game2, jgame['t'])
