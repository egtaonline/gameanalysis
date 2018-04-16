"""A module containing a canonicalized game"""
import functools

import numpy as np

from gameanalysis import gamereader
from gameanalysis import rsgame
from gameanalysis import utils


# TODO There's an issue here where incomplete payoffs for single strategy roles
# contribute to incomplete profiles. There's not an obvious way to remedy this
# with the current api in a way that works well.
class _CanonGame(rsgame._RsGame): # pylint: disable=protected-access
    """A game canonicalized to remove single strategy roles"""

    def __init__(self, game):
        role_mask = game.num_role_strats > 1
        super().__init__(
            tuple(r for r, m in zip(game.role_names, role_mask) if m),
            tuple(s for s, m in zip(game.strat_names, role_mask) if m),
            game.num_role_players[role_mask])
        self._game = game
        self._players = game.num_role_players[~role_mask]
        self._inds = np.cumsum(role_mask * game.num_role_strats)[~role_mask]
        self._mask = role_mask.repeat(game.num_role_strats)

    @property
    def num_complete_profiles(self):
        """Get the number of profiles with full data"""
        return self._game.num_complete_profiles

    @property
    def num_profiles(self):
        """Get the number of profiles"""
        return self._game.num_profiles

    @functools.lru_cache(maxsize=1)
    def profiles(self):
        """Get all profiles with any payoff data"""
        return self._game.profiles()[:, self._mask]

    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        """Get all payoff parallel with profiles()"""
        return self._game.payoffs()[:, self._mask]

    def deviation_payoffs(self, mixture, *, jacobian=False, **kw):
        """Get the deviation payoffs for a mixture"""
        unmix = np.insert(mixture, self._inds, 1.0)
        if not jacobian:
            return self._game.deviation_payoffs(unmix, **kw)[self._mask]

        dev, jac = self._game.deviation_payoffs(unmix, jacobian=True, **kw)
        return dev[self._mask], jac[self._mask][:, self._mask]

    def get_payoffs(self, profiles):
        """Get the payoffs for a profile or profiles"""
        unprofs = np.insert(profiles, self._inds, self._players, -1)
        return self._game.get_payoffs(unprofs)[..., self._mask]

    @utils.memoize
    def max_strat_payoffs(self):
        """Get the maximum strategy payoffs"""
        return self._game.max_strat_payoffs()[self._mask]

    @utils.memoize
    def min_strat_payoffs(self):
        """Get the minimum strategy payoffs"""
        return self._game.min_strat_payoffs()[self._mask]

    def restrict(self, restriction):
        """Restrict viable strategies for a canon game"""
        unrest = np.insert(restriction, self._inds, True)
        return _CanonGame(self._game.restrict(unrest))

    def _add_constant(self, constant):
        """Add a constant to a canon game"""
        return _CanonGame(self._game + constant)

    def _multiply_constant(self, constant):
        """Multiple canon game payoffs by a constant"""
        return _CanonGame(self._game * constant)

    def _add_game(self, _):
        """Add another game to canon game"""
        return NotImplemented

    def to_json(self):
        """Convert canon game to json object"""
        base = super().to_json()
        base['game'] = self._game.to_json()
        base['type'] = 'canon.1'
        return base

    def __contains__(self, profile):
        unprof = np.insert(profile, self._inds, self._players, -1)
        return unprof in self._game

    def __eq__(self, othr):
        # pylint: disable-msg=protected-access
        return super().__eq__(othr) and self._game == othr._game

    def __hash__(self):
        return hash((super().__hash__(), self._game))

    def __repr__(self):
        return '{}, {:d} / {:d})'.format(
            super().__repr__()[:-1], self.num_profiles, self.num_all_profiles)


def canon(game):
    """Canonicalize a game by removing single strategy roles

    Parameters
    ----------
    game : RsGame
        The game to canonizalize.
    """
    return _CanonGame(game)


def canon_json(jgame):
    """Read a canonicalized game from json"""
    return canon(gamereader.loadj(jgame['game']))
