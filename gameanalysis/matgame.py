"""module for complete independent games"""
import functools
import itertools

import numpy as np

from gameanalysis import rsgame
from gameanalysis import serialize
from gameanalysis import utils


class MatrixGame(rsgame._CompleteGame):
    """Matrix game representation

    This represents a dense independent game as a matrix of payoffs.

    Parameters
    ----------
    matrix : ndarray-like
        The matrix of payoffs for an asymmetric game. The last axis is the
        payoffs for each player, the first axes are the strategies for each
        player. matrix.shape[:-1] must correspond to the number of strategies
        for each player. matrix.ndim - 1 must equal matrix.shape[-1]. This must
        be specified by itself.
    """

    def __init__(self, payoff_matrix):
        super().__init__(np.ones(payoff_matrix.shape[-1], int),
                         np.asarray(payoff_matrix.shape[:-1], int))
        assert payoff_matrix.shape[-1] == payoff_matrix.ndim - 1, \
            "matrix shape is inconsistent with a matrix game {}".format(
                payoff_matrix.shape)
        self.payoff_matrix = payoff_matrix
        self.payoff_matrix.setflags(write=False)

        self._prof_offset = np.zeros(self.num_strats, int)
        self._prof_offset[self.role_starts] = 1
        self._prof_offset.setflags(write=False)

        self._payoff_view = self.payoff_matrix.view()
        self._payoff_view.shape = (self.num_profiles, self.num_roles)

    @utils.memoize
    def min_strat_payoffs(self):
        """Returns the minimum payoff for each role"""
        mpays = np.empty(self.num_strats)
        for r, (pays, min_pays, n) in enumerate(zip(
                np.rollaxis(self.payoff_matrix, -1),
                np.split(mpays, self.role_starts[1:]),
                self.num_role_strats)):
            np.rollaxis(pays, r).reshape((n, -1)).min(1, min_pays)
        mpays.setflags(write=False)
        return mpays

    @utils.memoize
    def max_strat_payoffs(self):
        """Returns the minimum payoff for each role"""
        mpays = np.empty(self.num_strats)
        for r, (pays, max_pays, n) in enumerate(zip(
                np.rollaxis(self.payoff_matrix, -1),
                np.split(mpays, self.role_starts[1:]),
                self.num_role_strats)):
            np.rollaxis(pays, r).reshape((n, -1)).max(1, max_pays)
        mpays.setflags(write=False)
        return mpays

    @property
    @functools.lru_cache(maxsize=1)
    def profiles(self):
        return self.all_profiles()

    @property
    @functools.lru_cache(maxsize=1)
    def payoffs(self):
        profiles = self.profiles
        payoffs = np.zeros(profiles.shape)
        payoffs[profiles > 0] = self.payoff_matrix.flat
        return payoffs

    def compress_profile(self, profile):
        """Compress profile in array of ints

        Normal profiles are an array of number of players playing a strategy.
        Since matrix games always have one player per role, this compresses
        each roles counts into a single int representing the played strategy
        per role.
        """
        assert self.is_profile(profile).all()
        profile = np.asarray(profile, int)
        return np.add.reduceat(np.cumsum(self._prof_offset - profile, -1),
                               self.role_starts, -1)

    def uncompress_profile(self, comp_prof):
        comp_prof = np.asarray(comp_prof, int)
        assert np.all(comp_prof >= 0) and np.all(
            comp_prof < self.num_role_strats)
        profile = np.zeros(comp_prof.shape[:-1] + (self.num_strats,), int)
        inds = (comp_prof.reshape((-1, self.num_roles)) +
                self.role_starts + self.num_strats *
                np.arange(int(np.prod(comp_prof.shape[:-1])))[:, None])
        profile.flat[inds] = 1
        return profile

    def get_payoffs(self, profile):
        """Returns an array of profile payoffs"""
        profile = np.asarray(profile, int)
        ids = self.profile_id(profile)
        payoffs = np.zeros(profile.shape[:-1] + (self.num_strats,))
        payoffs[profile > 0] = self._payoff_view[ids].flat
        return payoffs

    def deviation_payoffs(self, mix, *, jacobian=False):
        """Computes the expected value of each pure strategy played against all
        opponents playing mix.

        Parameters
        ----------
        mix : ndarray
            The mix all other players are using
        jacobian : bool
            If true, the second returned argument will be the jacobian of the
            deviation payoffs with respect to the mixture. The first axis is
            the deviating strategy, the second axis is the strategy in the mix
            the jacobian is taken with respect to.
        """
        # TODO This has a lot of for loops. It might be able to be more
        # efficient if we add _TINY to the mixture so we can just divide, or
        # potentially reshape payoff_matrix, but I'm not really sure.
        rmix = []
        for r, m in enumerate(np.split(mix, self.role_starts[1:])):
            shape = [1] * self.num_roles
            shape[r] = -1
            rmix.append(m.reshape(shape))
        devpays = np.empty(self.num_strats)
        for r, (out, n) in enumerate(zip(
                np.split(devpays, self.role_starts[1:]),
                self.num_role_strats)):
            pays = self.payoff_matrix[..., r].copy()
            for m in rmix[:r]:
                pays *= m
            for m in rmix[r + 1:]:
                pays *= m
            np.rollaxis(pays, r).reshape((n, -1)).sum(1, out=out)

        if not jacobian:
            return devpays

        jac = np.zeros((self.num_strats, self.num_strats))
        for r, (jout, nr) in enumerate(zip(
                np.split(jac, self.role_starts[1:]),
                self.num_role_strats)):
            for d, (out, nd) in enumerate(zip(
                    np.split(jout, self.role_starts[1:], 1),
                    self.num_role_strats)):
                if r == d:
                    continue
                pays = self.payoff_matrix[..., r].copy()
                f, s = min(r, d), max(r, d)
                for m in rmix[:f]:
                    pays *= m
                for m in rmix[f + 1:s]:
                    pays *= m
                for m in rmix[s + 1:]:
                    pays *= m
                np.rollaxis(np.rollaxis(pays, r), d + (r > d),
                            1).reshape((nr, nd, -1)).sum(2, out=out)

        # Normalize
        jac -= np.repeat(np.add.reduceat(jac, self.role_starts, 1) /
                         self.num_role_strats, self.num_role_strats, 1)
        return devpays, jac

    def subgame(self, subgame_mask):
        subgame_mask = np.asarray(subgame_mask, bool)
        assert self.is_subgame(subgame_mask), \
            "subgame_mask must be valid"
        matrix = self.payoff_matrix
        for i, mask in enumerate(np.split(subgame_mask, self.role_starts[1:])):
            matrix = matrix[(slice(None),) * i + (mask,)]
        return matgame(matrix.copy())

    def normalize(self):
        """Return a normalized MatGame"""
        scale = self.max_role_payoffs() - self.min_role_payoffs()
        scale[np.isclose(scale, 0)] = 1
        return matgame((self.payoff_matrix - self.min_role_payoffs()) / scale)

    @utils.memoize
    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return (super().__eq__(other) and
                # Identical payoffs
                np.allclose(self.payoff_matrix, other.payoff_matrix))

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            self.num_role_strats)


def matgame(payoff_matrix):
    """Create a game from a dense matrix

    Parameters
    ----------
    payoff_matrix : ndarray-like
        The matrix of payoffs for an asymmetric game. The last axis is the
        payoffs for each player, the first axes are the strategies for each
        player. matrix.shape[:-1] must correspond to the number of strategies
        for each player. matrix.ndim - 1 must equal matrix.shape[-1]. This must
        be specified by itself.
    """
    return MatrixGame(np.asarray(payoff_matrix, float))


def matgame_copy(copy_game):
    """Copy a matrix game from an existing game

    Parameters
    ----------
    copy_game : BaseGame
        Game to copy payoff out of. This game must be complete.
    """
    assert copy_game.is_complete()

    if hasattr(copy_game, 'payoff_matrix'):
        # Matrix Game
        return matgame(copy_game.payoff_matrix)
    else:
        num_role_strats = copy_game.num_role_strats.repeat(
            copy_game.num_role_players)
        shape = tuple(num_role_strats) + (num_role_strats.size,)
        payoff_matrix = np.empty(shape, float)
        offset = copy_game.role_starts.repeat(copy_game.num_role_players)
        for profile, payoffs in zip(copy_game.profiles, copy_game.payoffs):
            # TODO Is is possible to do this with array logic?
            inds = itertools.product(*[
                set(itertools.permutations(np.arange(s.size).repeat(s))) for s
                in np.split(profile, copy_game.role_starts[1:])])
            for nested in inds:
                ind = tuple(itertools.chain.from_iterable(nested))
                payoff_matrix[ind] = payoffs[ind + offset]
        return matgame(payoff_matrix)


class MatGameSerializer(serialize._BaseSerializer):
    """A serializer for agfn games

    Parameters
    ----------
    role_names : [str]
        Names of each role.
    strat_names : [[str]]
        Names of each strategy for each role.
    """
    # XXX A lot of this logic relies on the fact that serializer roles are
    # always sorted, and will fail badly if that assumption does not hold.

    def __init__(self, role_names, strat_names):
        super().__init__(role_names, strat_names)

    def _mat_from_json(self, dic, matrix, depth):
        """Copy roles to a matrix representation"""
        if depth == self.num_roles:
            for role, payoff in dic.items():
                matrix[self.role_index(role)] = payoff
        else:
            role = self.role_names[depth]
            offset = self.role_starts[depth]
            depth += 1
            for strat, subdic in dic.items():
                ind = self.role_strat_index(role, strat) - offset
                self._mat_from_json(subdic, matrix[ind], depth)

    def from_json(self, game):
        matrix = np.empty(tuple(self.num_role_strats) + (self.num_roles,),
                          float)
        self._mat_from_json(game['payoffs'], matrix, 0)
        return matgame(matrix)

    def _mat_to_json(self, matrix, role_index):
        """Convert a sub matrix into json representation"""
        if role_index == self.num_roles:
            return {role: float(pay) for role, pay
                    in zip(self.role_names, matrix)}
        else:
            strats = self.strat_names[role_index]
            role_index += 1
            return {strat: self._mat_to_json(mat, role_index)
                    for strat, mat in zip(strats, matrix)}

    def to_json(self, game):
        res = super().to_json(game)
        res['payoffs'] = self._mat_to_json(game.payoff_matrix, 0)
        res['type'] = 'matrix.1'
        return res

    def subserial(self, subgame_mask):
        """Restrict possible strategies"""
        subgame_mask = np.asarray(subgame_mask, bool)
        assert self.is_subgame(subgame_mask), \
            "subgame_mask must be valid"
        strat_names = [[s for s, m in zip(strats, mask) if m]
                       for strats, mask
                       in zip(self.strat_names,
                              np.split(subgame_mask, self.role_starts[1:]))]
        return matgameserializer(self.role_names, strat_names)


def matgameserializer(role_names, strat_names):
    """Static constructor for AgfnGameSerializer

    Parameters
    ----------
    role_names : [str]
    strat_names : [[str]]
    function_names : [str]
    """
    return MatGameSerializer(
        tuple(role_names), tuple(map(tuple, strat_names)))


def matgameserializer_json(json):
    """Static constructor for AgfnGameSerializer

    Takes a game that would be loaded from json and determines field names.

    Parameters
    ----------
    json : json
        A json format of a base AgfnGame. One standard output is the one output
        by to_agfngame_json. {strategies: {<role>: [<strat>]}, function_names:
        [<func>]}
    """
    return matgameserializer_copy(serialize.gameserializer_json(json))


def matgameserializer_copy(serial, players=None):
    """Copy a MatGameSerializer from a serializer

    If players is unspecified, we assume the game serializer this came from
    only had one player per role. If not, the serializer is repeated to match
    the number of the players.
    """
    assert players is None or len(players) == serial.num_roles, \
        "can't specify a different number of players than roles"
    if players is None or all(p == 1 for p in players):
        return MatGameSerializer(serial.role_names, serial.strat_names)
    else:
        if utils.is_sorted(r + 'p' for r in serial.role_names):
            # We can naively append player numbers
            role_names = serial.role_names
        else:
            # We have to prefix to preserve role order
            maxlen = max(map(len, serial.role_names))
            role_names = (p + '_' * (maxlen - len(r)) + r
                          for r, p
                          in zip(serial.role_names,
                                 utils.prefix_strings('', serial.num_roles)))
        roles = tuple(itertools.chain.from_iterable(
            (r + s for s in utils.prefix_strings('p', p))
            for r, p in zip(role_names, players)))
        strats = tuple(itertools.chain.from_iterable(
            itertools.repeat(s, p) for s, p
            in zip(serial.strat_names, players)))
        return MatGameSerializer(roles, strats)


def read_matgame(json):
    """Read a matgame and its associate serializer from json"""
    serial = matgameserializer_json(json)
    return serial.from_json(json), serial
