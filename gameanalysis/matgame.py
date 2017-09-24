"""module for complete independent games"""
import functools
import itertools

import numpy as np
import numpy.random as rand

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


class SampleMatrixGame(MatrixGame):
    """Create a game from a dense matrix

    Parameters
    ----------
    spayoff_matrix : ndarray-like
        The matrix of payoffs for an asymmetric game. The last axis is the
        payoffs for each player, the first axes are the strategies for each
        player. matrix.shape[:-1] must correspond to the number of strategies
        for each player. matrix.ndim - 2 must equal matrix.shape[-2]. The last
        dimension is the number of samples.
    """

    def __init__(self, spayoff_matrix):
        super().__init__(spayoff_matrix.mean(-1))
        self.spayoff_matrix = spayoff_matrix
        self.spayoff_matrix.setflags(write=False)

        self.num_samples = np.array([spayoff_matrix.shape[-1]], int)
        self.num_sample_profs = np.array([self.num_profiles], int)
        self.sample_starts = np.zeros(1, int)

        self._spayoff_view = self.spayoff_matrix.view()
        self._spayoff_view.shape = (self.num_profiles, self.num_roles,
                                    self.num_samples[0])

    @property
    @functools.lru_cache(maxsize=1)
    def sample_payoffs(self):
        profiles = self.profiles
        spayoffs = np.zeros(profiles.shape + (self.num_samples[0],))
        # This next set of steps is a hacky way of avoiding duplicating
        # mask by num_samples
        pview = spayoffs.view()
        pview.shape = (-1, self.num_samples[0])
        mask = profiles > 0
        mask.shape = (-1, 1)
        mask = np.broadcast_to(mask, (mask.size, self.num_samples[0]))
        np.place(pview, mask, self.spayoff_matrix.flat)
        return spayoffs,

    def resample(self, *, num_resamples=None, independent_profile=False,
                 independent_role=False, independent_strategy=False):
        """Overwrite payoff values with a bootstrap resample

        Keyword Arguments
        -----------------
        num_resamples:        The number of resamples to take for each realized
                              payoff. By default this is equal to the number of
                              observations for that profile.
        independent_profile:  Sample each profile independently.
                              (default: False)
        independent_role:     Sample each role independently. Within a profile,
                              the payoffs for each role will be drawn
                              independently. (default: False)
        independent_strategy: Ignored

        Each of the `independent_` arguments will increase the time to do a
        resample. `independent_strategy` doesn't work for matrix games.
        """
        num_resamples = num_resamples or self.num_samples[0]
        dim_first = (self.num_role_strats if independent_profile
                     else np.ones(self.num_roles, int))
        dim_last = self.num_roles if independent_role else 1

        sample = rand.multinomial(
            num_resamples, np.ones(self.num_samples[0]) / self.num_samples[0],
            tuple(dim_first) + (dim_last,))
        payoff_matrix = (np.mean(self.spayoff_matrix * sample, -1) *
                         (self.num_samples[0] / num_resamples))
        return matgame(payoff_matrix)

    def get_sample_payoffs(self, profile):
        """Get sample payoffs associated with a profile

        This returns an array of shape (num_samples, num_role_strats)"""
        profile = np.asarray(profile, int)
        ids = self.profile_id(profile)
        spayoffs = np.zeros(profile.shape[:-1] +
                            (self.num_strats, self.num_samples[0]))
        mask = np.broadcast_to((profile > 0)[..., None], spayoffs.shape)
        spayoffs[mask] = self._spayoff_view[ids].flat
        return spayoffs.swapaxes(-1, -2)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return (
            type(self) is type(other) and
            self.num_roles == other.num_roles and
            np.all(self.num_role_strats == other.num_role_strats) and
            np.all(self.num_role_players == other.num_role_players) and
            # Same sample number
            np.all(self.num_samples == other.num_samples) and
            # Identical payoffs
            all(_sample_payoff_mats_equal(p1, p2) for p1, p2 in zip(
                self.spayoff_matrix.reshape((-1, self.num_roles,
                                             self.num_samples[0])),
                other.spayoff_matrix.reshape((-1, other.num_roles,
                                              other.num_samples[0])))))

    def __repr__(self):
        return '{}, {})'.format(super().__repr__()[:-1], self.num_samples[0])


def _sample_payoff_mats_equal(p1, p2):
    """Returns true if two sample payoff matricies are almost equal"""
    # FIXME Pathological payoffs will make this fail
    return np.allclose(p1[:, np.lexsort(p1)], p2[:, np.lexsort(p2)])


def samplematgame(payoff_matrix):
    """Create a game from a dense matrix

    Parameters
    ----------
    spayoff_matrix : ndarray-like
        The matrix of payoffs for an asymmetric game. The last axis is the
        payoffs for each player, the first axes are the strategies for each
        player. matrix.shape[:-1] must correspond to the number of strategies
        for each player. matrix.ndim - 2 must equal matrix.shape[-2]. The last
        dimension is the number of samples.
    """
    return SampleMatrixGame(np.asarray(payoff_matrix, float))


def samplematgame_copy(copy_game):
    """Copy a sample matrix game from an existing game

    Parameters
    ----------
    copy_game : BaseGame
        Game to copy payoff out of. This game must be complete.
    """
    assert copy_game.is_complete()

    if hasattr(copy_game, 'spayoff_matrix'):
        # SampleMatGame
        return samplematgame(copy_game.spayoff_matrix)
    elif hasattr(copy_game, 'sample_payoffs'):
        # SampleGame
        spayoffs = itertools.chain.from_iterable(copy_game.sample_payoffs)
        num_role_strats = copy_game.num_role_strats.repeat(
            copy_game.num_role_players)
        num_samples = copy_game.num_samples.min()
        shape = tuple(num_role_strats) + (num_role_strats.size, num_samples)
        spayoff_matrix = np.empty(shape, float)
        for profile, spayoff in zip(copy_game.profiles, spayoffs):
            # TODO Is is possible to do this with array logic?
            inds = itertools.product(*[
                set(itertools.permutations(np.arange(s.size).repeat(s))) for s
                in np.split(profile, copy_game.role_starts[1:])])
            for nested in inds:
                ind = tuple(itertools.chain.from_iterable(nested))
                # XXX This is truncated, but may it should be shuffled first?
                spayoff_matrix[ind] = spayoff[profile > 0, :num_samples]
        return samplematgame(spayoff_matrix)
    else:
        return samplematgame(matgame_copy(copy_game).payoff_matrix[..., None])


class MatGameSerializer(serialize._BaseSerializer):
    """A serializer for agfn games

    Parameters
    ----------
    role_names : [str]
        Names of each role.
    strat_names : [[str]]
        Names of each strategy for each role.
    """

    def __init__(self, role_names, strat_names):
        super().__init__(role_names, strat_names)

    def from_json(self, game):
        return matgame(np.asarray(game['matrix'], float))

    def to_json(self, game):
        res = super().to_json(game)
        res['matrix'] = game.payoff_matrix.tolist()
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


def matgameserializer_copy(serial):
    """Copy a MatGameSerializer from a serializer"""
    return matgameserializer(serial.role_names, serial.strat_names)


def read_matgame(json):
    """Read a matgame and its associate serializer from json"""
    serial = matgameserializer_json(json)
    return serial.from_json(json), serial
