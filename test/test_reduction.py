"""Test reductions"""
import numpy as np
import numpy.random as rand
import pytest

from gameanalysis import gamegen
from gameanalysis import paygame
from gameanalysis import restrict
from gameanalysis import rsgame
from gameanalysis import utils
from gameanalysis.reduction import deviation_preserving as dpr
from gameanalysis.reduction import hierarchical as hr
from gameanalysis.reduction import identity as ir
from gameanalysis.reduction import twins as tr


@pytest.mark.parametrize('keep_prob', [0.6, 0.9, 1])
@pytest.mark.parametrize('game_desc', [
    ([1], [1], [1]),
    ([2], [1], [2]),
    ([3], [1], [2]),
    ([1], [2], [1]),
    ([1, 2], [2, 1], [1, 2]),
    ([1, 4], [2, 1], [1, 2]),
    ([4, 9], [3, 2], [2, 3]),
])
@pytest.mark.parametrize('_', range(10)) # pylint: disable=too-many-locals
def test_random_dpr(keep_prob, game_desc, _):
    """Simple test that dpr functions are consistent"""
    players, strategies, red_players = game_desc
    # Create game
    game = gamegen.game(players, strategies, keep_prob)

    # Try to reduce game
    red_game = dpr.reduce_game(game, red_players)
    assert (rsgame.empty(red_players, strategies) ==
            dpr.reduce_game(rsgame.empty(players, strategies),
                            red_players))

    # Assert that reducing all profiles covers reduced game
    reduced_profiles = utils.axis_to_elem(red_game.profiles())
    reduced_full_profiles = utils.axis_to_elem(
        dpr.reduce_profiles(red_game, game.profiles()))
    assert np.setdiff1d(reduced_profiles, reduced_full_profiles).size == 0, \
        "reduced game contained profiles it shouldn't have"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game, we need to first filter for complete profiles
    full_profiles = utils.axis_to_elem(game.profiles())
    complete_profs = ~np.isnan(red_game.payoffs()).any(1)
    full_reduced_profiles = utils.axis_to_elem(
        dpr.expand_profiles(game, red_game.profiles()[complete_profs]))
    assert np.setdiff1d(full_reduced_profiles, full_profiles).size == 0, \
        'full game did not have data for all profiles required of reduced'

    # Assert that dpr counts are accurate
    num_dpr_profiles = dpr.expand_profiles(
        game, red_game.all_profiles()).shape[0]
    assert num_dpr_profiles == red_game.num_all_dpr_profiles

    # Test the dpr deviation profile counts are accurate
    rest = red_game.random_restriction()
    dpr_devs = dpr.expand_profiles(
        game, restrict.deviation_profiles(red_game, rest)).shape[0]
    num = restrict.num_dpr_deviation_profiles(red_game, rest)
    assert dpr_devs == num, \
        "num_dpr_deviation_profiles didn't return correct number"


def test_empty_dpr_1():
    """Reduction is empty because profile is invalid"""
    profiles = [
        [2, 4],
    ]
    payoffs = [
        [1, 2],
    ]
    game = paygame.game(6, 2, profiles, payoffs)
    red_game = dpr.reduce_game(game, 2)
    assert np.all(red_game.num_role_players == [2])
    assert red_game.is_empty()


def test_empty_dpr_2():
    """Reduction is empty because profile doesn\'t have all payoffs"""
    profiles = [
        [1, 3],
    ]
    payoffs = [
        [1, 2],
    ]
    game = paygame.game(4, 2, profiles, payoffs)
    red_game = dpr.reduce_game(game, 2)
    assert np.all(red_game.num_role_players == [2])
    assert np.all(red_game.profiles() == [[1, 1]])
    assert [1, 1] not in red_game  # incomplete profiles don't register


def test_dpr_names():
    """Test names for dpr game"""
    base = rsgame.empty(3, 2)
    game = paygame.game_names(
        ['role'], 3, [['a', 'b']], base.all_profiles(),
        np.zeros((base.num_all_profiles, base.num_strats)))
    redgame = dpr.reduce_game(game, 2)
    expected = paygame.game_names(
        ['role'], 2, [['a', 'b']], redgame.all_profiles(),
        np.zeros((redgame.num_all_profiles, base.num_strats)))
    assert redgame == expected


@pytest.mark.parametrize('keep_prob', [0.6, 0.9, 1])
@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([2], [1]),
    ([3], [1]),
    ([1], [2]),
    ([1, 2], [2, 1]),
    ([1, 4], [2, 1]),
    ([4, 8], [3, 2]),
])
@pytest.mark.parametrize('_', range(10))
def test_random_twins(players, strategies, keep_prob, _):
    """Test random twins reduction"""
    # Create game and reduction
    game = gamegen.game(players, strategies, keep_prob)

    # Try to reduce game
    red_game = tr.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    reduced_full_profiles = utils.axis_to_elem(
        tr.reduce_profiles(red_game, game.profiles()))
    reduced_profiles = utils.axis_to_elem(red_game.profiles())
    assert np.setdiff1d(reduced_profiles, reduced_full_profiles).size == 0, \
        "reduced game contained profiles it shouldn't have"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game. We need to remove partial profiles first
    full_profiles = utils.axis_to_elem(game.profiles())
    complete_profs = ~np.isnan(red_game.payoffs()).any(1)
    full_reduced_profiles = utils.axis_to_elem(
        tr.expand_profiles(game, red_game.profiles()[complete_profs]))
    assert np.setdiff1d(full_reduced_profiles, full_profiles).size == 0, \
        'full game did not have data for all profiles required of reduced'


def test_twins_names():
    """Test twins with names"""
    base = rsgame.empty(3, 2)
    game = paygame.game_names(
        ['role'], 3, [['a', 'b']], base.all_profiles(),
        np.zeros((base.num_all_profiles, base.num_strats)))
    redgame = tr.reduce_game(game)
    expected = paygame.game_names(
        ['role'], 2, [['a', 'b']], redgame.all_profiles(),
        np.zeros((redgame.num_all_profiles, base.num_strats)))
    assert redgame == expected


@pytest.mark.parametrize('keep_prob', [0.6, 0.9, 1])
@pytest.mark.parametrize('players,strategies,red_players', [
    ([1], [1], [1]),
    ([2], [1], [2]),
    ([4], [1], [2]),
    ([1], [2], [1]),
    ([1, 2], [2, 1], [1, 2]),
    ([1, 4], [2, 1], [1, 2]),
    ([4, 9], [3, 2], [2, 3]),
])
@pytest.mark.parametrize('_', range(10))
def test_random_hierarchical(keep_prob, players, strategies, red_players, _):
    """Test random hierarchical"""
    # Create game and reduction
    game = gamegen.game(players, strategies, keep_prob)
    assert (rsgame.empty(red_players, strategies) ==
            hr.reduce_game(rsgame.empty(players, strategies), red_players))

    # Try to reduce game
    red_game = hr.reduce_game(game, red_players)

    # Assert that reducing all profiles covers reduced game
    reduced_full_profiles = utils.axis_to_elem(
        hr.reduce_profiles(red_game, game.profiles()))
    reduced_profiles = utils.axis_to_elem(red_game.profiles())
    assert np.setxor1d(reduced_profiles, reduced_full_profiles).size == 0, \
        "reduced game contained profiles it shouldn't have"

    # Assert that all contributing profiles are in the expansion of the reduced
    # game. Since hr doesn't add any incomplete profiles, it can't produce any
    full_profiles = utils.axis_to_elem(game.profiles())
    full_reduced_profiles = utils.axis_to_elem(
        hr.expand_profiles(game, red_game.profiles()))
    assert np.setdiff1d(full_reduced_profiles, full_profiles).size == 0, \
        'full game did not have data for all profiles required of reduced'


def test_hierarchical_names():
    """Test hierarchical with names"""
    base = rsgame.empty(4, 2)
    game = paygame.game_names(
        ['role'], 4, [['a', 'b']], base.all_profiles(),
        np.zeros((base.num_all_profiles, base.num_strats)))
    redgame = hr.reduce_game(game, 2)
    expected = paygame.game_names(
        ['role'], 2, [['a', 'b']], redgame.all_profiles(),
        np.zeros((redgame.num_all_profiles, base.num_strats)))
    assert redgame == expected


@pytest.mark.parametrize('keep_prob', [0, 0.6, 1])
@pytest.mark.parametrize('game_desc', [
    ([1], [1]),
    ([2], [1]),
    ([3], [1]),
    ([1], [2]),
    ([1, 2], [2, 1]),
    ([1, 4], [2, 1]),
    ([4, 9], [3, 2]),
])
@pytest.mark.parametrize('_', range(10))
# Simple test of identity reduction
def test_random_identity(keep_prob, game_desc, _):
    """Test random identity"""
    players, strategies = game_desc
    # Create game and reduction
    game = gamegen.game(players, strategies, keep_prob)
    assert (paygame.game_copy(rsgame.empty(players, strategies)) ==
            ir.reduce_game(rsgame.empty(players, strategies)))

    # Try to reduce game
    red_game = ir.reduce_game(game)

    # Assert that reducing all profiles covers reduced game
    reduced_full_profiles = utils.axis_to_elem(
        ir.reduce_profiles(red_game, game.profiles()))
    reduced_profiles = utils.axis_to_elem(red_game.profiles())
    assert np.setxor1d(reduced_full_profiles, reduced_profiles).size == 0, \
        "reduced game didn't match full game"

    full_profiles = utils.axis_to_elem(game.profiles())
    full_reduced_profiles = utils.axis_to_elem(
        ir.expand_profiles(game, red_game.profiles()))
    assert np.setxor1d(full_profiles, full_reduced_profiles).size == 0, \
        'full game did not match reduced game'


def test_identity_names():
    """Test identity with names"""
    base = rsgame.empty(3, 2)
    game = paygame.game_names(
        ['role'], 3, [['a', 'b']], base.all_profiles(),
        np.zeros((base.num_all_profiles, base.num_strats)))
    redgame = ir.reduce_game(game)
    expected = paygame.game_names(
        ['role'], 3, [['a', 'b']], redgame.all_profiles(),
        np.zeros((redgame.num_all_profiles, base.num_strats)))
    assert redgame == expected


def test_expand_tie_breaking():
    """Test that standard expansion breaks ties appropriately"""

    def expand(prof, full):
        """Expand a profile"""
        return hr.expand_profiles(rsgame.empty(full, len(prof)), prof)

    full = expand([4, 2], 20)
    assert np.all(full == [13, 7]), \
        "Didn't tie break on approximation first"
    full = expand([1, 3], 10)
    assert np.all(full == [2, 8]), \
        "Didn't tie break on larger count if approximations same"
    full = expand([2, 2], 11)
    assert np.all(full == [6, 5]), \
        "Didn't tie break on strategy name if all else same"


def test_reduce_tie_breaking():
    """Test that the standard reduction breaks ties appropriately"""

    def reduce_prof(prof, red):
        """Reduce a profile"""
        return hr.reduce_profiles(rsgame.empty(red, len(prof)), prof)

    red = reduce_prof([13, 7], 6)
    assert np.all(red == [4, 2]), \
        "Didn't tie break on approximation first"
    # Expanded profile has no valid reduction
    red = reduce_prof([14, 6], 6)
    assert red.size == 0, "Didn't properly find no reduced profile"
    red = reduce_prof([2, 8], 4)
    assert np.all(red == [1, 3]), \
        "Didn't tie break on larger count if approximations same"
    red = reduce_prof([6, 5], 4)
    assert np.all(red == [2, 2]), \
        "Didn't tie break on strategy name if all else same"


@pytest.mark.parametrize('_', range(1000))
def test_random_hr_identity(_):
    """Test hierarchical identity"""
    num_strats = rand.randint(2, 20)
    profile = rand.randint(20, size=num_strats)[None]
    reduced_players = profile.sum()
    while reduced_players < 2:  # pragma: no cover
        profile[0, rand.randint(num_strats)] += 1
        reduced_players += 1
    full_players = rand.randint(reduced_players + 1, reduced_players ** 2)
    game = rsgame.empty(reduced_players, num_strats)
    full_profile = hr.expand_profiles(
        rsgame.empty(full_players, num_strats), profile)
    red_profile = hr.reduce_profiles(game, full_profile)
    assert np.all(red_profile == profile), "reduction didn't pass identity"


def test_approximate_dpr_expansion():
    """Test expansion on approximate dpr"""
    full_game = rsgame.empty([8, 11], [2, 2])
    red_prof = [[1, 2, 2, 2]]
    full_profs = dpr.expand_profiles(
        full_game, red_prof)
    profs = utils.axis_to_elem(full_profs)
    expected = utils.axis_to_elem([
        [4, 4, 6, 5],
        [1, 7, 6, 5],
        [3, 5, 4, 7],
        [3, 5, 7, 4]])
    assert np.setxor1d(profs, expected).size == 0, \
        'generated different profiles than expected'


def test_approximate_dpr_reduce_game():
    """Test approximate dpr game reduction"""
    game = gamegen.game([3, 4], 2)
    redgame = dpr.reduce_game(game, 2)
    # Pure strategies are reduced properly
    assert (redgame.get_payoffs([2, 0, 0, 2])[0]
            == game.get_payoffs([3, 0, 0, 4])[0])
    # Mixed strategies are reduced properly
    assert (redgame.get_payoffs([1, 1, 1, 1])[0]
            == game.get_payoffs([1, 2, 2, 2])[0])
    assert (redgame.get_payoffs([1, 1, 1, 1])[1]
            == game.get_payoffs([2, 1, 2, 2])[1])
    assert (redgame.get_payoffs([1, 1, 1, 1])[2]
            == game.get_payoffs([2, 1, 1, 3])[2])
    assert (redgame.get_payoffs([1, 1, 1, 1])[3]
            == game.get_payoffs([2, 1, 3, 1])[3])


@pytest.mark.parametrize('_', range(20))
@pytest.mark.parametrize('players,strategies', [
    ([1], [1]),
    ([2], [1]),
    ([3], [1]),
    ([1], [2]),
    ([1, 2], [2, 1]),
    ([1, 4], [2, 1]),
    ([4, 9], [3, 2]),
])
def test_random_approximate_dpr(players, strategies, _):
    """Test approximate dpr preserves completeness on random games"""
    game = gamegen.game(players, strategies)
    red_counts = 2 + (rand.random(game.num_roles) *
                      (game.num_role_players - 1)).astype(int)
    red_counts[game.num_role_players == 1] = 1

    # Try to reduce game
    red_game = dpr.reduce_game(game, red_counts)

    # Assert that reducing all profiles covers reduced game
    assert red_game.is_complete(), 'DPR did not preserve completeness'


def test_identity_dev_expansion():
    """Test that identity dev expansion is correct"""
    game = rsgame.empty([3, 4], [4, 3])
    mask = [True, False, True, False, False, True, False]
    profs = ir.expand_deviation_profiles(game, mask)
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [2, 1, 0, 0, 0, 4, 0],
        [1, 1, 1, 0, 0, 4, 0],
        [0, 1, 2, 0, 0, 4, 0],
        [2, 0, 0, 1, 0, 4, 0],
        [1, 0, 1, 1, 0, 4, 0],
        [0, 0, 2, 1, 0, 4, 0],
        [3, 0, 0, 0, 1, 3, 0],
        [2, 0, 1, 0, 1, 3, 0],
        [1, 0, 2, 0, 1, 3, 0],
        [0, 0, 3, 0, 1, 3, 0],
        [3, 0, 0, 0, 0, 3, 1],
        [2, 0, 1, 0, 0, 3, 1],
        [1, 0, 2, 0, 0, 3, 1],
        [0, 0, 3, 0, 0, 3, 1],
    ])
    assert np.setxor1d(actual, expected).size == 0

    profs = ir.expand_deviation_profiles(game, mask, role_index=0)
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [2, 1, 0, 0, 0, 4, 0],
        [1, 1, 1, 0, 0, 4, 0],
        [0, 1, 2, 0, 0, 4, 0],
        [2, 0, 0, 1, 0, 4, 0],
        [1, 0, 1, 1, 0, 4, 0],
        [0, 0, 2, 1, 0, 4, 0],
    ])
    assert np.setxor1d(actual, expected).size == 0


def test_hierarchical_dev_expansion():
    """Test that hierarchical dev expansion is correct"""
    game = rsgame.empty([9, 16], [4, 3])
    mask = [True, False, True, False, False, True, False]
    profs = hr.expand_deviation_profiles(game, mask, [3, 4])
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [6, 3, 0, 0, 0, 16, 0],
        [3, 3, 3, 0, 0, 16, 0],
        [0, 3, 6, 0, 0, 16, 0],
        [6, 0, 0, 3, 0, 16, 0],
        [3, 0, 3, 3, 0, 16, 0],
        [0, 0, 6, 3, 0, 16, 0],
        [9, 0, 0, 0, 4, 12, 0],
        [6, 0, 3, 0, 4, 12, 0],
        [3, 0, 6, 0, 4, 12, 0],
        [0, 0, 9, 0, 4, 12, 0],
        [9, 0, 0, 0, 0, 12, 4],
        [6, 0, 3, 0, 0, 12, 4],
        [3, 0, 6, 0, 0, 12, 4],
        [0, 0, 9, 0, 0, 12, 4],
    ])
    assert np.setxor1d(actual, expected).size == 0

    profs = hr.expand_deviation_profiles(game, mask, [3, 4], 0)
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [6, 3, 0, 0, 0, 16, 0],
        [3, 3, 3, 0, 0, 16, 0],
        [0, 3, 6, 0, 0, 16, 0],
        [6, 0, 0, 3, 0, 16, 0],
        [3, 0, 3, 3, 0, 16, 0],
        [0, 0, 6, 3, 0, 16, 0],
    ])
    assert np.setxor1d(actual, expected).size == 0


def test_dpr_dev_expansion():
    """Test that dpr dev expansion is correct

    Note, this is the only one that has "new" code, so it's the most important
    to test."""
    game = rsgame.empty([9, 16], [4, 3])
    mask = [True, False, True, False, False, True, False]
    profs = dpr.expand_deviation_profiles(game, mask, [3, 4])
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [8, 1, 0, 0, 0, 16, 0],
        [4, 1, 4, 0, 0, 16, 0],
        [0, 1, 8, 0, 0, 16, 0],
        [8, 0, 0, 1, 0, 16, 0],
        [4, 0, 4, 1, 0, 16, 0],
        [0, 0, 8, 1, 0, 16, 0],
        [9, 0, 0, 0, 1, 15, 0],
        [6, 0, 3, 0, 1, 15, 0],
        [3, 0, 6, 0, 1, 15, 0],
        [0, 0, 9, 0, 1, 15, 0],
        [9, 0, 0, 0, 0, 15, 1],
        [6, 0, 3, 0, 0, 15, 1],
        [3, 0, 6, 0, 0, 15, 1],
        [0, 0, 9, 0, 0, 15, 1],
    ])
    assert np.setxor1d(actual, expected).size == 0

    profs = dpr.expand_deviation_profiles(game, mask, [3, 4], 0)
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [8, 1, 0, 0, 0, 16, 0],
        [4, 1, 4, 0, 0, 16, 0],
        [0, 1, 8, 0, 0, 16, 0],
        [8, 0, 0, 1, 0, 16, 0],
        [4, 0, 4, 1, 0, 16, 0],
        [0, 0, 8, 1, 0, 16, 0],
    ])
    assert np.setxor1d(actual, expected).size == 0


def test_twins_dev_expansion():
    """Test that dpr dev expansion is correct

    Note, this is the only one that has "new" code, so it's the most important
    to test."""
    game = rsgame.empty([9, 16], [4, 3])
    mask = [True, False, True, False, False, True, False]
    profs = tr.expand_deviation_profiles(game, mask)
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [8, 1, 0, 0, 0, 16, 0],
        [0, 1, 8, 0, 0, 16, 0],
        [8, 0, 0, 1, 0, 16, 0],
        [0, 0, 8, 1, 0, 16, 0],
        [9, 0, 0, 0, 1, 15, 0],
        [5, 0, 4, 0, 1, 15, 0],
        [0, 0, 9, 0, 1, 15, 0],
        [9, 0, 0, 0, 0, 15, 1],
        [5, 0, 4, 0, 0, 15, 1],
        [0, 0, 9, 0, 0, 15, 1],
    ])
    assert np.setxor1d(actual, expected).size == 0

    profs = tr.expand_deviation_profiles(game, mask, role_index=0)
    actual = utils.axis_to_elem(profs)
    expected = utils.axis_to_elem([
        [8, 1, 0, 0, 0, 16, 0],
        [0, 1, 8, 0, 0, 16, 0],
        [8, 0, 0, 1, 0, 16, 0],
        [0, 0, 8, 1, 0, 16, 0],
    ])
    assert np.setxor1d(actual, expected).size == 0


@pytest.mark.parametrize('_', range(10))
@pytest.mark.parametrize('players,strategies,red_players', [
    ([1], [1], [1]),
    ([2], [1], [2]),
    ([3], [1], [2]),
    ([1], [2], [1]),
    ([1, 2], [2, 1], [1, 2]),
    ([1, 4], [2, 1], [1, 2]),
    ([4, 9], [3, 2], [2, 3]),
])
def test_rand_dpr_dev_expandion(players, strategies, red_players, _):
    """Test that dpr devs works for random games"""
    game = rsgame.empty(players, strategies)
    sup = (rand.random(game.num_roles) * game.num_role_strats).astype(int) + 1
    inds = np.concatenate(
        [rand.choice(s, x) + o for s, x, o
         in zip(game.num_role_strats, sup, game.role_starts)])
    mask = np.zeros(game.num_strats, bool)
    mask[inds] = True
    devs = dpr.expand_deviation_profiles(game, mask, red_players)
    assert np.all(np.sum(devs * ~mask, 1) == 1)

    for r_ind in range(game.num_roles):
        r_devs = dpr.expand_deviation_profiles(game, mask, red_players, r_ind)
        assert np.all(np.sum(r_devs * ~mask, 1) == 1)


def test_dpr_incomplete_profile():
    """Test that when allow_incomplete, we get appropriate payoffs"""
    profiles = [[4, 0, 0, 9],
                [1, 3, 9, 0],
                [2, 2, 9, 0]]
    payoffs = [[1, 0, 0, 2],
               [3, 4, 5, 0],
               [6, 7, 8, 0]]
    game = paygame.game([4, 9], 2, profiles, payoffs)
    red_game = dpr.reduce_game(game, [2, 3])
    actual = red_game.get_payoffs([2, 0, 0, 3])
    assert np.allclose(actual, [1, 0, 0, 2])
    actual = red_game.get_payoffs([1, 1, 3, 0])
    assert np.allclose(actual, [3, np.nan, 8, 0], equal_nan=True)


def test_remove_dpr_profiles_with_no_data():
    """Test that dpr removes profiles with no data"""
    profiles = [[1, 3],
                [3, 1]]
    payoffs = [[3, 4],
               [6, np.nan]]
    game = dpr.reduce_game(paygame.game(4, 2, profiles, payoffs), 2)
    assert game.num_profiles == 1

    profiles = [[1, 3],
                [3, 1]]
    payoffs = [[np.nan, 4],
               [6, np.nan]]
    game = dpr.reduce_game(paygame.game(4, 2, profiles, payoffs), 2)
    assert game.is_empty()

    profiles = [[1, 3]]
    payoffs = [[3, 4]]
    game = dpr.reduce_game(paygame.game(4, 2, profiles, payoffs), 2)
    assert game.num_profiles == 1

    profiles = [[1, 3]]
    payoffs = [[np.nan, 4]]
    game = dpr.reduce_game(paygame.game(4, 2, profiles, payoffs), 2)
    assert game.is_empty()
