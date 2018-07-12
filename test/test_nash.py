"""Test nash"""
import functools
from os import path

import numpy as np
import pytest

from gameanalysis import aggfn
from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import nash
from gameanalysis import regret
from gameanalysis import utils


@pytest.fixture(scope='session', name='hardgame')
def fix_hardgame():
    """Hard nash game"""
    with open(path.join('example_games', 'hard_nash.json')) as fil:
        return gamereader.load(fil)


def test_pure_prisoners_dilemma():
    """Test prisoners dilemma"""
    game = gamegen.prisoners_dilemma()
    eqa = nash.pure_equilibria(game)

    assert eqa.shape[0] == 1, "didn't find exactly one equilibria in pd"
    expected = [0, 2]
    assert np.all(expected == eqa), \
        "didn't find pd equilibrium"


def test_pure_roshambo():
    """Test roshambo"""
    game = gamegen.rock_paper_scissors()
    eqa = nash.pure_equilibria(game)
    assert eqa.size == 0, 'found a pure equilibrium in roshambo'
    eqa = nash.pure_equilibria(game, epsilon=1)
    assert eqa.shape[0] == 3, \
        "didn't find low regret ties in roshambo"
    eqa = nash.pure_equilibria(game, epsilon=2)
    assert eqa.shape[0] == game.num_all_profiles, \
        'found profiles with more than 2 regret in roshambo'


def test_min_regret_profile():
    """Test minimum regret profile on rps"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_profile(game)
    assert utils.allequal_perm(eqm, [0, 0, 2]), \
        'min regret profile was not rr, pp, or ss'


def test_replicator_dynamics_noop():
    """Test that max_iters stops replicator dynamics"""
    game = gamegen.sym_2p2s_known_eq(1 / np.sqrt(2))
    eqm = nash.replicator_dynamics(game, [1/2, 1/2], max_iters=0) # pylint: disable=unexpected-keyword-arg
    assert np.allclose(eqm, [1/2, 1/2])


def test_replicator_dynamics():
    """Test that it works for games we know it works for"""
    game = gamegen.sym_2p2s_known_eq(1 / np.sqrt(2))
    eqm = nash.replicator_dynamics(game, [1/2, 1/2])
    assert np.allclose(eqm, [1 / np.sqrt(2), 1 - 1 / np.sqrt(2)])


def test_replicator_dynamics_failure():
    """Test that it fails on divergent games"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.replicator_dynamics(game, [0.6, 0.3, 0.1])
    assert utils.allclose_perm(eqm, [1, 0, 0])


def test_regret_matching_noop():
    """Test that we can make fictitious play noop"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.regret_matching(game, [2, 0, 0], max_iters=0) # pylint: disable=unexpected-keyword-arg
    assert np.allclose(eqm, [1, 0, 0])


def test_regret_matching():
    """Test that it works for games we know it works for"""
    game = gamegen.rock_paper_scissors()
    eqm = nash._regret_matching_mix(game, [1/2, 0, 1/2]) # pylint: disable=protected-access
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=5e-2)


def test_regret_matching_failure():
    """Test that it works for games we know it works for"""
    game = gamegen.sym_2p2s_known_eq(1 / np.sqrt(2))
    eqm = nash.regret_matching(game, [0, 2])
    assert np.allclose(eqm, [1/2, 1/2], atol=1e-2)


def test_regret_minimize():
    """Test that regret minimize works"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.regret_minimize(game, [0.6, 0.3, 0.1])
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=1e-4)


def test_regret_minimize_failure():
    """Test that regret minimize fails"""
    table = [[-.7, -.9, -1.3, -2],
             [-.7, -.9, -1.1, -1.6],
             [-.2, -.4, -.8, -1.5]]
    game = aggfn.aggfn(3, 3, np.eye(3), np.eye(3, dtype=bool), table)
    eqm = nash.regret_minimize(game, [0.9, 0.05, 0.05])
    assert regret.mixture_regret(game, eqm) > 0.1


def test_fictitious_play_noop():
    """Test that we can make fictitious play noop"""
    game = gamegen.rock_paper_scissors()
    eqm = nash.fictitious_play(game, [0.6, 0.3, 0.1], max_iters=0) # pylint: disable=unexpected-keyword-arg
    assert np.allclose(eqm, [0.6, 0.3, 0.1])


def test_fictitious_play():
    """Test that fictitious play works"""
    game = gamegen.rock_paper_scissors(win=2)
    eqm = nash.fictitious_play(game, [0.6, 0.3, 0.1], max_iters=10000) # pylint: disable=unexpected-keyword-arg
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=1e-4)


def test_fictitious_play_convergence():
    """Test that fictitious play converges"""
    game = gamegen.rock_paper_scissors(win=2)
    eqm = nash.fictitious_play(game, [0.3, 0.4, 0.3], converge_thresh=1e-3) # pylint: disable=unexpected-keyword-arg
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=1e-3)


def test_fictitious_play_failure():
    """Test that fictitious play fails"""
    game = gamegen.rock_paper_scissors(win=0.5)
    eqm = nash.fictitious_play(game, [0.6, 0.3, 0.1], max_iters=10000) # pylint: disable=unexpected-keyword-arg
    assert regret.mixture_regret(game, eqm) > 0.1


def test_multiplicative_weights_dist():
    """Test that multiplicative weights dist works"""
    game = gamegen.rock_paper_scissors(win=2)
    eqm = nash.multiplicative_weights_dist(game, [0.6, 0.3, 0.1])
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=1e-3)


def test_multiplicative_weights_stoch():
    """Test that multiplicative weights stoch works"""
    game = gamegen.rock_paper_scissors(win=2)
    eqm = nash.multiplicative_weights_stoch(
        game, [0.6, 0.3, 0.1], max_iters=10000)
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=1e-2)


def test_multiplicative_weights_bandit():
    """Test that multiplicative weights bandit works"""
    game = gamegen.rock_paper_scissors(win=2)
    eqm = nash.multiplicative_weights_bandit(
        game, [0.6, 0.3, 0.1], max_iters=100000)
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=0.1)


def test_multiplicative_weights_failure():
    """Test that multiplicative weights fails"""
    game = gamegen.rock_paper_scissors(win=0.5)
    eqm = nash.multiplicative_weights_dist(game, [0.6, 0.3, 0.1])
    assert regret.mixture_regret(game, eqm) > 0.1


def test_scarf():
    """Test that scarfs algorithm works"""
    game = gamegen.rock_paper_scissors(win=0.5)
    eqm = nash.scarfs_algorithm(game, [0.6, 0.3, 0.1])
    assert np.allclose(eqm, [1/3, 1/3, 1/3], atol=1e-4)


def test_noop_and_serial_nash():
    """Test that no op works"""
    game = gamegen.rock_paper_scissors()
    req, eqm = nash._serial_nash_func(game, (nash._noop, [1/2, 0, 1/2], True)) # pylint: disable=protected-access
    assert np.allclose(eqm, [1/2, 0, 1/2])
    assert req


@pytest.mark.parametrize('func', [
    functools.partial(nash.replicator_dynamics, max_iters=1),
    functools.partial(nash._regret_matching_mix, max_iters=1), # pylint: disable=protected-access
    functools.partial(nash.regret_minimize, gtol=1e-2),
    functools.partial(nash.fictitious_play, max_iters=1),
    functools.partial(nash.multiplicative_weights_dist, max_iters=1),
    functools.partial(nash.multiplicative_weights_stoch, max_iters=1),
    functools.partial(nash.multiplicative_weights_bandit, max_iters=1),
    functools.partial(nash.scarfs_algorithm, timeout=2),
])
def test_multi_role(func):
    """Test that at least one iteration works on a multi role game"""
    game = gamegen.game([2, 3], [3, 2])
    mix = game.random_mixture()
    eqm = func(game, mix)
    assert game.is_mixture(eqm)


def test_mixed_equilibria():
    """Test that mixed equilibria works for easy case"""
    prob = 1 / np.sqrt(2)
    game = gamegen.sym_2p2s_known_eq(prob)
    eqa = nash.mixed_equilibria(game, processes=1)
    assert eqa.shape == (1, 2)
    eqm, = eqa
    assert np.allclose(eqm, [prob, 1 - prob], atol=1e-3)


def test_fast_failure(hardgame):
    """Test that fast fails to find an equilibrium"""
    eqa = nash.mixed_equilibria(hardgame, 'fast', processes=1)
    assert not eqa.size


def test_faststar_failure(hardgame):
    """Test that fast fails to find an equilibrium"""
    eqa = nash.mixed_equilibria(hardgame, 'fast*', processes=1)
    assert eqa.shape == (1, 9)
    reg = regret.mixture_regret(hardgame, eqa[0])
    assert reg > 1e-2


@pytest.mark.xfail(raises=TimeoutError)
@utils.timeout(5)
def test_one_timesout(hardgame):
    """Test that one works but we can't wait, so timeout"""
    nash.mixed_equilibria(hardgame, 'one', processes=1)


# FIXME Ideally remove timeout
@pytest.mark.xfail(raises=TimeoutError)
@utils.timeout(60)
def test_hard_nash(hardgame):
    """Test hard nash"""
    expected = [0.54074609, 0.45925391, 0, 0, 0, 1, 0, 0, 0]
    eqa = nash.mixed_equilibria(hardgame, processes=4)
    assert np.allclose(eqa, expected, atol=1e-4, rtol=1e-4), \
        "Didn't find equilibrium in known hard instance"


def test_hard_scarf():
    """A buggy instance of scarfs algorithm

    This triggered a discretization error with the fixed point algorithm
    immediately, e.g. a timeout of 2s is fine"""
    with open(path.join('example_games', 'hard_scarf.json')) as fil:
        game = gamereader.load(fil)
    eqm = nash.scarfs_algorithm(game, game.uniform_mixture(), timeout=5) # pylint: disable=unexpected-keyword-arg
    assert game.is_mixture(eqm)


def test_old_nash():
    """Test old nash functions appropriately"""
    prob = 1 / np.sqrt(2)
    game = gamegen.sym_2p2s_known_eq(prob)
    eqa = nash.mixed_nash(game, processes=2)
    assert eqa.shape == (1, 2)
    eqm, = eqa
    assert np.allclose(eqm, [prob, 1 - prob], atol=1e-3)


def test_old_nash_at_least_one():
    """Test old nash functions appropriately"""
    prob = 1 / np.sqrt(2)
    game = gamegen.sym_2p2s_known_eq(prob)
    eqa = nash.mixed_nash(game, replicator=dict(max_iters=0), at_least_one=True)
    assert eqa.shape == (1, 2)
    eqm, = eqa
    assert np.allclose(eqm, [prob, 1 - prob], atol=1e-3)


def test_old_nash_min_reg():
    """Test old nash functions appropriately"""
    prob = 1 / np.sqrt(2)
    game = gamegen.sym_2p2s_known_eq(prob)
    eqa = nash.mixed_nash(game, replicator=dict(max_iters=0), min_reg=True)
    assert eqa.shape == (1, 2)
    eqm, = eqa
    reg = regret.mixture_regret(game, eqm)
    assert reg > 1e-3


def test_old_nash_failure():
    """Test old nash functions appropriately"""
    prob = 1 / np.sqrt(2)
    game = gamegen.sym_2p2s_known_eq(prob)
    eqa = nash.mixed_nash(game, replicator=dict(max_iters=0))
    assert not eqa.size
