"""Test script"""
# pylint: disable=too-many-lines
import contextlib
import io
import json
import subprocess
import sys
import traceback
from os import path
from unittest import mock

import numpy as np
import pytest

from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import utils
from gameanalysis.reduction import deviation_preserving as dpr
from gameanalysis.reduction import hierarchical as hr
from gameanalysis.reduction import twins as tr
from gameanalysis import __main__ as main


def run(*args):
    """Run a command line and return if it ran successfully"""
    try:
        main.amain(*args)
    except SystemExit as ex:
        return not int(str(ex))
    except Exception: # pylint: disable=broad-except
        traceback.print_exc()
        return False
    return True


def stdin(inp):
    """Patch stdin with input"""
    return mock.patch.object(sys, 'stdin', io.StringIO(inp))


def stdout():
    """Patch stdout and return stringio"""
    return contextlib.redirect_stdout(io.StringIO())


def stderr():
    """Patch stderr and return stringio"""
    return contextlib.redirect_stderr(io.StringIO())


def array_set_equal(arr, brr):
    """Return true if two sets of arrays are equal"""
    return not np.setxor1d(
        utils.axis_to_elem(arr), utils.axis_to_elem(brr)).size


@pytest.fixture(scope='session', name='ggame_file')
def fix_ggame_file():
    """Gambit file name"""
    return path.join(
        path.dirname(path.realpath(__file__)), '..', 'example_games',
        'ugly.nfg')


@pytest.fixture(scope='session', name='ggame_str')
def fix_ggame_str(ggame_file):
    """Gambit string"""
    with open(ggame_file) as fil:
        return fil.read()


@pytest.fixture(scope='session', name='game')
def fix_game():
    """Get a standard game"""
    return gamegen.game([3, 2], [2, 3])


@pytest.fixture(scope='session', name='game_json')
def fix_game_json(game):
    """Get the json structure for a game"""
    return game.to_json()


@pytest.fixture(scope='session', name='game_str')
def fix_game_str(game_json):
    """Get the json string for a game"""
    return json.dumps(game_json)


@pytest.fixture(scope='session', name='game_file')
def fix_game_file(game_str, tmpdir_factory):
    """Create info for handling a data game"""
    game_file = str(tmpdir_factory.mktemp('games').join('game.json'))
    with open(game_file, 'w') as fil:
        fil.write(game_str)
    return game_file


@pytest.fixture(scope='session', name='sgame')
def fix_sgame():
    """Create a sample game"""
    return gamegen.samplegame([2, 3], [4, 3], 0.05)


@pytest.fixture(scope='session', name='sgame_str')
def fix_sgame_str(sgame):
    """Sample game string"""
    return json.dumps(sgame.to_json())


@pytest.fixture(scope='session', name='sgame_file')
def fix_sgame_file(sgame_str, tmpdir_factory):
    """Sample game file"""
    sgame_file = str(tmpdir_factory.mktemp('games').join('sgame.json'))
    with open(sgame_file, 'w') as fil:
        fil.write(sgame_str)
    return sgame_file


@pytest.fixture(scope='session', name='hardgame_file')
def fix_hardgame_file():
    """Get the file for a hard nash game"""
    return path.join(
        path.dirname(path.realpath(__file__)), '..', 'example_games',
        'hard_nash.json')


@pytest.fixture(scope='session', name='hardgame_str')
def fix_hardgame_str(hardgame_file):
    """Get the string of a game with a hard nash equilibria"""
    with open(hardgame_file) as fil:
        return fil.read()


@pytest.fixture(scope='session', name='hardgame')
def fix_hardgame(hardgame_str):
    """Get a game with hard nash equilibria"""
    return gamereader.loads(hardgame_str)


@pytest.fixture(scope='session', name='hardprof')
def fix_hardprof():
    """Fixture for a hard profile"""
    return {
        'background': {
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
        'hft': {'noop': 1}}


@pytest.fixture(scope='session', name='hardprof_str')
def fix_hardprof_str(hardprof):
    """Fixture for a hard profile string"""
    return json.dumps(hardprof)


@pytest.fixture(scope='session', name='hardprof_file')
def fix_hardprof_file(hardprof_str, tmpdir_factory):
    """Fixture for a hard profile file"""
    prof_file = str(tmpdir_factory.mktemp('profs').join('hardprof.json'))
    with open(prof_file, 'w') as fil:
        fil.write(hardprof_str)
    return prof_file


@pytest.fixture(scope='session', name='hardmix')
def fix_hardmix():
    """Fixture for a hard mixture"""
    return {
        'background': {
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 1},
        'hft': {'noop': 1}}


@pytest.fixture(scope='session', name='hardmix_str')
def fix_hardmix_str(hardmix):
    """Fixture for a hard profile string"""
    return json.dumps(hardmix)


@pytest.fixture(scope='session', name='hardmix_file')
def fix_hardmix_file(hardmix_str, tmpdir_factory):
    """Fixture for a hard profile file"""
    mix_file = str(tmpdir_factory.mktemp('profs').join('hardmix.json'))
    with open(mix_file, 'w') as fil:
        fil.write(hardmix_str)
    return mix_file


@pytest.fixture(scope='session', name='mgame')
def fix_mgame():
    """Create a matrix game"""
    return gamegen.independent_game([2, 3])


@pytest.fixture(scope='session', name='mgame_str')
def fix_mgame_str(mgame):
    """Create info for handling a data game"""
    return json.dumps(mgame.to_json())


@pytest.fixture(scope='session', name='mgame_file')
def fix_mgame_file(mgame_str, tmpdir_factory):
    """Create info for handling a data game"""
    mgame_file = str(tmpdir_factory.mktemp('games').join('matgame.json'))
    with open(mgame_file, 'w') as fil:
        fil.write(mgame_str)
    return mgame_file


def test_from_module():
    """Test from module"""
    python = path.join(
        path.dirname(path.realpath(__file__)), '..', 'bin', 'python')
    proc = subprocess.run([python, '-m', 'gameanalysis', '--help'])
    assert not proc.returncode


def test_help():
    """Test help"""
    assert not run()
    assert not run('--fail')
    with stdout() as out, stderr() as err:
        assert run('--help'), err.getvalue()
    for cmd in (line.split()[0] for line in out.getvalue().split('\n')
                if line.startswith('    ') and line[4] != ' '):
        with stderr() as err:
            assert run(cmd, '--help'), err.getvalue()


def test_dominance_game(game, game_file):
    """Test basic dominance"""
    with stdout() as out, stderr() as err:
        assert run('dom', '-i', game_file), err.getvalue()
    game_dom = gamereader.loads(out.getvalue())
    assert game_dom == game


def test_dominance_strats(game_json, game_file):
    """Test dominance outputs strats"""
    with stdout() as out, stderr() as err:
        assert run('dom', '-i', game_file, '-s'), err.getvalue()
    assert json.loads(out.getvalue()) == game_json['strategies']


def test_dominance_weakdom(game_str):
    """Test weak dominance and stdin"""
    with stdin(game_str), stderr() as err:
        assert run('dom', '-cweakdom', '-o/dev/null'), err.getvalue()


def test_dominance_strictdom(game_file):
    """Test strict dominance"""
    with stdout() as out, stderr() as err:
        assert run('dom', '-cstrictdom', '-i', game_file), err.getvalue()
    gamereader.loads(out.getvalue())


def test_dominance_never_best_response(game_file):
    """Test never best response"""
    with stdout() as out, stderr() as err:
        assert run('dom', '-cneverbr', '-i', game_file), err.getvalue()
    gamereader.loads(out.getvalue())


def test_dominance_matgame(mgame_file):
    """Test dom works for other game types"""
    with stderr() as err:
        assert run(
            'dom', '-cweakdom', '-o/dev/null', '-i',
            mgame_file), err.getvalue()


def test_gamegen_fail():
    """Test gamegen"""
    assert not run('gen')
    assert not run('gen', 'ursym')


def test_gamegen_uzs():
    """Test uniform zero sum gamegen"""
    with stderr() as err:
        assert run('gen', 'uzs', '6', '-n', '-o/dev/null'), err.getvalue()


def test_gamegen_ursym():
    """Test uniform role symmetric gamegen"""
    with stdout() as out, stderr() as err:
        assert run('gen', 'ursym', '3:4,4:3'), err.getvalue()
    gamereader.loads(out.getvalue())


def test_gamegen_noise_uniform(game_str):
    """Test uniform noise gamegen"""
    with stdin(game_str), stderr() as err, stdout() as out:
        assert run(
            'gen', 'noise', '-d', 'uniform', '-w', '1.5', '-s',
            '5'), err.getvalue()
    gamereader.loads(out.getvalue())


def test_gamegen_noise_gumbel(game_file):
    """Test gumbel noise gamegen"""
    with stderr() as err, stdout() as out:
        assert run(
            'gen', 'noise', '-d', 'gumbel', '-w', '1.5', '-s', '5', '-i',
            game_file), err.getvalue()
    gamereader.loads(out.getvalue())


def test_gamegen_noise_bimodal(game_file):
    """Test bimodal noise gamegen"""
    with stderr() as err, stdout() as out:
        assert run(
            'gen', 'noise', '-d', 'bimodal', '-w', '1.5', '-s', '5', '-i',
            game_file), err.getvalue()
    gamereader.loads(out.getvalue())


def test_gamegen_noise_gaussian(game_file):
    """Test gamegen with gaussian noise"""
    with stderr() as err, stdout() as out:
        assert run(
            'gen', 'noise', '-d', 'gaussian', '-w', '1.5', '-s', '5', '-i',
            game_file), err.getvalue()
    gamereader.loads(out.getvalue())


def test_nash_fail(game_file):
    """Test nash fail"""
    assert not run('nash', '-tfail', '-i', game_file)


def test_nash_basic(game_str):
    """Test basic nash works"""
    with stdin(game_str), stderr() as err:
        assert run('nash'), err.getvalue()


def test_nash_options(game_file):
    """Test nash options"""
    with stderr() as err:
        assert run(
            'nash', '-i', game_file, '-o/dev/null', '-r1e-2', '-d1e-2',
            '-c1e-7', '-x100', '-s1e-2', '-m5', '-n', '-p1'), err.getvalue()


def test_nash_pure(hardgame, hardgame_file):
    """Test pure nash"""
    with stdout() as out, stderr() as err:
        assert run('nash', '-tpure', '-i', hardgame_file), err.getvalue()
    assert any(  # pragma: no branch
        np.all(hardgame.profile_from_json(prof) ==
               [4, 2, 0, 0, 0, 1, 0, 0, 0])
        for prof in json.loads(out.getvalue()))


def test_nash_prof(hardgame, hardgame_file):
    """Test nash prof"""
    with stdout() as out, stderr() as err:
        assert run(
            'nash', '-tmin-reg-prof', '-i', hardgame_file), err.getvalue()
    assert any(  # pragma: no branch
        np.all(hardgame.profile_from_json(prof) ==
               [4, 2, 0, 0, 0, 1, 0, 0, 0])
        for prof in json.loads(out.getvalue()))


def test_nash_grid(hardgame, hardgame_file):
    """Test nash grid"""
    with stdout() as out, stderr() as err:
        assert run(
            'nash', '-tmin-reg-grid', '-i', hardgame_file), err.getvalue()
    assert any(  # pragma: no branch
        np.allclose(hardgame.mixture_from_json(mix),
                    [0, 1, 0, 0, 0, 1, 0, 0, 0])
        for mix in json.loads(out.getvalue()))


def test_nash_pure_one():
    """Test nash with at_least_one"""
    sgame = gamegen.rock_paper_scissors()
    sgame_str = json.dumps(sgame.to_json())
    with stdin(sgame_str), stderr() as err:
        assert run('nash', '-tpure', '--one'), err.getvalue()


def test_nash_mat(mgame_file):
    """Test nash works with other games"""
    with stderr() as err:
        assert run('nash', '-o/dev/null', '-i', mgame_file), err.getvalue()


def test_payoff_pure(hardgame_file, hardprof_str):
    """Test payoff pure"""
    with stdin(hardprof_str), stderr() as err:
        assert run(
            'pay', '-i', hardgame_file, '-', '-o/dev/null'), err.getvalue()


def test_payoff_pure_welfare(hardgame_str, hardprof_file):
    """Test welfare payoff pure"""
    with stdin(hardgame_str), stdout() as out, stderr() as err:
        assert run(
            'pay', hardprof_file, '-twelfare'), err.getvalue()
    assert np.isclose(json.loads(out.getvalue())[0], -315.4034577992763)


def test_payoff_mixed(hardgame_file, hardmix_str):
    """Test mixed payoff"""
    with stdin(hardmix_str), stderr() as err:
        assert run(
            'pay', '-i', hardgame_file, '-', '-o/dev/null'), err.getvalue()


def test_payoff_mixed_welfare(hardgame_str, hardmix_file):
    """Test mixed welfare"""
    with stdin(hardgame_str), stdout() as out, stderr() as err:
        assert run(
            'pay', hardmix_file, '-twelfare'), err.getvalue()
        assert np.isclose(json.loads(out.getvalue())[0], -315.4034577992763)


def test_payoff_pure_single(hardgame, hardgame_file, hardprof_file):
    """Test payoff pure single"""
    with stdout() as out, stderr() as err:
        assert run('pay', '-i', hardgame_file, hardprof_file), err.getvalue()
    pay = hardgame.payoff_from_json(json.loads(out.getvalue())[0])
    assert np.allclose(pay, [0, -52.56724296654605, 0, 0, 0, 0, 0, 0, 0])


def test_payoff_pure_string(hardgame, hardgame_file, hardprof_str):
    """Test payoff pure string"""
    with stdout() as out, stderr() as err:
        assert run('pay', '-i', hardgame_file, hardprof_str), err.getvalue()
    pay = hardgame.payoff_from_json(json.loads(out.getvalue())[0])
    assert np.allclose(pay, [0, -52.56724296654605, 0, 0, 0, 0, 0, 0, 0])


def test_reduction_basic(hardgame, hardgame_str):
    """Test basic reduction"""
    with stdin(hardgame_str), stdout() as out, stderr() as err:
        assert run('red', 'background:2;hft:1'), err.getvalue()
    game = gamereader.loads(out.getvalue())
    assert game == dpr.reduce_game(hardgame, [2, 1])


def test_reduction_hierarchical(game, game_file):
    """Test hierarchical reduction"""
    with stdout() as out, stderr() as err:
        assert run('red', '-thr', '-s', '2,1', '-i', game_file), err.getvalue()
    ogame = gamereader.loads(out.getvalue())
    assert ogame == hr.reduce_game(game, [2, 1])


def test_reduction_twins(game, game_file):
    """Test twins reduction"""
    with stdout() as out, stderr() as err:
        assert run('red', '-ttr', '-i', game_file), err.getvalue()
    ogame = gamereader.loads(out.getvalue())
    assert ogame == tr.reduce_game(game)


def test_reduction_identity(game, game_file):
    """Test identity reduction"""
    with stdout() as out, stderr() as err:
        assert run('red', '-tidr', '-i', game_file), err.getvalue()
    ogame = gamereader.loads(out.getvalue())
    assert ogame == game


def test_reduction_6(mgame_file):
    """Test that reduction works for other games"""
    with stdout() as out, stderr() as err:
        assert run('red', '-tidr', '-i', mgame_file), err.getvalue()
    gamereader.loads(out.getvalue())


def test_regret_pure(hardgame_file, hardprof_file):
    """Test regret of pure profile"""
    with stdout() as out, stderr() as err:
        assert run('reg', '-i', hardgame_file, hardprof_file), err.getvalue()
    assert np.isclose(json.loads(out.getvalue())[0], 7747.618428)


def test_regret_pure_gains(hardgame, hardgame_str, hardprof, hardprof_file):
    """Test gains of pure profile"""
    with stdin(hardgame_str), stdout() as out, stderr() as err:
        assert run('reg', hardprof_file, '-tgains'), err.getvalue()
    dev_pay = json.loads(out.getvalue())[0]
    for role, strats in dev_pay.items():
        prof_strats = hardprof[role]
        assert prof_strats.keys() == strats.keys()
        role_strats = set(hardgame.strat_names[hardgame.role_index(role)])
        for strat, dev_strats in strats.items():
            rstrats = role_strats.copy()
            rstrats.remove(strat)
            assert rstrats == dev_strats.keys()


def test_regret_mixed(hardgame_file, hardmix_file):
    """Test mixture regret"""
    with stdout() as out, stderr() as err:
        assert run('reg', '-i', hardgame_file, hardmix_file), err.getvalue()
    assert np.isclose(json.loads(out.getvalue())[0], 7747.618428)


def test_regret_mixed_gains(hardgame, hardgame_file, hardmix_file):
    """Test mixture regret"""
    with stdout() as out, stderr() as err:
        assert run(
            'reg', hardmix_file, '-tgains', '-i',
            hardgame_file), err.getvalue()
    assert np.allclose(
        hardgame.payoff_from_json(json.loads(out.getvalue())[0]),
        [581.18996992, 0., 0., 4696.19261, 3716.207196, 7747.618428,
         4569.842172, 4191.665254, 4353.146694])


def test_regret_single(hardgame_file, hardprof_file):
    """Test regret of single profile"""
    with stdout() as out, stderr() as err:
        assert run('reg', '-i', hardgame_file, hardprof_file), err.getvalue()
    assert np.isclose(json.loads(out.getvalue())[0], 7747.618428)


def test_restriction_detect(hardgame, hardgame_file):
    """Test detect maximal restrictions"""
    with stdout() as out, stderr() as err:
        assert run('rest', '-nd', '-i', hardgame_file), err.getvalue()
    assert hardgame.restriction_from_json(json.loads(out.getvalue())[0]).all()


def test_restriction_extract_string(hardgame, hardgame_file):
    """Test restriction extraction with a string"""
    with stdout() as out, stderr() as err:
        assert run(
            'rest', '-n', '-t',
            'background:markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9;'
            'hft:noop', '-s', '0,3,4', '-i', hardgame_file), err.getvalue()

    expected = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0]], bool)
    assert array_set_equal(expected, [  # pragma: no branch
        hardgame.restriction_from_json(s)
        for s in json.loads(out.getvalue())])


def test_restriction_extract_file(game, game_file):
    """Test restriction extraction"""
    rest = game.random_restriction()
    rest_str = json.dumps([game.restriction_to_json(rest)])
    with stdin(rest_str), stdout() as out, stderr() as err:
        assert run('rest', '-i', game_file, '-f', '-'), err.getvalue()
    rgame = gamereader.loadj(json.loads(out.getvalue())[0])
    assert rgame == game.restrict(rest)


def test_analysis_output(game_str):
    """Test analysis"""
    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('analyze'), err.getvalue()
    out = out.getvalue()
    start = """Game Analysis
=============
Game:
    Roles: r0, r1
    Players:
        3x r0
        2x r1
    Strategies:
        r0:
            s0
            s1
        r1:
            s2
            s3
            s4
payoff data for 24 out of 24 profiles"""
    assert out.startswith(start)
    assert 'Social Welfare\n--------------' in out
    assert 'Maximum social welfare profile:' in out
    assert 'Maximum "r0" welfare profile:' in out
    assert 'Maximum "r1" welfare profile:' in out
    assert 'Equilibria\n----------' in out
    assert 'No-equilibria Subgames\n----------------------' in out
    assert ('Unconfirmed Candidate Equilibria\n'
            '--------------------------------') in out
    assert ('Unexplored Best-response Subgames\n'
            '---------------------------------') in out
    assert 'Json Data\n=========' in out


def test_analysis_dpr(game_file):
    """Test analysis with dpr"""
    with stdout() as out, stderr() as err:
        assert run(
            'analyze', '-i', game_file, '--restrictions', '--dominance',
            '--dpr', 'r0:3;r1:2', '-p1', '--dist-thresh', '1e-3', '-r1e-3',
            '-t1e-3', '--rand-restarts', '0', '-m10000',
            '-c1e-8'), err.get_value()
    assert 'With deviation preserving reduction: r0:3 r1:2' in out.getvalue()


def test_analysis_hr(game_file):
    """Test analysis with hr"""
    with stdout() as out, stderr() as err:
        assert run(
            'analyze', '-i', game_file, '--hr', 'r0:3;r1:2',
            '-p1'), err.getvalue()
    assert 'With hierarchical reduction: r0:3 r1:2' in out.getvalue()


def test_analysis_equilibria():
    """Test analysis with equilibria"""
    profiles = [
        # Complete deviations but unexplored
        [4, 0, 0, 0, 0],
        [3, 1, 0, 0, 0],
        [3, 0, 1, 0, 0],
        [3, 0, 0, 1, 0],
        [3, 0, 0, 0, 1],
        # Deviating restriction also explored
        [0, 4, 0, 0, 0],
        [0, 3, 1, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 1, 3, 0, 0],
        [0, 0, 4, 0, 0],
        # Deviations
        [1, 3, 0, 0, 0],
        [1, 2, 1, 0, 0],
        [1, 1, 2, 0, 0],
        [1, 0, 3, 0, 0],
        [0, 3, 0, 1, 0],
        [0, 2, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 3, 1, 0],
        [0, 3, 0, 0, 1],
        [0, 2, 1, 0, 1],
        [0, 1, 2, 0, 1],
        [0, 0, 3, 0, 1],
        # Deviating restriction
        [0, 2, 0, 2, 0],
        [0, 1, 0, 3, 0],
        [0, 0, 0, 4, 0],
    ]
    payoffs = [
        # Complete deviations but unexplored
        [4, 0, 0, 0, 0],
        [4, 1, 0, 0, 0],
        [4, 0, 1, 0, 0],
        [4, 0, 0, 1, 0],
        [4, 0, 0, 0, 0],
        # Deviating restriction also explored
        [0, 1, 0, 0, 0],
        [0, 1, 4, 0, 0],
        [0, 1, 4, 0, 0],
        [0, 1, 4, 0, 0],
        [0, 0, 4, 0, 0],
        # Deviations
        [1, 3, 0, 0, 0],
        [1, 2, 1, 0, 0],
        [1, 1, 2, 0, 0],
        [1, 0, 3, 0, 0],
        [0, 3, 0, 5, 0],
        [0, 2, 1, 5, 0],
        [0, 1, 2, 5, 0],
        [0, 0, 3, 5, 0],
        [0, 3, 0, 0, 0],
        [0, 2, 1, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 3, 0, 0],
        # Deviating restriction
        [0, 2, 0, 2, 0],
        [0, 1, 0, 3, 0],
        [0, 0, 0, 4, 0],
    ]
    game = paygame.game([4], [5], profiles, payoffs)
    game_str = json.dumps(game.to_json())

    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('analyze', '-sd'), err.getvalue()
    out = out.getvalue()
    assert 'Found 1 dominated strategy' in out
    assert 'Found 1 unconfirmed candidate' in out
    assert 'Found 1 unexplored best-response restricted game' in out


def test_analysis_dup_equilibria():
    """Test analysis dpr equilibria"""
    # Two restrictions, but dominated, so identical equilibria
    profiles = [
        [2, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 2, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 2, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 2],
    ]
    payoffs = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    game = paygame.game(2, 4, profiles, payoffs)
    game_str = json.dumps(game.to_json())

    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('analyze', '-s'), err.getvalue()
    assert 'Found 2 maximal complete restricted games' in out.getvalue()


def test_analysis_dev_explored():
    """Test analysis deviations explored"""
    # Beneficial deviation to an already explored restriction
    profiles = [
        [2, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 2, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 2, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 2],
    ]
    payoffs = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    game = paygame.game(2, 4, profiles, payoffs)
    game_str = json.dumps(game.to_json())

    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('analyze', '-s'), err.getvalue()
    assert ('Found no unexplored best-response restricted games'
            in out.getvalue())


def test_analysis_no_data():
    """Test analysis on empty game"""
    game = paygame.game([2], [2], [[1, 1]], [[5, float('nan')]])
    game_str = json.dumps(game.to_json())

    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('analyze', '-s'), err.getvalue()
    out = out.getvalue()
    assert 'There was no profile with complete payoff data' in out
    assert 'Found no complete restricted games' in out


def test_analysis_no_eqa(game_file):
    """Test analysis with no equilibria"""
    with stdout() as out, stderr() as err:
        assert run(
            'analyze', '-i', game_file, '--restrictions', '--dominance',
            '--dpr', 'r0:3;r1:2', '-p1', '-r0', '-m0'), err.getvalue()
    out = out.getvalue()
    assert 'Found no equilibria' in out
    assert 'Found 1 no-equilibria restricted game' in out


def test_learning_output(game_str):
    """Test learning output"""
    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('learning'), err.getvalue()
    out = out.getvalue()
    start = """Game Learning
=============
RbfGpGame:
    Roles: r0, r1
    Players:
        3x r0
        2x r1
    Strategies:
        r0:
            s0
            s1
        r1:
            s2
            s3
            s4
"""
    assert out.startswith(start)
    assert 'Equilibria\n----------' in out
    assert 'Json Data\n=========' in out


def test_learning_args(game_file):
    """Test learning options"""
    with stderr() as err:
        assert run(
            'learning', '-i', game_file, '-o/dev/null', '-p1', '--dist-thresh',
            '1e-3', '-r1e-3', '-t1e-3', '--rand-restarts', '0', '-m10000',
            '-c1e-8'), err.getvalue()


def test_learning_no_eqa():
    """Test learning with no equilibria"""
    game = gamegen.congestion(10, 3, 1)
    game_str = json.dumps(game.to_json())
    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('learning', '-m0', '-r0'), err.getvalue()
    assert 'Found no equilibria' in out.getvalue()


def test_boot_basic(sgame, sgame_file):
    """Test bootstrap"""
    profs = [sgame.profile_to_json(sgame.uniform_mixture())]
    profs_str = json.dumps(profs)
    with stdin(profs_str), stderr() as err:
        assert run('boot', '-i', sgame_file, '-', '-o/dev/null'), err.getvalue()


def test_boot_keys_perc(sgame, sgame_file):
    """Test bootstrap"""
    profs = [sgame.mixture_to_json(sgame.random_mixture())]
    profs_str = json.dumps(profs)
    with stdin(profs_str), stdout() as out, stderr() as err:
        assert run(
            'boot', '-i', sgame_file, '-', '-tsurplus', '--processes', '1',
            '-n21', '-p', '5', '-p', '95'), err.getvalue()
    data = json.loads(out.getvalue())
    assert all(j.keys() == {'5', '95', 'mean'} for j in data)
    assert all(j['5'] <= j['95'] for j in data)


def test_boot_keys(sgame, sgame_file):
    """Test bootstrap"""
    profs = [sgame.mixture_to_json(sgame.random_mixture())]
    profs_str = json.dumps(profs)
    with stdin(profs_str), stdout() as out, stderr() as err:
        assert run(
            'boot', '-i', sgame_file, '-', '-tsurplus', '--processes', '1',
            '-n21'), err.getvalue()
    data = json.loads(out.getvalue())
    expected = {'mean'}.union(set(map(str, range(0, 101, 5))))
    assert all(j.keys() == expected for j in data)
    assert all(j['5'] <= j['95'] for j in data)


def test_samp_restriction(hardgame, hardgame_file):
    """Test sample restriction"""
    with stdout() as out, stderr() as err:
        assert run(
            'samp', '-i', hardgame_file, 'restriction', '-p',
            '0.5'), err.getvalue()
    hardgame.restriction_from_json(json.loads(out.getvalue()))


def test_samp_mix(hardgame, hardgame_file):
    """Test sample mixture"""
    with stdout() as out, stderr() as err:
        assert run(
            'samp', '-i', hardgame_file, 'mix', '-a', '0.5'), err.getvalue()
    hardgame.mixture_from_json(json.loads(out.getvalue()))


def test_samp_sparse_mix(hardgame, hardgame_file):
    """Test sample sparse mixture"""
    with stdout() as out, stderr() as err:
        assert run(
            'samp', '-i', hardgame_file, 'mix', '-a', '0.5',
            '-s'), err.getvalue()
    hardgame.mixture_from_json(json.loads(out.getvalue()))


def test_samp_sparse_mix_prob(hardgame, hardgame_file):
    """Test sample sparse mixture probability"""
    with stdout() as out, stderr() as err:
        assert run('samp', '-i', hardgame_file, 'mix', '-a', '0.5', '-s',
                   '0.5'), err.getvalue()
    hardgame.mixture_from_json(json.loads(out.getvalue()))


def test_samp_prof(hardgame, hardgame_file):
    """Test sample profile"""
    with stdout() as out, stderr() as err:
        assert run('samp', '-i', hardgame_file, 'prof'), err.getvalue()
    hardgame.profile_from_json(json.loads(out.getvalue()))


def test_samp_prof_alpha(hardgame, hardgame_file):
    """Test sample profile alpha"""
    with stdout() as out, stderr() as err:
        assert run(
            'samp', '-i', hardgame_file, 'prof', '-a', '0.5'), err.getvalue()
    hardgame.profile_from_json(json.loads(out.getvalue()))


def test_samp_prof_mix(hardgame, hardgame_file, hardmix_file):
    """Test sample profile mixture"""
    with stdout() as out, stderr() as err:
        assert run(
            'samp', '-i', hardgame_file, 'prof', '-m',
            hardmix_file), err.getvalue()
    hardgame.profile_from_json(json.loads(out.getvalue()))


def test_samp_prof_error(hardgame_file, hardmix_file):
    """Test sample profile error"""
    assert not run(
        'samp', '-i', hardgame_file, 'prof', '-a', '0.5', '-m', hardmix_file)


def test_samp_seed(hardgame, hardgame_file):
    """Test sample seed"""
    prof = {
        'background': {
            'markov:rmin_30000_rmax_30000_thresh_0.001_priceVarEst_1e6': 0.5,
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 0.5},
        'hft': {'noop': 1}}
    prof_str = json.dumps(prof)

    with stdin(prof_str), stdout() as out1, stderr() as err1:
        assert run(
            'samp', '-i', hardgame_file, '-n', '100', '--seed', '1234', 'prof',
            '-m', '-'), err1.getvalue()
    for line in out1.getvalue()[:-1].split('\n'):
        hardgame.profile_from_json(json.loads(line))

    # Setting seed produces identical output
    with stdin(prof_str), stdout() as out2, stderr() as err2:
        assert run(
            'samp', '-i', hardgame_file, '-n', '100', '--seed', '1234', 'prof',
            '-m', '-'), err2.getvalue()
    assert out1.getvalue() == out2.getvalue()

    # Not setting it causes failure
    # This can technically fail, but the probability is very small
    with stdin(prof_str), stdout() as out3, stderr() as err3:
        assert run(
            'samp', '-i', hardgame_file, '-n', '100', 'prof', '-m',
            '-'), err3.getvalue()
    assert out1.getvalue() != out3.getvalue()


def test_conv_game_empty(game_str):
    """Test convert empty game"""
    with stdin(game_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'empty'), err.getvalue()


def test_conv_game_game(game_str):
    """Test convert game"""
    with stdin(game_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'game'), err.getvalue()


def test_conv_game_sgame(game_str):
    """Test convert game to sampel game"""
    with stdin(game_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'samp'), err.getvalue()


def test_conv_game_mat(game_str):
    """Test convert game to matrix game"""
    with stdin(game_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'mat'), err.getvalue()


def test_conv_game_str(game_str):
    """Test convert game to string"""
    with stdin(game_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'str'), err.getvalue()


def test_conv_game_gambit(game_str):
    """"Test convert game to gambit"""
    with stdin(game_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'gambit'), err.getvalue()


def test_conv_game_norm(game_str):
    """Test game to normalized version"""
    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('conv', 'norm'), err.getvalue()
    game = gamereader.loads(out.getvalue())
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.all(np.isclose(game.max_role_payoffs(), 1) |
                  np.isclose(game.max_role_payoffs(), 0))


def test_conv_mat_empty(mgame_str):
    """Test convert metrix to empty"""
    with stdin(mgame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'empty'), err.getvalue()


def test_conv_mat_game(mgame_str):
    """Test convert matrix to game"""
    with stdin(mgame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'game'), err.getvalue()


def test_conv_mat_sgame(mgame_str):
    """Test convert matrix to sample game"""
    with stdin(mgame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'samp'), err.getvalue()


def test_conv_mat_mat(mgame_str):
    """Test convert matrix to itself"""
    with stdin(mgame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'mat'), err.getvalue()


def test_conv_mat_str(mgame_str):
    """test convert matrix to string"""
    with stdin(mgame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'str'), err.getvalue()


def test_conv_mat_gambit(mgame_str):
    """Test convert matrix to gambit"""
    with stdin(mgame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'gambit'), err.getvalue()


@pytest.mark.filterwarnings('ignore:gambit player names')
@pytest.mark.filterwarnings('ignore:gambit strategy names')
def test_conv_gambit_mat(ggame_str):
    """Test convert gambit to matrix"""
    with stdin(ggame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'mat'), err.getvalue()


@pytest.mark.filterwarnings('ignore:gambit player names')
@pytest.mark.filterwarnings('ignore:gambit strategy names')
def test_conv_gambit_game(ggame_str):
    """Test convert gambit to game"""
    with stdin(ggame_str), stderr() as err:
        assert run('conv', '-o/dev/null', 'game'), err.getvalue()


def test_conv_mat_norm(mgame_str):
    """Test convert matrix to normalized version"""
    with stdin(mgame_str), stdout() as out, stderr() as err:
        assert run('conv', 'norm'), err.getvalue()
    game = gamereader.loads(out.getvalue())
    assert np.allclose(game.min_role_payoffs(), 0)
    assert np.all(np.isclose(game.max_role_payoffs(), 1) |
                  np.isclose(game.max_role_payoffs(), 0))


def test_conv_game_mat_inv(game, game_str):
    """Test convert game to matrix and back"""
    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('conv', 'matgame'), err.getvalue()
    with stdin(out.getvalue()), stdout() as out, stderr() as err:
        assert run('conv', 'game'), err.getvalue()
    copy = gamereader.loads(out.getvalue())
    assert copy == paygame.game_copy(matgame.matgame_copy(game))


def test_conv_game_gambit_inv(game, game_str):
    """Test game to gambit and back"""
    with stdin(game_str), stdout() as out, stderr() as err:
        assert run('conv', 'gambit'), err.getvalue()
    with stdin(out.getvalue()), stdout() as out, stderr() as err:
        assert run('conv', 'game'), err.getvalue()
    copy = gamereader.loads(out.getvalue())
    assert copy == paygame.game_copy(matgame.matgame_copy(game))


def test_conv_mat_game_inv(mgame, mgame_str):
    """Test convert mat to game and back"""
    with stdin(mgame_str), stdout() as out, stderr() as err:
        assert run('conv', 'game'), err.getvalue()
    with stdin(out.getvalue()), stdout() as out, stderr() as err:
        assert run('conv', 'matgame'), err.getvalue()
    copy = gamereader.loads(out.getvalue())
    assert copy == mgame


def test_conv_mat_gambit_inv(mgame, mgame_str):
    """Test convert mat to gambit and back"""
    with stdin(mgame_str), stdout() as out, stderr() as err:
        assert run('conv', 'gambit'), err.getvalue()
    with stdin(out.getvalue()), stdout() as out, stderr() as err:
        assert run('conv', 'matgame'), err.getvalue()
    copy = gamereader.loads(out.getvalue())
    assert copy == mgame
