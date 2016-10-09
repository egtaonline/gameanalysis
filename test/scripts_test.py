import json
import subprocess
import tempfile
from os import path

# XXX To pass files to some scripts we use tempfile.NamedTemporaryFile and just
# flush it. This will likely fail on windows.

DIR = path.dirname(path.realpath(__file__))
GA = path.join(DIR, '..', 'bin', 'ga')
GAME = path.join(DIR, 'hard_nash_game_1.json')


def test_help():
    assert not subprocess.run([GA, '--help']).returncode
    assert not subprocess.run([GA, 'help']).returncode
    assert not subprocess.run([GA, 'help', 'nash']).returncode
    assert subprocess.run([GA, '--fail']).returncode
    assert subprocess.run([GA]).returncode


def test_convert():
    with open(GAME) as f:
        assert not subprocess.run([GA, 'conv'], stdin=f).returncode
    assert not subprocess.run([GA, 'conv', '-i', GAME,
                               '-o/dev/null']).returncode
    assert not subprocess.run([GA, 'conv', '-fjson', '-i', GAME]).returncode


def test_dominance():
    assert not subprocess.run([GA, 'dom', '-h']).returncode
    assert not subprocess.run([GA, 'dom', '-i', GAME]).returncode
    with open(GAME) as f:
        assert not subprocess.run([GA, 'dom', '-cweakdom', '-o/dev/null'],
                                  stdin=f).returncode
    assert not subprocess.run([GA, 'dom', '-cstrictdom', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'dom', '-cneverbr', '-i', GAME]).returncode


def test_gamegen():
    assert subprocess.run([GA, 'gen']).returncode
    assert not subprocess.run([GA, 'gen', 'uzs', '6', '-n',
                               '-o/dev/null']).returncode
    assert subprocess.run([GA, 'gen', 'ursym', '5']).returncode
    assert not subprocess.run([GA, 'gen', 'ursym', '3', '4', '4',
                               '3']).returncode
    assert not subprocess.run([GA, 'gen', 'congest', '3', '2', '4']).returncode
    with open(GAME) as f:
        assert not subprocess.run([GA, 'gen', 'noise', 'uniform', '1.5', '5'],
                                  stdin=f).returncode
    assert not subprocess.run([GA, 'gen', 'noise', 'gumbel', '1.5', '5', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'gen', 'noise', 'bimodal', '1.5', '5', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'gen', 'noise', 'gaussian', '1.5', '5',
                               '-i', GAME]).returncode
    assert not subprocess.run([GA, 'gen', 'help']).returncode
    assert not subprocess.run([GA, 'gen', 'help', 'ursym']).returncode


def test_nash():
    with open(GAME) as f:
        assert not subprocess.run([GA, 'nash'], stdin=f).returncode
    assert not subprocess.run([
        GA, 'nash', '-i', GAME, '-o/dev/null', '-r1e-2', '-d1e-2', '-c1e-7',
        '-x100', '-s1e-2', '-m5', '-n', '-p1']).returncode
    assert not subprocess.run([GA, 'nash', '-tpure', '-i', GAME]).returncode
    assert not subprocess.run([GA, 'nash', '-tmin-reg-prof', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'nash', '-tmin-reg-grid', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'nash', '-tmin-reg-rand', '-m10', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'nash', '-trand', '-m10', '-i',
                               GAME]).returncode
    assert subprocess.run([GA, 'nash', '-tfail', '-i', GAME]).returncode


def test_payoff():
    # Pure profile
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = [{
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}]
        json.dump(prof, pure)
        pure.flush()
        assert not subprocess.run([GA, 'pay', '-i', GAME, pure.name,
                                   '-o/dev/null']).returncode
        with open(GAME) as f:
            assert not subprocess.run([GA, 'pay', pure.name, '-twelfare'],
                                      stdin=f).returncode

    # Mixed profile
    with tempfile.NamedTemporaryFile('w') as mixed:
        prof = [{
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 1},
            'hft': {'noop': 1}}]
        json.dump(prof, mixed)
        mixed.flush()
        assert not subprocess.run([GA, 'pay', '-i', GAME, mixed.name,
                                   '-o/dev/null']).returncode
        with open(GAME) as f:
            assert not subprocess.run([GA, 'pay', mixed.name, '-twelfare'],
                                      stdin=f).returncode


def test_reduction():
    with open(GAME) as f:
        assert not subprocess.run([GA, 'red', 'background', '2', 'hft', '1'],
                                  stdin=f).returncode
    assert not subprocess.run([GA, 'red', '-ms', '2', '1', '-i', GAME,
                               '-o/dev/null']).returncode
    assert not subprocess.run([GA, 'red', '-thr', '-s', '2', '1', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'red', '-ttr', '-i', GAME]).returncode
    assert not subprocess.run([GA, 'red', '-tidr', '-i', GAME]).returncode


def test_regret():
    # Pure profile
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = [{
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}]
        json.dump(prof, pure)
        pure.flush()
        assert not subprocess.run([GA, 'reg', '-i', GAME, pure.name,
                                   '-o/dev/null']).returncode
        with open(GAME) as f:
            assert not subprocess.run([GA, 'reg', pure.name, '-tgains'],
                                      stdin=f).returncode
    # Mixed profile
    with tempfile.NamedTemporaryFile('w') as mixed:
        prof = [{
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 1},
            'hft': {'noop': 1}}]
        json.dump(prof, mixed)
        mixed.flush()
        assert not subprocess.run([GA, 'reg', '-i', GAME, mixed.name,
                                   '-o/dev/null']).returncode
        with open(GAME) as f:
            assert not subprocess.run([GA, 'reg', mixed.name, '-tgains'],
                                      stdin=f).returncode
    # Single input
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = {
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}
        json.dump(prof, pure)
        pure.flush()
        assert not subprocess.run([GA, 'reg', '-i', GAME,
                                   pure.name]).returncode


def test_subgame():
    assert not subprocess.run([GA, 'sub', '-nd', '-i', GAME,
                               '-o/dev/null']).returncode
    with open(GAME) as f:
        assert not subprocess.run([
            GA, 'sub', '-n', '-t', 'background',
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9', 'hft',
            '-s', '0', '3', '4'], stdin=f).returncode
    with tempfile.NamedTemporaryFile('w') as sub:
        prof = [{
            'background': [
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9'],
            'hft': ['noop']}]
        json.dump(prof, sub)
        sub.flush()
        assert not subprocess.run([GA, 'sub', '-i', GAME, '-f',
                                   sub.name]).returncode


def test_analysis():
    with open(GAME) as f:
        assert not subprocess.run([GA, 'analyze'], stdin=f).returncode
    assert not subprocess.run([
        GA, 'analyze', '-i', GAME, '-o/dev/null', '--subgames', '--dominance',
        '--dpr', 'background', '6', 'hft', '1', '-p1', '--dist-thresh', '1e-3',
        '-r1e-3', '-t1e-3', '--rand-restarts', '0', '-m10000',
        '-c1e-8']).returncode


def test_congestion():
    assert not subprocess.run([GA, 'congest', '3', '2', '4']).returncode
