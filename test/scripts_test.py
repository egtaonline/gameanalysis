import json
import subprocess
import tempfile
from os import path

from gameanalysis import gamegen

# XXX To pass files to some scripts we use tempfile.NamedTemporaryFile and just
# flush it. This will likely fail on windows.

DIR = path.dirname(path.realpath(__file__))
GA = path.join(DIR, '..', 'bin', 'ga')
GAME = path.join(DIR, 'hard_nash_game_1.json')


def test_help():
    assert not subprocess.run([GA, '--help']).returncode
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
    assert not subprocess.run([GA, 'dom', '-i', GAME, '-s']).returncode
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
    assert not subprocess.run([GA, 'gen', 'congest', '3', '4', '2']).returncode
    with open(GAME) as f:
        assert not subprocess.run([GA, 'gen', 'noise', 'uniform', '1.5', '5'],
                                  stdin=f).returncode
    assert not subprocess.run([GA, 'gen', 'noise', 'gumbel', '1.5', '5', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'gen', 'noise', 'bimodal', '1.5', '5', '-i',
                               GAME]).returncode
    assert not subprocess.run([GA, 'gen', 'noise', 'gaussian', '1.5', '5',
                               '-i', GAME]).returncode


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

    with tempfile.NamedTemporaryFile('w') as game:
        sgame = gamegen.rock_paper_scissors()
        serial = gamegen.game_serializer(sgame)
        json.dump(sgame.to_json(serial), game)
        game.flush()
        assert not subprocess.run([GA, 'nash', '-tpure', '--one', '-i',
                                   game.name]).returncode


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

    # Singleton profile
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = {
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}
        json.dump(prof, pure)
        pure.flush()
        assert not subprocess.run([GA, 'pay', '-i', GAME, pure.name,
                                   '-o/dev/null']).returncode

    # Singleton payoff as string
    prof = {
        'background': {
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
        'hft': {'noop': 1}}
    profstr = json.dumps(prof)
    assert not subprocess.run([GA, 'pay', '-i', GAME, profstr,
                               '-o/dev/null']).returncode


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


def test_learning():
    with tempfile.NamedTemporaryFile('w') as game:
        sgame = gamegen.add_noise(gamegen.role_symmetric_game([2, 2], [3, 3]),
                                  10)
        serial = gamegen.game_serializer(sgame)
        json.dump(sgame.to_json(serial), game)
        game.flush()

        assert not subprocess.run([GA, 'learning', '-i', game.name,
                                   '-o/dev/null', '-p1', '--dist-thresh',
                                   '1e-3', '-r1e-3', '-t1e-3',
                                   '--rand-restarts', '0', '-m10000',
                                   '-c1e-8']).returncode
        game.seek(0)
        assert not subprocess.run([GA, 'learning'], stdin=game).returncode


def test_congestion():
    assert not subprocess.run([GA, 'congest', '3', '4', '2']).returncode


def test_sgboot():
    with tempfile.NamedTemporaryFile('w') as mixed, \
            tempfile.NamedTemporaryFile('w') as game:
        sgame = gamegen.add_noise(gamegen.role_symmetric_game([2, 3], [4, 3]),
                                  20)
        serial = gamegen.game_serializer(sgame)
        json.dump(sgame.to_json(serial), game)
        game.flush()

        profs = [serial.to_prof_json(sgame.uniform_mixture())]
        json.dump(profs, mixed)
        mixed.flush()

        assert not subprocess.run([GA, 'sgboot', '-i', game.name, mixed.name,
                                   '-o/dev/null']).returncode
        game.seek(0)
        assert not subprocess.run([GA, 'sgboot', mixed.name, '-tsurplus',
                                   '--processes', '1', '-n21', '-p', '5', '95',
                                   '-m'], stdin=game).returncode


def test_samp():
    with tempfile.NamedTemporaryFile('w') as mixed:
        prof = {
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 1},
            'hft': {'noop': 1}}
        json.dump(prof, mixed)
        mixed.flush()
        assert not subprocess.run([GA, 'samp', '-i', GAME, '-m', mixed.name,
                                   '-o/dev/null']).returncode
        with open(GAME) as f:
            assert not subprocess.run([GA, 'samp', '-m', mixed.name, '-n2',
                                       '-d'], stdin=f).returncode
