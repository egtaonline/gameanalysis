import json
import os
import random
import subprocess
import tempfile

import numpy as np

from gameanalysis import gamegen
from gameanalysis import gameio
from gameanalysis import reduction
from gameanalysis import rsgame
from gameanalysis import subgame
from gameanalysis import utils

# XXX To pass files to some scripts we use tempfile.NamedTemporaryFile and just
# flush it. This will likely fail on windows.

DIR = os.path.dirname(os.path.realpath(__file__))
GA = os.path.join(DIR, '..', 'bin', 'ga')
GAME = os.path.join(DIR, 'hard_nash_game_1.json')
with open(GAME, 'r') as f:
    GAME_STR = f.read()
GAME_JSON = json.loads(GAME_STR)
GAME_DATA, SERIAL = gameio.read_game(GAME_JSON)


def run(*cmd, fail=False, input=''):
    res = subprocess.run((GA,) + cmd, input=input.encode('utf-8'),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = res.stdout.decode('utf-8')
    err = res.stderr.decode('utf-8')
    return not res.returncode, out, err


def test_help():
    assert not run()[0]
    assert not run('--fail')[0]
    success, out, err = run('--help')
    assert success, err
    for cmd in (line.split()[0] for line in out.split('\n')
                if line.startswith('    ') and line[4] != ' '):
        success, _, err = run(cmd, '--help')
        assert success, err


def test_dominance_1():
    success, out, err = run('dom', '-i', GAME)
    assert success, err
    game, serial = gameio.read_game(json.loads(out))
    assert serial == SERIAL
    assert game == GAME_DATA


def test_dominance_2():
    success, out, err = run('dom', '-i', GAME, '-s')
    assert success, err
    assert json.loads(out) == GAME_JSON['strategies']


def test_dominance_3():
    success, _, err = run('dom', '-cweakdom', '-o/dev/null', input=GAME_STR)
    assert success, err


def test_dominance_4():
    success, out, err = run('dom', '-cstrictdom', '-i', GAME)
    assert success, err
    gameio.read_game(json.loads(out))


def test_dominance_5():
    success, out, err = run('dom', '-cneverbr', '-i', GAME)
    assert success, err
    gameio.read_game(json.loads(out))


def test_gamegen_1():
    assert not run('gen')[0]
    assert not run('gen', 'ursym', '5')[0]
    success, _, err = run('gen', 'uzs', '6', '-n', '-o/dev/null')
    assert success, err


def test_gamegen_2():
    success, out, err = run('gen', 'ursym', '3', '4', '4', '3')
    assert success, err
    gameio.read_game(json.loads(out))


def test_gamegen_3():
    success, out, err = run('gen', 'noise', 'uniform', '1.5', '5',
                            input=GAME_STR)
    assert success, err
    gameio.read_game(json.loads(out))


def test_gamegen_4():
    success, out, err = run('gen', 'noise', 'gumbel', '1.5', '5', '-i', GAME)
    assert success, err
    gameio.read_game(json.loads(out))


def test_gamegen_5():
    success, out, err = run('gen', 'noise', 'bimodal', '1.5', '5', '-i', GAME)
    assert success, err
    gameio.read_game(json.loads(out))


def test_gamegen_6():
    success, out, err = run('gen', 'noise', 'gaussian', '1.5', '5', '-i', GAME)
    assert success, err
    gameio.read_game(json.loads(out))


def test_nash_1():
    assert not run('nash', '-tfail', '-i', GAME)[0]

    success, out, err = run('nash', input=GAME_STR)
    assert success, err
    assert any(
        np.allclose(SERIAL.from_mix_json(mix),
                    [0.54074617,  0.45925383,  0., 0., 0., 1., 0., 0., 0.])
        for mix in json.loads(out))


def test_nash_2():
    success, _, err = run(
        'nash', '-i', GAME, '-o/dev/null', '-r1e-2', '-d1e-2', '-c1e-7',
        '-x100', '-s1e-2', '-m5', '-n', '-p1')
    assert success, err


def test_nash_3():
    success, out, err = run('nash', '-tpure', '-i', GAME)
    assert success, err
    assert any(
        np.all(SERIAL.from_prof_json(prof) == [4, 2, 0, 0, 0, 1, 0, 0, 0])
        for prof in json.loads(out))


def test_nash_4():
    success, out, err = run('nash', '-tmin-reg-prof', '-i', GAME)
    assert success, err
    assert any(
        np.all(SERIAL.from_prof_json(prof) == [4, 2, 0, 0, 0, 1, 0, 0, 0])
        for prof in json.loads(out))


def test_nash_5():
    success, out, err = run('nash', '-tmin-reg-grid', '-i', GAME)
    assert success, err
    assert any(
        np.allclose(SERIAL.from_mix_json(mix), [0, 1, 0, 0, 0, 1, 0, 0, 0])
        for mix in json.loads(out))


def test_nash_6():
    success, out, err = run('nash', '-tmin-reg-rand', '-m10', '-i', GAME)
    assert success, err
    for mix in json.loads(out):
        SERIAL.from_mix_json(mix)


def test_nash_7():
    success, out, err = run('nash', '-trand', '-m10', '-i', GAME)
    assert success, err
    for mix in json.loads(out):
        SERIAL.from_mix_json(mix)


def test_nash_8():
    with tempfile.NamedTemporaryFile('w') as game:
        sgame = gamegen.rock_paper_scissors()
        serial = gamegen.serializer(sgame)
        json.dump(serial.to_game_json(sgame), game)
        game.flush()
        success, _, err = run('nash', '-tpure', '--one', '-i', game.name)
        assert success, err


def test_payoff_pure():
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = [{
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}]
        json.dump(prof, pure)
        pure.flush()
        success, _, err = run('pay', '-i', GAME, pure.name, '-o/dev/null')
        assert success, err

        success, out, err = run('pay', pure.name, '-twelfare', input=GAME_STR)
        assert success, err
        assert np.isclose(json.loads(out)[0], -315.4034577992763)


def test_payoff_mixed():
    with tempfile.NamedTemporaryFile('w') as mixed:
        prof = [{
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 1},
            'hft': {'noop': 1}}]
        json.dump(prof, mixed)
        mixed.flush()
        success, _, err = run('pay', '-i', GAME, mixed.name, '-o/dev/null')
        assert success, err

        success, out, err = run('pay', mixed.name, '-twelfare', input=GAME_STR)
        assert success, err
        assert np.isclose(json.loads(out)[0], -315.4034577992763)


def test_payoff_pure_single():
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = {
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}
        json.dump(prof, pure)
        pure.flush()
        success, out, err = run('pay', '-i', GAME, pure.name)
        assert success, err
        assert np.allclose(SERIAL.from_payoff_json(json.loads(out)[0]),
                           [0, -52.56724296654605, 0, 0, 0, 0, 0, 0, 0])


def test_payoff_pure_string():
    # Singleton payoff as string
    prof = {
        'background': {
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
        'hft': {'noop': 1}}
    profstr = json.dumps(prof)
    success, out, err = run('pay', '-i', GAME, profstr)
    assert success, err
    assert np.allclose(SERIAL.from_payoff_json(json.loads(out)[0]),
                       [0, -52.56724296654605, 0, 0, 0, 0, 0, 0, 0])


def test_reduction_1():
    success, out, err = run('red', 'background:2,hft:1', input=GAME_STR)
    assert success, err
    game, serial = gameio.read_game(json.loads(out))
    red = reduction.DeviationPreserving([2, 7], [6, 1], [2, 1])
    assert serial == SERIAL
    assert game == red.reduce_game(GAME_DATA)


def test_reduction_2():
    success, out, err = run('red', '-m', '-s', '2,1', '-i', GAME)
    assert success, err
    game, serial = gameio.read_samplegame(json.loads(out))
    red = reduction.DeviationPreserving([2, 7], [6, 1], [2, 1])
    assert serial == SERIAL
    assert game == red.reduce_game(rsgame.samplegame_copy(GAME_DATA))


def test_reduction_3():
    success, out, err = run('red', '-thr', '-s', '2,1', '-i', GAME)
    assert success, err
    game, serial = gameio.read_game(json.loads(out))
    red = reduction.Hierarchical([2, 7], [6, 1], [2, 1])
    assert serial == SERIAL
    assert game == red.reduce_game(GAME_DATA)


def test_reduction_4():
    success, out, err = run('red', '-ttr', '-i', GAME)
    assert success, err
    game, serial = gameio.read_game(json.loads(out))
    red = reduction.Twins([2, 7], [6, 1])
    assert serial == SERIAL
    assert game == red.reduce_game(GAME_DATA)


def test_reduction_5():
    success, out, err = run('red', '-tidr', '-i', GAME)
    assert success, err
    game, serial = gameio.read_game(json.loads(out))
    assert serial == SERIAL
    assert game == GAME_DATA


def test_regret_pure():
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = {
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}
        json.dump([prof], pure)
        pure.flush()
        success, out, err = run('reg', '-i', GAME, pure.name)
        assert success, err
        assert np.isclose(json.loads(out)[0], 7747.618428)

        success, out, err = run('reg', pure.name, '-tgains', input=GAME_STR)
        assert success, err
        dev_pay = json.loads(out)[0]
        for role, strats in dev_pay.items():
            prof_strats = prof[role]
            assert prof_strats.keys() == strats.keys()
            role_strats = set(SERIAL.strat_names[SERIAL.role_index(role)])
            for strat, dev_strats in strats.items():
                rstrats = role_strats.copy()
                rstrats.remove(strat)
                assert rstrats == dev_strats.keys()


def test_regret_mixed():
    with tempfile.NamedTemporaryFile('w') as mixed:
        prof = [{
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 1},
            'hft': {'noop': 1}}]
        json.dump(prof, mixed)
        mixed.flush()
        success, out, err = run('reg', '-i', GAME, mixed.name)
        assert success, err
        assert np.isclose(json.loads(out)[0], 7747.618428)

        success, out, err = run('reg', mixed.name, '-tgains', input=GAME_STR)
        assert success, err
        assert np.allclose(SERIAL.from_payoff_json(json.loads(out)[0]),
                           [581.18996992, 0., 0., 4696.19261, 3716.207196,
                            7747.618428, 4569.842172, 4191.665254,
                            4353.146694])


def test_regret_single():
    with tempfile.NamedTemporaryFile('w') as pure:
        prof = {
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 6},
            'hft': {'noop': 1}}
        json.dump(prof, pure)
        pure.flush()
        success, out, err = run('reg', '-i', GAME, pure.name)
        assert success, err
        assert np.isclose(json.loads(out)[0], 7747.618428)


def test_subgame_detect():
    success, out, err = run('sub', '-nd', '-i', GAME)
    assert success, err
    assert SERIAL.from_subgame_json(json.loads(out)[0]).all()


def test_subgame_extract_1():
    success, out, err = run(
        'sub', '-n', '-t', 'background',
        'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9', 'hft', 'noop',
        '-s', '0', '3', '4', input=GAME_STR)
    assert success, err

    expected = {utils.hash_array([False,  True,  True, False, False, False,
                                  False, False, False]),
                utils.hash_array([True, False, False,  True,  True, False,
                                  False, False, False])}
    assert {utils.hash_array(SERIAL.from_subgame_json(s))
            for s in json.loads(out)} == expected


def test_subgame_extract_2():
    with tempfile.NamedTemporaryFile('w') as sub:
        subg = [False, True, True, False, False, False, False, False, False]
        json.dump([SERIAL.to_subgame_json(subg)], sub)
        sub.flush()
        success, out, err = run('sub', '-i', GAME, '-f', sub.name)
        assert success, err
        game, serial = gameio.read_game(json.loads(out)[0])
        assert serial == subgame.subserializer(SERIAL, subg)
        assert game == subgame.subgame(GAME_DATA, subg)


def test_analysis_1():
    success, out, err = run('analyze', input=GAME_STR)
    assert success, err
    start = '''Game Analysis
=============
Game:
    Roles: background, hft
    Players:
        6x background
        1x hft
    Strategies:
        background:
            markov:rmin_30000_rmax_30000_thresh_0.001_priceVarEst_1e6
            markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9
        hft:
            noop
            trend:trendLength_5_profitDemanded_100_expiration_100
            trend:trendLength_5_profitDemanded_20_expiration_100
            trend:trendLength_5_profitDemanded_50_expiration_50
            trend:trendLength_8_profitDemanded_100_expiration_50
            trend:trendLength_8_profitDemanded_20_expiration_50
            trend:trendLength_8_profitDemanded_50_expiration_50
payoff data for 49 out of 49 profiles'''
    assert out.startswith(start)
    assert 'Social Welfare\n--------------' in out
    assert 'Maximum social welfare profile:' in out
    assert 'Maximum "background" welfare profile:' in out
    assert 'Maximum "hft" welfare profile:' in out
    assert 'Equilibria\n----------' in out
    assert 'No-equilibria Subgames\n----------------------' in out
    assert ('Unconfirmed Candidate Equilibria\n'
            '--------------------------------') in out
    assert ('Unexplored Best-response Subgames\n'
            '---------------------------------') in out
    assert 'Json Data\n=========' in out


def test_analysis_2():
    run('analyze', '-i', GAME, '-o/dev/null', '--subgames', '--dominance',
        '--dpr', 'background:6,hft:1', '-p1', '--dist-thresh', '1e-3',
        '-r1e-3', '-t1e-3', '--rand-restarts', '0', '-m10000', '-c1e-8')


def test_analysis_3():
    profiles = [
        # Complete deviations but unexplored
        [4, 0, 0, 0, 0],
        [3, 1, 0, 0, 0],
        [3, 0, 1, 0, 0],
        [3, 0, 0, 1, 0],
        [3, 0, 0, 0, 1],
        # Deviating subgame also explored
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
        # Deviating subgame
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
        # Deviating subgame also explored
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
        # Deviating subgame
        [0, 2, 0, 2, 0],
        [0, 1, 0, 3, 0],
        [0, 0, 0, 4, 0],
    ]
    game = rsgame.game([4], [5], profiles, payoffs)
    serial = gamegen.serializer(game)
    game_str = json.dumps(serial.to_game_json(game))

    success, out, err = run('analyze', '-sd', input=game_str)
    assert success, err
    assert 'Found 1 dominated strategy' in out
    assert 'Found 1 unconfirmed candidate' in out
    assert 'Found 1 unexplored best-response subgame' in out


def test_analysis_4():
    game = rsgame.game([2], [2], [[1, 1]], [[5, float('nan')]])
    serial = gamegen.serializer(game)
    game_str = json.dumps(serial.to_game_json(game))

    success, out, err = run('analyze', '-s', input=game_str)
    assert success, err
    assert 'There was no profile with complete payoff data' in out
    assert 'Found no complete subgames' in out


def test_learning_1():
    success, out, err = run('learning', input=GAME_STR)
    assert success, err
    start = '''Game Learning
=============
Game:
    Roles: background, hft
    Players:
        6x background
        1x hft
    Strategies:
        background:
            markov:rmin_30000_rmax_30000_thresh_0.001_priceVarEst_1e6
            markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9
        hft:
            noop
            trend:trendLength_5_profitDemanded_100_expiration_100
            trend:trendLength_5_profitDemanded_20_expiration_100
            trend:trendLength_5_profitDemanded_50_expiration_50
            trend:trendLength_8_profitDemanded_100_expiration_50
            trend:trendLength_8_profitDemanded_20_expiration_50
            trend:trendLength_8_profitDemanded_50_expiration_50
payoff data for 49 out of 49 profiles'''
    assert out.startswith(start)
    assert 'Social Welfare\n--------------' in out
    assert 'Maximum social welfare profile:' in out
    assert 'Maximum "background" welfare profile:' in out
    assert 'Maximum "hft" welfare profile:' in out
    assert 'Equilibria\n----------' in out
    assert 'Json Data\n=========' in out


def test_learning_2():
    run('learning', '-i', GAME, '-o/dev/null', '-p1', '--dist-thresh', '1e-3',
        '-r1e-3', '-t1e-3', '--rand-restarts', '0', '-m10000', '-c1e-8')


def test_sgboot_1():
    with tempfile.NamedTemporaryFile('w') as mixed, \
            tempfile.NamedTemporaryFile('w') as game:
        sgame = gamegen.add_noise(gamegen.role_symmetric_game([2, 3], [4, 3]),
                                  20)
        serial = gamegen.serializer(sgame)
        json.dump(serial.to_samplegame_json(sgame), game)
        game.flush()

        profs = [serial.to_prof_json(sgame.uniform_mixture())]
        json.dump(profs, mixed)
        mixed.flush()

        run('sgboot', '-i', game.name, mixed.name, '-o/dev/null')


def test_sgboot_2():
    with tempfile.NamedTemporaryFile('w') as mixed:
        sgame = gamegen.add_noise(gamegen.role_symmetric_game([2, 3], [4, 3]),
                                  20)
        serial = gamegen.serializer(sgame)
        game_str = json.dumps(serial.to_samplegame_json(sgame))

        profs = [serial.to_prof_json(sgame.uniform_mixture())]
        json.dump(profs, mixed)
        mixed.flush()

        success, out, err = run(
            'sgboot', mixed.name, '-tsurplus', '--processes', '1', '-n21',
            '-p', '5', '95', '-m', input=game_str)
        assert success, err
        data = json.loads(out)
        assert all(j.keys() == {'5', '95', 'mean'} for j in data)
        assert all(j['5'] <= j['95'] for j in data)


def test_sampboot():
    inp = json.dumps([random.random() for _ in range(10)])
    success, out, err = run('sampboot', '-n21', '-m', input=inp)
    assert success, err
    data = json.loads(out)
    keys = list(map(str, range(0, 101, 5)))
    assert data.keys() == set(keys + ['mean'])
    ordered = [data[k] for k in keys]
    assert all(a <= b for a, b in zip(ordered[:-1], ordered[1:]))


def test_sampboot_reg():
    inp = []
    for _ in range(10):
        for role, strats in zip(SERIAL.role_names, SERIAL.strat_names):
            for strat in strats:
                inp.append({'role': role, 'strategy': strat,
                            'payoff': random.random()})
    with tempfile.NamedTemporaryFile('w') as mixed:
        json.dump(SERIAL.to_mix_json(GAME_DATA.random_mixtures()), mixed)
        mixed.flush()

        success, out, err = run(
            'sampboot', '--regret', GAME, mixed.name, '-n21', '-p', '5', '95',
            '-m', input=json.dumps(inp))
        assert success, err

        data = json.loads(out)
        assert data.keys() == {'5', '95', 'mean'}
        assert data['5'] <= data['95']


def test_sampboot_dev_surp():
    inp = []
    for _ in range(10):
        for role, strats in zip(SERIAL.role_names, SERIAL.strat_names):
            for strat in strats:
                inp.append({'role': role, 'strategy': strat,
                            'payoff': random.random()})
    with tempfile.NamedTemporaryFile('w') as mixed:
        json.dump(SERIAL.to_mix_json(GAME_DATA.random_mixtures()), mixed)
        mixed.flush()

        success, out, err = run(
            'sampboot', '--dev-surplus', GAME, mixed.name, '-n21', '-p', '5',
            '95', '-m', input=json.dumps(inp))
        assert success, err

        data = json.loads(out)
        assert data.keys() == {'5', '95', 'mean'}
        assert data['5'] <= data['95']


def test_samp():
    with tempfile.NamedTemporaryFile('w') as mixed:
        prof = {
            'background': {
                'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9': 1},
            'hft': {'noop': 1}}
        json.dump(prof, mixed)
        mixed.flush()
        success, out, err = run('samp', '-i', GAME, '-m', mixed.name)
        assert success, err
        prof = SERIAL.from_prof_json(json.loads(out))
        assert GAME_DATA.verify_profile(prof)

        success, out, err = run('samp', '-m', mixed.name, '-n2', '-d',
                                input=GAME_STR)
        assert success, err
        lines = out[:-1].split('\n')
        assert len(lines) == 2 * 9
        for line in lines:
            prof = SERIAL.from_prof_json(json.loads(line)['profile'])
            assert GAME_DATA.verify_profile(prof)
