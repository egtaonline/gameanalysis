import copy
import json
import warnings

import numpy as np
import pytest

from gameanalysis import serialize
from gameanalysis import rsgame
from gameanalysis import utils


SERIAL = serialize.gameserializer(['role'], [['strat1', 'strat2']])
SSERIAL = serialize.samplegameserializer_copy(SERIAL)
SERIAL2 = serialize.gameserializer(['a', 'b'], [['bar', 'foo'], ['baz']])

GAME = rsgame.samplegame(
    [2], [2],
    [[2, 0],
     [1, 1],
     [0, 2]],
    [
        [[[-1, 0, 1], [0, 0, 0]],
         [[9, 10, 11], [21, 20, 19]]],
        [[[0, 0, 0, 0], [32, 28, 30, 30]]],
    ],
)

BASEGAME_JSON = {
    'players': {
        'role': 2,
    },
    'strategies': {
        'role': [
            'strat1',
            'strat2',
        ],
    },
}

GAME_JSON = {
    'players': {
        'role': 2,
    },
    'strategies': {
        'role': [
            'strat1',
            'strat2',
        ],
    },
    'profiles': [
        {
            'role': [
                ('strat1', 2, 0.0),
            ],
        },
        {
            'role': [
                ('strat1', 1, 10.0),
                ('strat2', 1, 20.0),
            ],
        },
        {
            'role': [
                ('strat2', 2, 30.0),
            ],
        },
    ],
}

SAMPLEGAME_JSON = {
    'players': {
        'role': 2,
    },
    'strategies': {
        'role': [
            'strat1',
            'strat2',
        ],
    },
    'profiles': [
        {
            'role': [
                ('strat1', 2, [-1.0, 0.0, 1.0]),
            ],
        },
        {
            'role': [
                ('strat1', 1, [9.0, 10.0, 11.0]),
                ('strat2', 1, [21.0, 20.0, 19.0]),
            ],
        },
        {
            'role': [
                ('strat2', 2, [32.0, 28.0, 30.0, 30.0]),
            ],
        },
    ],
}

EMPTYGAME_JSON = {
    'roles': [
        {
            'name': 'role',
            'strategies': [
                'strat1',
                'strat2',
            ],
            'count': 2,
        },
    ],
}

SUMMARYGAME_JSON = {
    'roles': [
        {
            'name': 'role',
            'strategies': [
                'strat1',
                'strat2',
            ],
            'count': 2,
        },
    ],
    'profiles': [
        {
            'symmetry_groups': [
                {
                    'payoff': 0,
                    'count': 2,
                    'strategy': 'strat1',
                    'role': 'role',
                },
            ],
        },
        {
            'symmetry_groups': [
                {
                    'payoff': 10,
                    'count': 1,
                    'strategy': 'strat1',
                    'role': 'role',
                },
                {
                    'payoff': 20,
                    'count': 1,
                    'strategy': 'strat2',
                    'role': 'role',
                },
            ],
        },
        {
            'symmetry_groups': [
                {
                    'payoff': 30,
                    'count': 2,
                    'strategy': 'strat2',
                    'role': 'role',
                },
            ],
        },
    ],
}

OBSERVATIONGAME_JSON = {
    'roles': [
        {
            'name': 'role',
            'strategies': [
                'strat1',
                'strat2',
            ],
            'count': 2,
        },
    ],
    'profiles': [
        {
            'symmetry_groups': [
                {
                    'strategy': 'strat1',
                    'id': 0,
                    'role': 'role',
                    'count': 2,
                },
            ],
            'observations': [
                {
                    'symmetry_groups': [
                        {
                            'id': 0,
                            'payoff': -1,
                        },
                    ],
                },
                {
                    'symmetry_groups': [
                        {
                            'id': 0,
                            'payoff': 0,
                        },
                    ],
                },
                {
                    'symmetry_groups': [
                        {
                            'id': 0,
                            'payoff': 1,
                        },
                    ],
                },
            ],
        },
        {
            'symmetry_groups': [
                {
                    'strategy': 'strat1',
                    'id': 1,
                    'role': 'role',
                    'count': 1,
                },
                {
                    'strategy': 'strat2',
                    'id': 2,
                    'role': 'role',
                    'count': 1,
                },
            ],
            'observations': [
                {
                    'symmetry_groups': [
                        {
                            'id': 1,
                            'payoff': 9,
                        },
                        {
                            'id': 2,
                            'payoff': 21,
                        },
                    ],
                },
                {
                    'symmetry_groups': [
                        {
                            'id': 1,
                            'payoff': 10,
                        },
                        {
                            'id': 2,
                            'payoff': 20,
                        },
                    ],
                },
                {
                    'symmetry_groups': [
                        {
                            'id': 1,
                            'payoff': 11,
                        },
                        {
                            'id': 2,
                            'payoff': 19,
                        },
                    ],
                },
            ],
        },
        {
            'symmetry_groups': [
                {
                    'strategy': 'strat2',
                    'id': 3,
                    'role': 'role',
                    'count': 2,
                },
            ],
            'observations': [
                {
                    'symmetry_groups': [
                        {
                            'id': 3,
                            'payoff': 32,
                        },
                    ],
                },
                {
                    'symmetry_groups': [
                        {
                            'id': 3,
                            'payoff': 28,
                        },
                    ],
                },
                {
                    'symmetry_groups': [
                        {
                            'id': 3,
                            'payoff': 30,
                        },
                    ],
                },
                {
                    'symmetry_groups': [
                        {
                            'id': 3,
                            'payoff': 30,
                        },
                    ],
                },
            ],
        },
    ],
}

FULLGAME_JSON = {
    'roles': [
        {
            'name': 'role',
            'strategies': [
                'strat1',
                'strat2',
            ],
            'count': 2,
        },
    ],
    'profiles': [
        {
            'symmetry_groups': [
                {
                    'strategy': 'strat1',
                    'id': 0,
                    'role': 'role',
                    'count': 2,
                },
            ],
            'observations': [
                {
                    'players': [
                        {
                            'sid': 0,
                            'p': -2,
                        },
                        {
                            'sid': 0,
                            'p': 0,
                        },
                    ],
                },
                {
                    'players': [
                        {
                            'sid': 0,
                            'p': 0,
                        },
                        {
                            'sid': 0,
                            'p': 0,
                        },
                    ],
                },
                {
                    'players': [
                        {
                            'sid': 0,
                            'p': 0,
                        },
                        {
                            'sid': 0,
                            'p': 2,
                        },
                    ],
                },
            ],
        },
        {
            'symmetry_groups': [
                {
                    'strategy': 'strat1',
                    'id': 1,
                    'role': 'role',
                    'count': 1,
                },
                {
                    'strategy': 'strat2',
                    'id': 2,
                    'role': 'role',
                    'count': 1,
                },
            ],
            'observations': [
                {
                    'players': [
                        {
                            'sid': 1,
                            'p': 9,
                        },
                        {
                            'sid': 2,
                            'p': 21,
                        },
                    ],
                },
                {
                    'players': [
                        {
                            'sid': 1,
                            'p': 10,
                        },
                        {
                            'sid': 2,
                            'p': 20,
                        },
                    ],
                },
                {
                    'players': [
                        {
                            'sid': 1,
                            'p': 11,
                        },
                        {
                            'sid': 2,
                            'p': 19,
                        },
                    ],
                },
            ],
        },
        {
            'symmetry_groups': [
                {
                    'strategy': 'strat2',
                    'id': 3,
                    'role': 'role',
                    'count': 2,
                },
            ],
            'observations': [
                {
                    'players': [
                        {
                            'sid': 3,
                            'p': 32,
                        },
                        {
                            'sid': 3,
                            'p': 32,
                        },
                    ],
                },
                {
                    'players': [
                        {
                            'sid': 3,
                            'p': 30,
                        },
                        {
                            'sid': 3,
                            'p': 26,
                        },
                    ],
                },
                {
                    'players': [
                        {
                            'sid': 3,
                            'p': 34,
                        },
                        {
                            'sid': 3,
                            'p': 26,
                        },
                    ],
                },
                {
                    'players': [
                        {
                            'sid': 3,
                            'p': 28,
                        },
                        {
                            'sid': 3,
                            'p': 32,
                        },
                    ],
                },
            ],
        },
    ],
}


@pytest.mark.parametrize('jgame', [BASEGAME_JSON, GAME_JSON, SAMPLEGAME_JSON,
                                   EMPTYGAME_JSON, SUMMARYGAME_JSON,
                                   OBSERVATIONGAME_JSON, FULLGAME_JSON])
def test_game_from_json(jgame):
    game, serial = serialize.read_game(jgame)
    copy = serial.from_json(serial.to_json(game))
    assert game == copy


@pytest.mark.parametrize('jgame', [BASEGAME_JSON, GAME_JSON, SAMPLEGAME_JSON,
                                   EMPTYGAME_JSON, SUMMARYGAME_JSON,
                                   OBSERVATIONGAME_JSON, FULLGAME_JSON])
def test_samplegame_from_json(jgame):
    game, serial = serialize.read_samplegame(jgame)
    copy = serial.from_json(serial.to_json(game))
    assert game == copy


def test_emptygame_equality():
    game, serial = serialize.read_game(BASEGAME_JSON)
    assert game == rsgame.emptygame_copy(GAME)
    assert serial == SERIAL


@pytest.mark.parametrize('jgame', [GAME_JSON, SAMPLEGAME_JSON,
                                   SUMMARYGAME_JSON, OBSERVATIONGAME_JSON,
                                   FULLGAME_JSON])
def test_game_equality(jgame):
    game, serial = serialize.read_game(jgame)
    assert game == rsgame.game_copy(GAME)
    assert serial == SERIAL


@pytest.mark.parametrize('jgame', [SAMPLEGAME_JSON, OBSERVATIONGAME_JSON,
                                   FULLGAME_JSON])
def test_samplegame_equality(jgame):
    game, serial = serialize.read_samplegame(jgame)
    assert game == GAME
    assert serial == SSERIAL


def test_output():
    VGAME_JSON = GAME_JSON.copy()
    VGAME_JSON['type'] = 'game.1'

    EMPTYGAME_JSON = BASEGAME_JSON.copy()
    EMPTYGAME_JSON['profiles'] = []
    EMPTYGAME_JSON['type'] = 'game.1'

    assert VGAME_JSON == SERIAL.to_json(GAME)
    assert VGAME_JSON == SERIAL.to_json(rsgame.game_copy(GAME))
    assert EMPTYGAME_JSON == SERIAL.to_json(rsgame.emptygame_copy(GAME))

    VSAMPLEGAME_JSON = SAMPLEGAME_JSON.copy()
    VSAMPLEGAME_JSON['type'] = 'samplegame.1'

    SAMPLEDGAME_JSON = copy.deepcopy(GAME_JSON)
    for prof in SAMPLEDGAME_JSON['profiles']:
        for pays in prof.values():
            pays[:] = [(s, c, [p]) for s, c, p in pays]
    SAMPLEDGAME_JSON['type'] = 'samplegame.1'

    EMPTYGAME_JSON['type'] = 'samplegame.1'

    assert VSAMPLEGAME_JSON == SSERIAL.to_json(GAME)
    assert SAMPLEDGAME_JSON == SSERIAL.to_json(
        rsgame.game_copy(GAME))
    assert EMPTYGAME_JSON == SSERIAL.to_json(
        rsgame.emptygame_copy(GAME))

    expected = """
Game:
    Roles: a, b
    Players:
        3x a
        4x b
    Strategies:
        a:
            bar
            foo
        b:
            baz
payoff data for 0 out of 4 profiles
"""[1:-1]
    assert expected == SERIAL2.to_printstr(rsgame.emptygame(
        [3, 4], SERIAL2.num_role_strats))

    expected = """
SampleGame:
    Roles: role
    Players:
        2x role
    Strategies:
        role:
            strat1
            strat2
payoff data for 3 out of 3 profiles
"""[1:-1]
    assert expected == SERIAL.to_printstr(GAME)
    expected = """
Game:
    Roles: role
    Players:
        2x role
    Strategies:
        role:
            strat1
            strat2
payoff data for 3 out of 3 profiles
"""[1:-1]
    assert expected == SERIAL.to_printstr(rsgame.game_copy(GAME))
    expected = """
Game:
    Roles: role
    Players:
        2x role
    Strategies:
        role:
            strat1
            strat2
payoff data for 0 out of 3 profiles
"""[1:-1]
    assert expected == SERIAL.to_printstr(rsgame.emptygame_copy(GAME))

    expected = """
SampleGame:
    Roles: role
    Players:
        2x role
    Strategies:
        role:
            strat1
            strat2
payoff data for 3 out of 3 profiles
3 to 4 observations per profile
"""[1:-1]
    assert expected == SSERIAL.to_printstr(GAME)
    expected = """
SampleGame:
    Roles: role
    Players:
        2x role
    Strategies:
        role:
            strat1
            strat2
payoff data for 3 out of 3 profiles
1 observation per profile
"""[1:-1]
    assert expected == SSERIAL.to_printstr(rsgame.game_copy(GAME))
    expected = """
SampleGame:
    Roles: role
    Players:
        2x role
    Strategies:
        role:
            strat1
            strat2
payoff data for 0 out of 3 profiles
no observations
"""[1:-1]
    assert expected == SSERIAL.to_printstr(
        rsgame.emptygame_copy(GAME))


@pytest.mark.parametrize('_', range(20))
def test_sorted_strategy_loading(_):
    with open('test/hard_nash_game_1.json') as f:
        _, serial = serialize.read_game(json.load(f))
    assert utils.is_sorted(serial.role_names), \
        "loaded json game didn't have sorted roles"
    assert all(utils.is_sorted(strats) for strats in serial.strat_names), \
        "loaded json game didn't have sorted strategies"


def test_to_from_prof_json():
    prof = [6, 5, 3]
    json_prof = {'a': {'foo': 5, 'bar': 6}, 'b': {'baz': 3}}
    assert SERIAL2.to_prof_json(prof) == json_prof
    new_prof = SERIAL2.from_prof_json(json_prof)
    assert np.all(new_prof == prof)
    assert new_prof.dtype == int

    player_prof = {'players': [
        {'role': 'a', 'strategy': 'foo', 'payoff': 0},
        {'role': 'a', 'strategy': 'foo', 'payoff': 0},
        {'role': 'a', 'strategy': 'foo', 'payoff': 0},
        {'role': 'a', 'strategy': 'foo', 'payoff': 0},
        {'role': 'a', 'strategy': 'foo', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'b', 'strategy': 'baz', 'payoff': 0},
        {'role': 'b', 'strategy': 'baz', 'payoff': 0},
        {'role': 'b', 'strategy': 'baz', 'payoff': 0},
    ]}
    new_prof = SERIAL2.from_prof_json(player_prof)
    assert np.all(new_prof == prof)
    assert new_prof.dtype == int


def test_to_from_payoff_json_roles():
    pay = [1.0, 2.0, 3.0]
    json_pay = {'a': {'foo': 2.0, 'bar': 1.0}, 'b': {'baz': 3.0}}
    assert SERIAL2.to_payoff_json(pay) == json_pay
    new_pay = SERIAL2.from_payoff_json(json_pay)
    assert np.allclose(new_pay, pay)
    assert new_pay.dtype == float

    player_pay = {'players': [
        {'role': 'a', 'strategy': 'foo', 'payoff': 4},
        {'role': 'a', 'strategy': 'foo', 'payoff': 2},
        {'role': 'a', 'strategy': 'foo', 'payoff': 0},
        {'role': 'a', 'strategy': 'foo', 'payoff': 4},
        {'role': 'a', 'strategy': 'foo', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 2},
        {'role': 'a', 'strategy': 'bar', 'payoff': 2},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'a', 'strategy': 'bar', 'payoff': 2},
        {'role': 'a', 'strategy': 'bar', 'payoff': 0},
        {'role': 'b', 'strategy': 'baz', 'payoff': 0},
        {'role': 'b', 'strategy': 'baz', 'payoff': 6},
        {'role': 'b', 'strategy': 'baz', 'payoff': 3},
    ]}
    new_pay = SERIAL2.from_payoff_json(player_pay)
    assert np.allclose(new_pay, pay)
    assert new_pay.dtype == float


def test_to_from_mix_json():
    mix = [.6, .4, 1]
    json_mix = {'a': {'foo': .4, 'bar': .6}, 'b': {'baz': 1}}
    assert SERIAL2.to_mix_json(mix) == json_mix
    new_mix = SERIAL2.from_mix_json(json_mix)
    assert np.all(new_mix == mix)
    assert new_mix.dtype == float


def test_to_from_subgame_json():
    sub = [True, False, True]
    json_sub = {'a': ['bar'], 'b': ['baz']}
    assert SERIAL2.to_subgame_json(sub) == json_sub
    new_sub = SERIAL2.from_subgame_json(json_sub)
    assert np.all(new_sub == sub)
    assert new_sub.dtype == bool


def test_to_from_prof_str():
    prof = [6, 5, 3]
    prof_str = 'a: 5 foo, 6 bar; b: 3 baz'
    assert np.all(SERIAL2.from_prof_str(prof_str) == prof)
    assert set(SERIAL2.to_prof_str(prof)) == set(prof_str)


def test_to_from_samplepay_json():
    prof = [3, 0, 4]
    spay = [[3, 0, 7], [4, 0, 8], [5, 0, 9]]
    json_spay = {'a': {'bar': [3, 4, 5]}, 'b': {'baz': [7, 8, 9]}}
    json_spay_0 = {'a': {'bar': [3, 4, 5], 'foo': [0, 0, 0]},
                   'b': {'baz': [7, 8, 9]}}
    assert SERIAL2.to_samplepay_json(spay, prof) == json_spay
    assert SERIAL2.to_samplepay_json(spay) == json_spay_0
    assert np.allclose(SERIAL2.from_samplepay_json(json_spay), spay)

    with pytest.raises(AssertionError):
        SERIAL2.from_samplepay_json(
            json_spay, np.empty((0, SERIAL2.num_strats)))

    json_prof_spay = {'a': [('bar', 3, [3, 4, 5])],
                      'b': [('baz', 4, [7, 8, 9])]}
    with pytest.raises(AssertionError):
        SERIAL2.from_samplepay_json(
            json_prof_spay, np.empty((0, SERIAL2.num_strats)))


def test_to_from_profsamplepay_json():
    prof = [3, 0, 4]
    spay = [[3, 0, 7], [4, 0, 8], [5, 0, 9]]
    json_profspay = {'a': [('bar', 3,  [3, 4, 5])],
                     'b': [('baz', 4, [7, 8, 9])]}
    assert SERIAL2.to_profsamplepay_json(spay, prof) == json_profspay
    p, sp = SERIAL2.from_profsamplepay_json(json_profspay)
    assert np.all(p == prof)
    assert np.allclose(sp, spay)


def test_to_prof_printstr():
    prof = [6, 5, 3]
    expected = """
a:
    bar: 6
    foo: 5
b:
    baz: 3
"""[1:]
    assert SERIAL2.to_prof_printstr(prof) == expected


def test_to_from_mix_printstr():
    mix = [0.3, 0.7, 1]
    expected = """
a:
    bar:  30.00%
    foo:  70.00%
b:
    baz: 100.00%
"""[1:]
    assert SERIAL2.to_mix_printstr(mix) == expected


def test_to_from_subgame_printstr():
    sub = [True, False, True]
    expected = """
a:
    bar
b:
    baz
"""[1:]
    assert SERIAL2.to_subgame_printstr(sub) == expected


def test_to_from_role_json():
    role = [6, 3]
    json_role = {'a': 6, 'b': 3}
    assert SERIAL2.to_role_json(role) == json_role
    assert np.all(SERIAL2.from_role_json(json_role) == role)
    assert SERIAL2.from_role_json(json_role).dtype == float


def test_dev_payoff_json():
    prof = [3, 0, 4]
    devpay = [5, 0]
    json_devpay = {'a': {'bar': {'foo': 5}}, 'b': {'baz': {}}}
    json_devpay2 = {'a': {'bar': {'foo': 5}, 'foo': {'bar': 0}},
                    'b': {'baz': {}}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert SERIAL2.to_dev_payoff_json(devpay, prof) == json_devpay
        assert SERIAL2.to_dev_payoff_json(devpay) == json_devpay2
        dest = np.empty(SERIAL2.num_devs)
        SERIAL2.from_dev_payoff_json(json_devpay, dest)
        assert np.allclose(dest, devpay)
        assert np.allclose(SERIAL2.from_dev_payoff_json(json_devpay), devpay)

    prof = [2, 1, 4]
    devpay = [5, 4]
    json_devpay = {'a': {'bar': {'foo': 5},
                         'foo': {'bar': 4}},
                   'b': {'baz': {}}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert SERIAL2.to_dev_payoff_json(devpay, prof) == json_devpay
        assert SERIAL2.to_dev_payoff_json(devpay) == json_devpay
        dest = np.empty(SERIAL2.num_devs)
        SERIAL2.from_dev_payoff_json(json_devpay, dest)
        assert np.allclose(dest, devpay)
        assert np.allclose(SERIAL2.from_dev_payoff_json(json_devpay), devpay)


def test_to_pay_json():
    jprof = SERIAL.to_payoff_json(GAME.payoffs[0], GAME.profiles[0])
    assert jprof == {'role': {'strat1': 0}}
    jprof = SERIAL.to_payoff_json(GAME.payoffs[0])
    assert jprof == {'role': {'strat1': 0, 'strat2': 0}}
    jprof = SERIAL.to_payoff_json(GAME.payoffs[1], GAME.profiles[1])
    assert jprof == {'role': {'strat1': 10, 'strat2': 20}}
    jprof = SERIAL.to_payoff_json(GAME.payoffs[1])
    assert jprof == {'role': {'strat1': 10, 'strat2': 20}}
    jprof = SERIAL.to_payoff_json(GAME.payoffs[2], GAME.profiles[2])
    assert jprof == {'role': {'strat2': 30}}
    jprof = SERIAL.to_payoff_json(GAME.payoffs[2])
    assert jprof == {'role': {'strat1': 0, 'strat2': 30}}

    jprof = SERIAL.to_profpay_json(GAME.payoffs[0], GAME.profiles[0])
    assert jprof == {'role': [('strat1', 2, 0)]}
    jprof = {k: set(v) for k, v in SERIAL.to_profpay_json(
        GAME.payoffs[1], GAME.profiles[1]).items()}
    assert jprof == {'role': set([('strat1', 1, 10), ('strat2', 1, 20)])}
    jprof = SERIAL.to_profpay_json(GAME.payoffs[2], GAME.profiles[2])
    assert jprof == {'role': [('strat2', 2, 30)]}


@pytest.mark.parametrize('jgame', [GAME_JSON, SAMPLEGAME_JSON,
                                   SUMMARYGAME_JSON, OBSERVATIONGAME_JSON,
                                   FULLGAME_JSON])
def test_to_from_payoff_json(jgame):
    _, serial = serialize.read_game(jgame)
    payoffs = np.concatenate([serial.from_payoff_json(p)[None]
                              for p in jgame['profiles']])
    expected = [[0, 0],
                [10, 20],
                [0, 30]]
    assert np.allclose(expected, payoffs)


def test_load_empty_observations():
    serial = serialize.gameserializer(['a', 'b'], [['bar', 'foo'], ['baz']])
    profile = {
        'symmetry_groups': [
            {
                'strategy': 'bar',
                'id': 0,
                'role': 'a',
                'count': 1,
            },
            {
                'strategy': 'baz',
                'id': 1,
                'role': 'b',
                'count': 1,
            },
        ],
        'observations': [],
    }
    payoff = serial.from_payoff_json(profile)
    assert np.allclose(payoff, [np.nan, 0, np.nan], equal_nan=True)

    profile = {
        'a': {
            'bar': []
        },
        'b': {
            'baz': []
        },
    }
    payoff = serial.from_payoff_json(profile)
    assert np.allclose(payoff, [np.nan, 0, np.nan], equal_nan=True)


def test_sorted_errors():
    with pytest.raises(AssertionError):
        serialize.gameserializer(['b', 'a'], [['a', 'b'], ['a', 'b']])
    with pytest.raises(AssertionError):
        serialize.gameserializer(['role'], [['b', 'a']])


def test_invalid_game():
    with pytest.raises(ValueError):
        SERIAL.from_json({})
    with pytest.raises(ValueError):
        SSERIAL.from_json({})
    with pytest.raises(ValueError):
        serialize.read_game({})


def test_repr():
    assert repr(SERIAL) is not None


def test_strat_name():
    serial = serialize.gameserializer(
        ['a', 'b'], [['e', 'q', 'w'], ['r', 't']])
    for i, s in enumerate(['e', 'q', 'w', 'r', 't']):
        assert s == serial.strat_name(i)


def test_index():
    serial = serialize.gameserializer(
        ['a', 'b'], [['e', 'q', 'w'], ['r', 't']])
    assert 0 == serial.role_index('a')
    assert 1 == serial.role_index('b')
    assert 0 == serial.role_strat_index('a', 'e')
    assert 1 == serial.role_strat_index('a', 'q')
    assert 2 == serial.role_strat_index('a', 'w')
    assert 3 == serial.role_strat_index('b', 'r')
    assert 4 == serial.role_strat_index('b', 't')


def test_subgameserial():
    mask = [True, False, True, True, False]
    serial = serialize.gameserializer(
        ['a', 'b'], [['e', 'q', 'w'], ['r', 't']])
    sserial = serialize.gameserializer(
        ['a', 'b'], [['e', 'w'], ['r']])
    assert serial.subserial(mask) == sserial


def test_subsamplegameserial():
    mask = [True, False, True, True, False]
    serial = serialize.samplegameserializer(
        ['a', 'b'], [['e', 'q', 'w'], ['r', 't']])
    sserial = serialize.samplegameserializer(
        ['a', 'b'], [['e', 'w'], ['r']])
    assert serial.subserial(mask) == sserial


def test_serialization():
    """This will fail if numpy types make it to final output"""
    json.dumps(SERIAL.to_json(GAME))
    json.dumps(SSERIAL.to_json(GAME))


def test_serializer_constructors():
    copy = serialize.gameserializer_copy(SERIAL)
    assert copy == SERIAL

    copy = serialize.gameserializer_copy(SSERIAL)
    assert copy == SERIAL

    copy = serialize.gameserializer(SERIAL.role_names, SERIAL.strat_names)
    assert copy == SERIAL

    copy = serialize.samplegameserializer_copy(SERIAL)
    assert copy == SSERIAL

    copy = serialize.samplegameserializer_copy(SSERIAL)
    assert copy == SSERIAL

    copy = serialize.samplegameserializer(
        SSERIAL.role_names, SSERIAL.strat_names)
    assert copy == SSERIAL
