import json
import warnings

import numpy as np
import pytest

from gameanalysis import gameio

EMPTY_GAME_JSON = {
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
                ['strat1', 2, 4.5],
            ],
        },
        {
            'role': [
                ['strat1', 1, 2],
                ['strat2', 1, 1],
            ],
        },
        {
            'role': [
                ['strat2', 2, 10],
            ],
        },
    ],
}

SAMPLE_GAME_JSON = {
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
                ['strat1', 2, [4, 7, 4.5]],
            ],
        },
        {
            'role': [
                ['strat1', 1, [3, 6, 2]],
                ['strat2', 1, [1, 2]],
            ],
        },
        {
            'role': [
                ['strat2', 2, [10, 15]],
            ],
        },
    ],
}


@pytest.mark.parametrize('json_',
                         [EMPTY_GAME_JSON, GAME_JSON, SAMPLE_GAME_JSON])
def test_base_game_from_json(json_):
    gameio.read_base_game(json_)


@pytest.mark.parametrize('json_',
                         [EMPTY_GAME_JSON, GAME_JSON, SAMPLE_GAME_JSON])
def test_game_from_json(json_):
    gameio.read_game(json_)


@pytest.mark.parametrize('json_',
                         [EMPTY_GAME_JSON, GAME_JSON, SAMPLE_GAME_JSON])
def test_sample_game_from_json(json_):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='Truncating observation data')
        gameio.read_sample_game(json_)


@pytest.mark.parametrize('_', range(20))
def test_sorted_strategy(_):
    with open('test/hard_nash_game_1.json') as f:
        _, conv = gameio.read_game(json.load(f))
    assert all(a < b for a, b in zip(conv.role_names[:-1],
                                     conv.role_names[1:])), \
        "loaded json game didn't have sorted roles"
    assert all(all(a < b for a, b in zip(strats[:-1], strats[1:]))
               for strats in conv.strat_names), \
        "loaded json game didn't have sorted strategies"


def test_failed_auto_from_json():
    conv = gameio.GameSerializer(['a', 'b'], [['bar', 'foo'], ['baz']])
    json_prof = 5
    with pytest.raises(ValueError):
        conv.from_prof(json_prof)


def test_to_from_prof_json():
    game = gameio.GameSerializer(['a', 'b'], [['bar', 'foo'], ['baz']])
    prof = [6, 5, 3]
    json_prof = {'a': {'foo': 5, 'bar': 6}, 'b': {'baz': 3}}
    assert game.to_prof_json(prof) == json_prof
    assert np.all(game.from_prof_json(json_prof) == prof)
    assert np.all(game.from_prof(json_prof) == prof)
    assert game.from_prof_json(json_prof).dtype == int

    mix = [.6, .4, 1]
    json_mix = {'a': {'foo': .4, 'bar': .6}, 'b': {'baz': 1}}
    assert game.to_prof_json(mix) == json_mix
    assert np.all(game.from_prof_json(json_mix) == mix)
    assert game.from_prof_json(json_mix).dtype == float

    sym_grps = [
        dict(role='a', strategy='foo', count=5),
        dict(role='a', strategy='bar', count=6),
        dict(role='b', strategy='baz', count=3),
    ]
    assert ({tuple(sorted(x.items())) for x
             in game.to_prof_symgrp(prof)}
            == {tuple(sorted(x.items())) for x in sym_grps})
    assert np.all(game.from_prof_symgrp(sym_grps) == prof)
    assert np.all(game.from_prof(sym_grps) == prof)

    prof_str = 'a: 5 foo, 6 bar; b: 3 baz'
    assert np.all(game.from_prof_string(prof_str) == prof)
    assert np.all(game.from_prof(prof_str) == prof)
    assert set(game.to_prof_string(prof)) == set(prof_str)


def test_null_payoff():
    """Test that null payoff warnings are detected"""
    json_data = {
        'players': {'role': 1},
        'strategies': {'role': ['strat']},
        'profiles': [{'role': [['strat', 1, None]]}],
    }
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UserWarning):
            gameio.read_game(json_data)


def test_index():
    conv = gameio.GameSerializer(['a', 'b'], [['e', 'q', 'w'], ['r', 't']])
    assert 0 == conv.role_index('a')
    assert 1 == conv.role_index('b')
    assert 0 == conv.role_strat_index('a', 'e')
    assert 1 == conv.role_strat_index('a', 'q')
    assert 2 == conv.role_strat_index('a', 'w')
    assert 3 == conv.role_strat_index('b', 'r')
    assert 4 == conv.role_strat_index('b', 't')
