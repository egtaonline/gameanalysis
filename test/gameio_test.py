import warnings

from gameanalysis import rsgame
from test import testutils


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


@testutils.apply(zip([EMPTY_GAME_JSON, GAME_JSON, SAMPLE_GAME_JSON]))
def empty_game_from_json_test(json_):
    rsgame.EmptyGame.from_json(json_)


@testutils.apply(zip([EMPTY_GAME_JSON, GAME_JSON, SAMPLE_GAME_JSON]))
def game_from_json_test(json_):
    rsgame.Game.from_json(json_)


@testutils.apply(zip([EMPTY_GAME_JSON, GAME_JSON, SAMPLE_GAME_JSON]))
def sample_game_from_json_test(json_):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='Truncating observation data')
        rsgame.SampleGame.from_json(json_)
