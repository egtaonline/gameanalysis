from os import path

import pytest
import numpy as np

from gameanalysis import matgame
from gameanalysis import gamegen
from gameanalysis import gambit
from test import utils


@utils.warnings_filter(UserWarning)
@pytest.mark.parametrize(
    'filename', ['2x2x2-nau.nfg', '2x2x2.nfg', 'sample.nfg', 'ugly.nfg'])
def test_load(filename):
    with open(path.join('example_games', filename)) as f:
        game = gambit.load(f)
    assert game.is_complete()


@pytest.mark.parametrize('_', range(5))
@pytest.mark.parametrize('players,strats', [
    ([1], [1]),
    ([1], [2]),
    ([2], [1]),
    ([1, 1], [1, 1]),
    ([1, 1], [2, 2]),
    ([2, 2], [1, 1]),
    ([1, 2], [2, 1]),
    ([2, 1], [1, 2]),
])
def test_random_identity_test(players, strats, _):
    game = matgame.matgame_copy(gamegen.game(players, strats))
    string = gambit.dumps(game)
    copy = gambit.loads(string)
    assert game == copy


def test_parse_error():
    with pytest.raises(AssertionError):
        gambit.loads('')


@utils.warnings_filter(UserWarning)
def test_parse_ugly():
    with open(path.join('example_games', 'ugly.nfg')) as f:
        game = gambit.load(f)
    payoffs = np.zeros((2, 2, 2, 3))
    payoffs[1, 0, 0] = [8, 12, 9]
    payoffs[0, 0, 1] = [0.08, 0.2, -90]
    payoffs[1, 1, 1] = payoffs[0, 1, 0] = [4, -6.2, 3]
    expected = matgame.matgame_names(
        ['a0', 'a1', 'player\n "1"'],
        [['"2"\n', '1'], ['10', '11'], ['1', '2']],
        payoffs)
    assert game == expected
