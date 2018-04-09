import functools
import io
import json
import warnings

import numpy as np
import pytest

from gameanalysis import agggen
from gameanalysis import canongame
from gameanalysis import gambit
from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import learning
from gameanalysis import matgame
from gameanalysis import rsgame


@functools.lru_cache()
def egame():
    return rsgame.emptygame([3, 4], [4, 3])


@functools.lru_cache()
def game():
    return gamegen.gen_profiles(egame(), 0.5)


def sgame():
    return gamegen.gen_noise(game())


@functools.lru_cache()
def agg():
    return agggen.normal_aggfn([3, 4], [4, 3], 10)


def mat():
    return matgame.matgame(np.random.random((4, 3, 2, 3)))


@functools.lru_cache()
def rbf():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'some lengths were at their bounds, this may indicate a poor fit',
            UserWarning)
        return learning.rbfgame_train(game())


def point():
    return learning.point(rbf())


def sample():
    return learning.sample(rbf())


def neighbor():
    return learning.neighbor(rbf())


def add():
    return canon() + agg()


def canon():
    return canongame.canon(game())


def const():
    return rsgame.const_replace(egame(), 0)


@pytest.mark.parametrize('game', [
    egame, game, sgame, agg, mat, rbf, point, sample, neighbor, add, canon,
    const, 'gambit'])
def test_automatic_deserialization(game):
    '''Test that we can serialize and deserialize arbitrary games'''
    if game == 'gambit':
        game = mat()
        string = gambit.dumps(game)
    else:
        game = game()
        string = json.dumps(game.to_json())
    copy = gamereader.loads(string)
    assert game == copy

    copy = gamereader.load(io.StringIO(string))
    assert game == copy


def test_parse_fail():
    with pytest.raises(ValueError):
        gamereader.loads('')
    with pytest.raises(ValueError):
        gamereader.loadj({'type': 'unknown.0'})
