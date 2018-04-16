"""Test game reader"""
import functools
import io
import json
import warnings

import numpy as np
import pytest

from gameanalysis import canongame
from gameanalysis import gambit
from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import learning
from gameanalysis import matgame
from gameanalysis import rsgame


@functools.lru_cache()
def egame():
    """Empty game"""
    return rsgame.empty([3, 4], [4, 3])


@functools.lru_cache()
def game():
    """Base game"""
    return gamegen.gen_profiles(egame(), 0.5)


def sgame():
    """Sample game"""
    return gamegen.gen_noise(game())


@functools.lru_cache()
def agg():
    """Action graph game"""
    return gamegen.normal_aggfn([3, 4], [4, 3], 10)


def mat():
    """Matrix game"""
    return matgame.matgame(np.random.random((4, 3, 2, 3)))


@functools.lru_cache()
def rbf():
    """Rbf learning game"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'some lengths were at their bounds, this may indicate a poor fit',
            UserWarning)
        return learning.rbfgame_train(game())


def point():
    """Point learning game"""
    return learning.point(rbf())


def sample():
    """Sample learning game"""
    return learning.sample(rbf())


def neighbor():
    """Neighbor learning game"""
    return learning.neighbor(rbf())


def add():
    """Add game"""
    return canon() + agg()


def canon():
    """Canon game"""
    return canongame.canon(game())


def const():
    """Constant game"""
    return rsgame.const_replace(egame(), 0)


@pytest.mark.parametrize('base', [
    egame, game, sgame, agg, mat, rbf, point, sample, neighbor, add, canon,
    const, 'gambit'])
def test_automatic_deserialization(base):
    """Test that we can serialize and deserialize arbitrary games"""
    if base == 'gambit':
        base = mat()
        string = gambit.dumps(base)
    else:
        base = base()
        string = json.dumps(base.to_json())
    copy = gamereader.loads(string)
    assert base == copy

    copy = gamereader.load(io.StringIO(string))
    assert base == copy


def test_parse_fail():
    """Test invalid games"""
    with pytest.raises(ValueError):
        gamereader.loads('')
    with pytest.raises(ValueError):
        gamereader.loadj({'type': 'unknown.0'})
