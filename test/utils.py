"""Utilities for testing"""
import itertools

from gameanalysis import rsgame


def basic_games():
    """Small basic games for testing"""
    yield rsgame.empty(1, 2)
    yield rsgame.empty(2, 2)
    yield rsgame.empty(2, 3)
    yield rsgame.empty(3, 2)
    yield rsgame.empty(3, 3)
    yield rsgame.empty([2, 3], [3, 2])
    yield rsgame.empty([1, 1, 1], 2)


def singleton_games():
    """Games that have singleton roles"""
    yield rsgame.empty([2, 1], [1, 2])
    yield rsgame.empty([1, 2], [2, 1])
    for strats in itertools.islice(itertools.product(*[[1, 2]] * 3), 1, None):
        yield rsgame.empty(1, strats)


def large_games():
    """Games that test functionality in large spaces"""
    yield rsgame.empty([1, 1], [5, 5])
    yield rsgame.empty([2, 2], [5, 5])
    yield rsgame.empty([5, 5], [2, 2])
    yield rsgame.empty([1, 1, 1, 1], 2)
    yield rsgame.empty([3, 3, 3], [3, 3, 3])
    yield rsgame.empty([2, 3, 4], [4, 3, 2])
    yield rsgame.empty(170, 2)
    yield rsgame.empty(180, 2)


def edge_games():
    """Small number of edge games"""
    yield rsgame.empty(4, 3)
    yield rsgame.empty([3, 2], [2, 3])
    yield rsgame.empty([2, 2, 2], 2)
