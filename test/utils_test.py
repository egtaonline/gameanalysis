import numpy as np
import pytest

from gameanalysis import utils


def test_only():
    assert utils.only([None]) is None, \
        "only didn't return only element"
    with pytest.raises(ValueError):
        utils.only([])
    with pytest.raises(ValueError):
        utils.only([None, None])
    with pytest.raises(ValueError):
        utils.only(5)


def test_game_size():
    assert utils.game_size(2000, 10) == 1442989326579174917694151


def test_one_line():
    short = "This is a short string, so it won't get truncated"
    assert utils.one_line(short, 100) == short, \
        "short string still got truncated"
    long_str = "This is relatively long"
    expected = "This is rela...g"
    assert utils.one_line(long_str, 16) == expected, \
        "one_line didn't truncate as expected"


def test_ordered_permutations():
    assert list(utils.ordered_permutations([])) == [], \
        "empty ordered permutations wasn't empty"

    result = list(utils.ordered_permutations([1, 2, 1, 2]))
    expected = [
        (1, 1, 2, 2),
        (1, 2, 1, 2),
        (1, 2, 2, 1),
        (2, 1, 1, 2),
        (2, 1, 2, 1),
        (2, 2, 1, 1),
    ]
    assert result == expected, \
        "ordered_permutations didn't produce the correct result"


def test_simplex_project():
    res = utils.simplex_project(np.array([0, 0, 0]))
    assert np.allclose(res, [1/3]*3), \
        "projecting [0, 0, 0] didn't result in uniform"

    res = utils.simplex_project(np.array([1.2, 1.4]))
    assert np.allclose(res, [.4, .6]), \
        "simplex project didn't return correct result"

    res = utils.simplex_project(np.array([-0.1, 0.8]))
    assert np.allclose(res, [0.05, 0.95]), \
        "simplex project didn't return correct result"
