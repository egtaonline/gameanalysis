import os

import pytest


long_tests = bool(os.getenv('LONG'))

games = [
    ([1], [1]),
    ([1], [2]),
    ([2], [1]),
    ([1, 1], [1, 1]),
    ([1, 1], [2, 2]),
    ([2, 2], [1, 1]),
    ([1, 2], [2, 1]),
    ([2, 1], [1, 2]),
    ([3, 4], [2, 3]),
    ([2, 3, 4], [4, 3, 2]),
]


def long_test(func):
    return pytest.mark.skipif(
        not long_tests,
        reason="test takes a long time and wasn't enabled with LONG=Y")(func)
