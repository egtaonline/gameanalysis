import os

import pytest


big_tests = bool(os.getenv('BIG_TESTS'))

small_games = [
    ([1], 1),
    ([1], 2),
    ([2], 1),
    ([2], 2),
    ([2], 5),
    ([5], 2),
    ([5], 5),
    (2 * [1], 1),
    (2 * [1], 2),
    (2 * [2], 1),
    (2 * [2], 2),
    (5 * [1], 2),
]
games = small_games + [
    (2 * [1], 5),
    (2 * [2], 5),
    (2 * [5], 2),
    (2 * [5], 5),
    (3 * [3], 3),
    (5 * [1], 5),
    ([170], 2),
    ([180], 2),
    ([1, 2], 2),
    ([1, 2], [2, 1]),
    (2, [1, 2]),
    ([3, 4], [2, 3]),
    ([2, 3, 4], [4, 3, 2]),
]
big_games = games + ([] if not big_tests else [
    (1000, 2),
    (5, 40),
    (3, 160),
    (50, 2),
    (20, 5),
    (90, 5),
    ([2] * 2, 40),
    (12, 12),
])


def run_if_big(func):
    return pytest.mark.skipif(
        not big_tests,
        reason="test is large and big tests aren't enabled")(func)
