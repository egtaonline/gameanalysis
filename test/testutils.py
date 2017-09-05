import functools
import os
import warnings

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


# FIXME Replace with @pytest.mark.slow
def long_test(func):
    return pytest.mark.skipif(
        not long_tests,
        reason="test takes a long time and wasn't enabled with LONG=Y")(func)


def warnings_filter(status='ignore'):
    """decorator to apply warning filter to a function

    Arguments
    ---------
    status : str, optional
        The status to treat all warnings as. Default ignores all wanrings."""
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return func(*args, **kwargs)
        return wrapped
    return decorator
