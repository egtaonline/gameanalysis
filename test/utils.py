import functools
import warnings


games = [
    ([1], [2]),
    ([2, 1], [1, 2]),
    ([1, 2], [2, 1]),
    ([1, 1], [2, 2]),
    ([2, 3], [3, 2]),
    ([1, 1, 1], [1, 1, 2]),
    ([1, 1, 1], [1, 2, 1]),
    ([1, 1, 1], [2, 1, 1]),
    ([1, 1, 1], [1, 2, 2]),
    ([1, 1, 1], [2, 1, 2]),
    ([1, 1, 1], [2, 2, 1]),
    ([1, 1, 1], [2, 2, 2]),
]


def warnings_filter(category=Warning, status='ignore'):
    """decorator to apply warning filter to a function

    Arguments
    ---------
    status : str, optional
        The status to treat all warnings as. Default ignores all wanrings."""
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category)
                return func(*args, **kwargs)
        return wrapped
    return decorator
