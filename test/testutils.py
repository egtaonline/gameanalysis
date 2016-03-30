import functools
import os


def apply(inputs=((),), repeat=1):
    """Decorator that runs a test with the given arguments"""
    def wrap(test):
        @functools.wraps(test)
        def generator_test():
            for args in inputs:
                for _ in range(repeat):
                    yield (test,) + tuple(args)
        return generator_test
    return wrap


# iterator over a good amount of game sizes
def game_sizes(allow_big=False, only_ints=False):
    yield 1, 1, 1
    yield 1, 1, 2
    yield 1, 2, 1
    yield 1, 2, 2
    yield 1, 2, 5
    yield 1, 5, 2
    yield 1, 5, 5
    yield 2, 1, 1
    yield 2, 1, 2
    yield 2, 1, 5
    yield 5, 1, 2
    yield 5, 1, 5
    yield 2, 2, 1
    yield 2, 2, 2
    yield 2, 2, 5
    yield 2, 5, 2
    yield 2, 5, 5
    yield 1, 170, 2
    yield 1, 180, 2

    if not only_ints:
        yield 2, [1, 2], 2
        yield 2, [1, 2], [2, 1]
        yield 2, 2, [1, 2]
        yield 2, [3, 4], [2, 3]

    if allow_big and os.getenv('BIG_TESTS') == 'ON':  # Big Games
        yield 1, 1000, 2
        yield 1, 5, 40
        yield 1, 3, 160
        yield 1, 50, 2
        yield 1, 20, 5
        yield 1, 90, 5
        yield 2, 2, 40
        yield 1, 12, 12
