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
def game_sizes(size='medium'):
    yield [1], 1
    yield [1], 2
    yield [2], 1
    yield [2], 2
    yield [2], 5
    yield [5], 2
    yield [5], 5
    yield 2 * [1], 1
    yield 2 * [1], 2
    yield 2 * [2], 1
    yield 2 * [2], 2
    yield 5 * [1], 2

    if size != 'small':
        yield 2 * [1], 5
        yield 2 * [2], 5
        yield 2 * [5], 2
        yield 2 * [5], 5
        yield 3 * [3], 3
        yield 5 * [1], 5
        yield [170], 2
        yield [180], 2
        yield [1, 2], 2
        yield [1, 2], [2, 1]
        yield 2, [1, 2]
        yield [3, 4], [2, 3]
        yield [2, 3, 4], [4, 3, 2]

    if size == 'big' and os.getenv('BIG_TESTS') == 'ON':  # Big Games
        yield 1000, 2  # pragma: no cover
        yield 5, 40  # pragma: no cover
        yield 3, 160  # pragma: no cover
        yield 50, 2  # pragma: no cover
        yield 20, 5  # pragma: no cover
        yield 90, 5  # pragma: no cover
        yield [2] * 2, 40  # pragma: no cover
        yield 12, 12  # pragma: no cover
