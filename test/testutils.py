import functools


def repeat(count=1):
    '''Decorator that allows you to run a test multiple times'''
    def wrap(test):
        @functools.wraps(test)
        def generator_test():
            for i in range(count):
                yield test
        return generator_test
    return wrap


def apply(inputs=()):
    '''Decorator that runs a test with the given arguments'''
    def wrap(test):
        @functools.wraps(test)
        def generator_test():
            for args in inputs:
                yield lambda: test(*args)
        return generator_test
    return wrap
