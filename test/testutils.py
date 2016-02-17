import functools


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
