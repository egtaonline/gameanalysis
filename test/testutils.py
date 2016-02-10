import functools


# TODO Change name to reflect no args but given arg?
def apply(inputs=((),), repeat=1):
    """Decorator that runs a test with the given arguments"""
    def wrap(test):
        @functools.wraps(test)
        def generator_test():
            for args in inputs:
                for _ in range(repeat):
                    yield lambda: test(*args)
        return generator_test
    return wrap
