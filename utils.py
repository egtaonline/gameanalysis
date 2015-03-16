'''python file with a bunch of utility methods'''
import random
import string

def random_string(length, choices=string.ascii_letters + string.digits):
    '''Returns a random string of length length.

    Can optinally specify the characters it's drawn from

    '''
    return ''.join(random.choice(choices) for _ in xrange(length))


def only(gen):
    '''Returns the only element in a collection

    Throws a LookupError if collection contains more or less than one element

    '''
    gen = iter(gen)
    try:
        res = next(gen)
    except StopIteration:
        raise LookupError('Iterator was empty')
    try:
        next(gen)
    except StopIteration:
        return res
    raise LookupError('Iterator had more than one element')
