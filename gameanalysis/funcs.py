import operator
import math
import functools
#import numpy as np
import scipy.misc as spm


def prod(collection):
    '''Product of all elements in the collection'''
    return functools.reduce(operator.mul, collection)


def game_size(n, s, exact=True):
    '''Number of profiles in a symmetric game with n players and s strategies

    '''
    return spm.comb(n+s-1, n, exact=exact)


def profile_repetitions(p):
    '''Number of normal form profiles that correspond to a role-symmetric profile

    '''
    return prod(math.factorial(sum(row)) // prod(map(math.factorial, row))
                for row in p)


def only(iterable):
    '''Return the only element of an iterable

    Throws a value error if the iterable doesn't contain only one element

    '''
    try:
        it = iter(iterable)
        value = next(it)
        try:
            next(it)
        except StopIteration:
            return value
        raise ValueError('Iterable had more than one element')
    except TypeError:
        raise ValueError('Input was not iterable')
    except StopIteration:
        raise ValueError('Input was empty')

# def mean(numbers):
#     '''Arithmetic mean'''
#     n = 0
#     mean = 0.0
#     for x in numbers:
#         n += 1
#         mean += (x - mean)/n
#     return mean


def one_line(string, line_width=80):
    '''If string s is longer than line width, cut it off and append "..."'''
    string = string.replace('\n', '')
    if len(string) > line_width:
        return string[:3*line_width/4] + "..." + string[-line_width/4+3:]
    return string


# def weighted_least_squares(x, y, weights):
#     '''appends the ones for you; puts 1D weights into a diagonal matrix'''
#     try:
#         A = np.append(x, np.ones([x.shape[0],1]), axis=1)
#         W = np.zeros([x.shape[0]]*2)
#         np.fill_diagonal(W, weights)
#         return y.T.dot(W).dot(A).dot(np.linalg.inv(A.T.dot(W).dot(A)))
#     except np.linalg.linalg.LinAlgError:
#         z = A.T.dot(W).dot(A)
#         for i in range(z.shape[0]):
#             for j in range(z.shape[1]):
#                 z[i,j] += np.random.uniform(-tiny,tiny)
#         return y.T.dot(W).dot(A).dot(np.linalg.inv(z))
