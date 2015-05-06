import operator
import scipy
import math
import numpy as np
from subprocess import PIPE, Popen


def prod(collection):
    '''Product of all elements in the collection'''
    return reduce(operator.mul, collection)


def choose(n, r):
    '''Returns the number of ways to choose r things from n things'''
    return scipy.comb(n, r, exact=True)


def game_size(n, s):
    '''Number of profiles in a symmetric game with n players and s strategies

    '''
    return choose(n+s-1, n)


def profile_repetitions(p):
    '''Number of normal form profiles that correspond to a role-symmetric profile

    '''
    return prod(math.factorial(sum(row)) / prod(map(math.factorial, row))
                for row in p)


def mean(numbers):
    '''Arithmetic mean'''
    n = 0
    mean = 0.0
    for x in numbers:
        n += 1
        mean += (x - mean)/n
    return mean


def one_line(string, line_width=80):
    '''If string s is longer than line width, cut it off and append "..."'''
    string = string.replace('\n', '')
    if len(string) > line_width:
        return string[:3*line_width/4] + "..." + string[-line_width/4+3:]
    return string


def leading_zeros(i, m):
    '''Pad the string of integer i with leading zeros to equal length of m.'''
    return str(i).zfill(len(str(m)))


def call(proc_name, in_pipe):
    p = Popen(proc_name, shell=True, stdin=PIPE, stdout=PIPE)
    return p.communicate(in_pipe)[0]


def weighted_least_squares(x, y, weights):
    '''appends the ones for you; puts 1D weights into a diagonal matrix'''
    try:
        A = np.append(x, np.ones([x.shape[0],1]), axis=1)
        W = np.zeros([x.shape[0]]*2)
        np.fill_diagonal(W, weights)
        return y.T.dot(W).dot(A).dot(np.linalg.inv(A.T.dot(W).dot(A)))
    except np.linalg.linalg.LinAlgError:
        z = A.T.dot(W).dot(A)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i,j] += np.random.uniform(-tiny,tiny)
        return y.T.dot(W).dot(A).dot(np.linalg.inv(z))
