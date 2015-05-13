import numpy as np


def find_primes(n):
    '''Returns a list of all primes up to n'''
    primes = [2]
    for i in range(3, n+1):
        is_prime = True
        for prime in primes:
            if i % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    return primes


def prime_factor(n, dtype=np.uint64):
    '''Create a lookup array of prime factorizations

    Returns two things:
    0) a n+1 x (number of primes <= n) array with dtype dtype.
       Row i of this array is the prime factorization of i, where the int in
       the jth column of row i is the number of prime j in the factorization.
    1) an array with all of the primes less than or equal to n.  Corresponds
       with the columns of the first array.

    '''
    primes = np.array(find_primes(n), dtype=dtype)
    factors = np.zeros((n+1, len(primes)), dtype=dtype)
    for i in range(2, n+1):
        for j, p in enumerate(primes):
            if i % p == 0:
                resid = i // p
                factors[i, j] = 1
                factors[i] += factors[resid]
                break
    return factors, primes


def multinomial_coef(array, dtype=np.uint64):
    '''Calculate the multinomial coefficient for an array of ints

    WARNING: Because numpy does not have overflow checking for int math, this
    computes in modulo dtype space.

    multinomial_coef([a, b]) == (a + b) choose (a or b)

    This routine is vectoried. It will compute the multinomial coefficients
    assuming the coefficients are in the -1st axis. If you want compute many
    unequal multinomial coeficients at once, padding with 0s will return the
    proper result.

    Thus, if your input has shape (d0, d1, ..., dn-2, dn-2), the result will
    have shape (d0, d1, ..., dn-2).

    This does math in prime facorization space to avoid overflows, but they are
    obviously still possible.

    '''
    array = np.asarray(array)
    shp = array.shape[:-1]
    num_mult = array.shape[-1]
    array = array.reshape((-1, num_mult))
    sums = array.sum(1)
    prime_factors, primes = prime_factor(sums.max(), dtype=dtype)
    factorials = np.cumsum(prime_factors, 0)
    combs = factorials[sums] - factorials[array].sum(1)
    return np.prod(primes ** combs, 1).reshape(shp)
