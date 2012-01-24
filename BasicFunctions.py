from math import factorial
from operator import mul

def prod(collection):
	"""
	Product of all elements in the collection.
	elements must support multiplication
	"""
	return reduce(mul, collection)


def nCr(n,k):
	"""
	Number of combinations: n choose k.
	"""
	return prod(range(n-k+1,n+1)) / factorial(k)


def game_size(n,s):
	"""
	Number of profiles in a symmetric game with n players and s strategies.
	"""
	return nCr(n+s-1,n)


def profile_repetitions(p):
	"""
	Number of normal form profiles that correspond to a role-symmetric profile.
	"""
	return prod([factorial(sum(row)) / prod(map(factorial, row)) for row in p])


def list_repr(l, sep=", "):
	"""
	Creates a string representation of the elements of a collection.
	"""
	try:
		return reduce(lambda x,y: str(x) + sep + str(y), l)
	except TypeError:
		return ""

