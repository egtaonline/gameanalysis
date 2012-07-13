#! /usr/bin/env python2.7

import numpy as np

from RoleSymmetricGame import is_mixed_profile, is_mixture_array, Profile, tiny
from BasicFunctions import one_line


def threshold(prof, supp_thresh=0.1):
	if is_mixture_array(prof):
		arr = np.array(prof)
		arr[arr < supp_thresh] = 0
		sums = arr.sum(1).reshape(arr.shape[0], 1)
		if np.any(sums == 0):
			raise ValueError("no probability greater than threshold.")
		return arr / sums
	if is_mixed_profile(prof):
		dct = prof.asDict()
		for role in dct:
			for strat in prof[role]:
				if prof[role][strat] < supp_thresh:
					del dct[role][strat]
		sums = {r:sum(dct[r].values()) for r in dct}
		if any([s == 0 for s in sums.values()]):
			raise ValueError("no probability greater than threshold.")
		for role in dct:
			for strat in dct[role]:
				dct[role][strat] /= sums[role]
		return Profile(dct)
	raise TypeError(one_line("unrecognized profile type: " + str(prof), 69))


def purify(prof):
	if is_mixture_array(prof):
		arr = np.array(prof)
		arr[arr < arr.max(1).reshape(arr.shape[0],1)] = 0
		return arr / arr.sum(1).reshape(arr.shape[0], 1)
	if is_mixed_profile(prof):
		dct = prof.asDict()
		maxima = {r:max(prof[r].values()) for r in prof}
		for role in prof:
			for strat in prof[role]:
				if prof[role][strat] < maxima[role]:
					del dct[role][strat]
		sums = {r:sum(dct[r].values()) for r in prof}
		for role in dct:
			for strat in dct[role]:
				dct[role][strat] /= sums[role]
		return Profile(dct)
	raise TypeError(one_line("unrecognized profile type: " + str(prof), 69))


