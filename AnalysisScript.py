#!/usr/local/bin/python2.7

from GameIO import *
from GameAnalysis import *

from sys import argv
from os import listdir

#folder = "/Users/bryce/Documents/publications/Reductions_AAMAS_2012/games" + \
#		"/12p_6s/CongestionGames/"
#print "full, HR2, HR4, DPR2"
#for filename in filter(lambda s: len(s) > 4 and s[-4:] == '.xml', listdir( \
#		folder)):
#	full_game = readGame(folder + filename)
#	hr2 = readGame(folder + "HR/HR_2_" + filename)
#	hr4 = readGame(folder + "HR/HR_4_" + filename)
#	dpr = readGame(folder + "DPR_big/DPR_big_2_" + filename)
#	print IteratedElimination(full_game, PureStrategyDominance).numStrategies[ \
#			0],",",IteratedElimination(hr2, PureStrategyDominance). \
#			numStrategies[0],",",IteratedElimination(hr4, \
#			PureStrategyDominance).numStrategies[0],",",IteratedElimination( \
#			dpr, PureStrategyDominance).numStrategies[0]


g = readGame(argv[1])
print len(g), "profiles"
print g.regret(ReplicatorDynamics(g, g.uniformMixture(), verbose=True))

