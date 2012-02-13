#!/usr/local/bin/python2.7

from GameIO import *
from GameAnalysis import *

from sys import argv
from os.path import abspath
from argparse import ArgumentParser


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("file", type=str, help="Game file to be analyzed. " +\
			"Suported file types: EGAT symmetric XML, EGAT strategic XML, " +\
			"testbed role-symmetric JSON.")
	parser.add_argument("-r", metavar="REGRET", type=float, default=0, \
			help="Max allowed regret for approximate Nash equilibria.")
	parser.add_argument("-d", metavar="DISTANCE", type=float, default=1e-2, \
			help="L2-distance threshold to consider equilibria distinct.")
	args = parser.parse_args()
	game = readGame(args.file)
	args = {"file":args.file, "regret_thresh":args.r, "dist_thresh":args.d}
	return game, args


if __name__ == "__main__":
	print "command: " + list_repr(argv, sep=" ") + "\n"
	input_game, args = parse_args()
	print "input game =", abspath(args["file"]), "\n", input_game, "\n\n"

	#iterated elimination of dominated strategies
	rational_game = IteratedElimination(input_game, PureStrategyDominance)
	eliminated = {r:sorted(set(input_game.strategies[r]) - set( \
			rational_game.strategies[r])) for r in input_game.roles}
	if any(map(len, eliminated.values())):
		print "dominated strategies:"
		for r in rational_game.roles:
			if eliminated[r]:
				print r, ":", list_repr(eliminated[r])


