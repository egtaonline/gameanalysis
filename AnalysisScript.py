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
	parser.add_argument("-r", metavar="REGRET", type=float, default=1e-3, \
			help="Max allowed regret for approximate Nash equilibria.")
	parser.add_argument("-d", metavar="DISTANCE", type=float, default=1e-3, \
			help="L2-distance threshold to consider equilibria distinct.")
	args = parser.parse_args()
	game = readGame(args.file)
	args = {"file":args.file, "regret_thresh":args.r, "dist_thresh":args.d}
	return game, args


if __name__ == "__main__":
	print "command: " + list_repr(argv, sep=" ") + "\n"
	input_game, args = parse_args()
	print "input game =", abspath(args.pop("file")), "\n", input_game, "\n\n"

	#iterated elimination of dominated strategies
	rational_game = IteratedElimination(input_game, PureStrategyDominance)
	eliminated = {r:sorted(set(input_game.strategies[r]) - set( \
			rational_game.strategies[r])) for r in input_game.roles}
	if any(map(len, eliminated.values())):
		print "dominated strategies:"
		for r in rational_game.roles:
			if eliminated[r]:
				print r, ":", list_repr(eliminated[r])

	#pure strategy Nash equilibrium search
	pure_equilibria = PureNash(rational_game, args["regret_thresh"])
	l = len(pure_equilibria)
	if l > 0:
		print "\n" + str(len(pure_equilibria)), "pure strategy Nash equilibri" \
				+ ("um:" if l == 1 else "a:")
		for i, eq in enumerate(pure_equilibria):
			print str(i+1)+". regret =", input_game.regret(eq)
			for role in rational_game.roles:
				print "    " + role + ":", list_repr(map(lambda sc: \
						str(sc[1]) + "x " + str(sc[0]), eq[role].items()))
	else:
		print "\nno pure strategy Nash equilibria found."
		print "minimum regret pure strategy profile:", \
				MinRegretProfile(rational_game)

	#find maximal subgames
	maximal_subgames = Cliques(rational_game)
	l = len(maximal_subgames)
	print "\n" + str(l), "maximal subgame" + ("" if l == 1 else "s")

	#mixed strategy Nash equilibrium search
	for subgame in maximal_subgames:
		print "\nsubgame:", list_repr(map(lambda x: x[0] + ":\n\t\t\t" + \
				list_repr(x[1], sep="\n\t\t\t"), sorted( \
				input_game.strategies.items())), "\n\t\t").expandtabs(4)
		mixed_equilibria = MixedNash(subgame, **args)
		l = len(mixed_equilibria)
		print "\n" + str(l), "approximate mixed strategy Nash equilibri" + \
				("um:" if l == 1 else "a:")
		print list_repr(map(lambda eq: input_game.translate(subgame, eq), \
				mixed_equilibria))

