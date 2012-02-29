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
	parser.add_argument("-d", metavar="DIST", type=float, default=1e-3, \
			help="L2-distance threshold to consider equilibria distinct.")
	parser.add_argument("-s", metavar="SUPPORT", type=float, default=1e-3, \
			help="Min probability for a strategy of a strategy in support.")
	parser.add_argument("-c", metavar="CONVERGE", type=float, default=1e-8, \
			help="Replicator dynamics convergence thrshold.")
	parser.add_argument("-i", metavar="ITERS", type=int, default=10000, \
			help="Max replicator dynamics iterations.")
	args = parser.parse_args()
	game = readGame(args.file)
	return game, args


if __name__ == "__main__":
	print "command: " + list_repr(argv, sep=" ") + "\n"
	input_game, args = parse_args()
	print "input game =", abspath(args.file), "\n", input_game, "\n\n"

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
	pure_equilibria = PureNash(rational_game, args.r)
	l = len(pure_equilibria)
	if l > 0:
		print "\n" + str(len(pure_equilibria)), "pure strategy Nash equilibri" \
				+ ("um:" if l == 1 else "a:")
		for i, eq in enumerate(pure_equilibria):
			print str(i+1) + ". regret =", round(input_game.regret(eq), 4)
			for role in input_game.roles:
				print "    " + role + ":", list_repr(map(lambda pair: \
						str(pair[1]) + "x " + str(pair[0]), eq[role].items()))
	else:
		print "\nno pure strategy Nash equilibria found."
		mrp = MinRegretProfile(rational_game)
		print "regret =", input_game.regret(mrp)
		print "minimum regret pure strategy profile (regret = " + \
				str(round(input_game.regret(mrp), 4)) + "):"
		for role in input_game.roles:
			print "    " + role + ":", list_repr(map(lambda pair: \
					str(pair[1]) + "x " + str(pair[0]), mrp[role].items()))

	#find maximal subgames
	maximal_subgames = Cliques(rational_game)
	l = len(maximal_subgames)
	if l == 1 and maximal_subgames[0] == input_game:
		l = 0
		print "\ninput game is maximal"
	else:
		print "\n" + str(l), "maximal subgame" + ("" if l == 1 else "s") + \
				(" among non-dominated strategies" if any(map(len, \
				eliminated.values())) else "")

	#mixed strategy Nash equilibrium search
	for i, subgame in enumerate(maximal_subgames):
		if l != 0:
			print "\nsubgame "+str(i+1)+":\n", list_repr(map(lambda x: x[0] + \
					":\n\t\t" + list_repr(x[1], sep="\n\t\t"), sorted( \
					subgame.strategies.items())), "\n").expandtabs(4)
		mixed_equilibria = MixedNash(subgame, args.r, args.d, iters=args.i, \
			converge_thresh=args.c)
		l = len(mixed_equilibria)
		print "\n" + str(l), "approximate mixed strategy Nash equilibri" + \
				("um:" if l == 1 else "a:")
		for j, eq in enumerate(mixed_equilibria):
			full_eq = input_game.translate(subgame, eq)
			if all(map(lambda p: p in input_game, input_game.neighbors(\
					full_eq))):
				print str(j+1) + ". regret =", round(input_game.regret(\
						full_eq), 4)
			else:
				print str(j+1) + ". regret >=", round(input_game.regret( \
						full_eq, bound=True), 4)
			for k,role in enumerate(input_game.roles):
				print role + ":"
				for l,strategy in enumerate(input_game.strategies[role]):
					if full_eq[k][l] >= args.s:
						print "    " + strategy + ":" + str(round(100 * \
								full_eq[k][l], 1)) + "%"
			print "best responses:"
			print "\t" + list_repr(sorted([str(r)+": "+list_repr(br[0]) for \
					r,br in input_game.bestResponses(full_eq).items()]), \
					"\n\t")

