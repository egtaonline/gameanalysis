#! /usr/bin/env python2.7

from GameIO import *
from Reductions import *
from Subgames import *
from Dominance import *
from Regret import *
from Nash import *

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
			help="Min probability for a strategy to be considered in support.")
	parser.add_argument("-c", metavar="CONVERGE", type=float, default=1e-8, \
			help="Replicator dynamics convergence thrshold.")
	parser.add_argument("-i", metavar="ITERS", type=int, default=10000, \
			help="Max replicator dynamics iterations.")
	args = parser.parse_args()
	game = readGame(args.file)
	return game, args


def main(input_game, args):
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
			print str(i+1) + ". regret =", round(regret(input_game, eq), 4)
			for role in input_game.roles:
				print "    " + role + ":", list_repr(map(lambda pair: \
						str(pair[1]) + "x " + str(pair[0]), eq[role].items()))
	else:
		print "\nno pure strategy Nash equilibria found."
		mrp = MinRegretProfile(rational_game)
		print "regret =", regret(input_game, mrp)
		print "minimum regret pure strategy profile (regret = " + \
				str(round(regret(input_game, mrp), 4)) + "):"
		for role in input_game.roles:
			print "    " + role + ":", list_repr(map(lambda pair: \
					str(pair[1]) + "x " + str(pair[0]), mrp[role].items()))

	#find maximal subgames
	maximal_subgames = Cliques(rational_game)
	num_subgames = len(maximal_subgames)
	if num_subgames == 1 and maximal_subgames[0] == input_game:
		print "\ninput game is maximal"
	else:
		print "\n" + str(num_subgames), "maximal subgame" + ("" if num_subgames\
				== 1 else "s") + (" among non-dominated strategies" if any(map(\
				len, eliminated.values())) else "")

	#mixed strategy Nash equilibrium search
	for i, subgame in enumerate(maximal_subgames):
		print "\nsubgame "+str(i+1)+":\n", list_repr(map(lambda x: x[0] + \
				":\n\t\t" + list_repr(x[1], sep="\n\t\t"), sorted( \
				subgame.strategies.items())), "\n").expandtabs(4)
		mixed_equilibria = MixedNash(subgame, args.r, args.d, iters=args.i, \
			converge_thresh=args.c)
		print "\n" + str(len(mixed_equilibria)), "approximate mixed strategy"+ \
				" Nash equilibri" + ("um:" if len(mixed_equilibria) == 1 \
				else "a:")
		for j, eq in enumerate(mixed_equilibria):
			full_eq = translate(eq, subgame, input_game)
			if all(map(lambda p: p in input_game, neighbors(input_game, \
					full_eq))):
				print str(j+1) + ". regret =", round(regret(input_game, \
						full_eq), 4)
			else:
				print str(j+1) + ". regret >=", round(regret(input_game,  \
						full_eq, bound=True), 4)

			for k,role in enumerate(input_game.roles):
				print role + ":"
				for l,strategy in enumerate(input_game.strategies[role]):
					if full_eq[k][l] >= args.s:
						print "    " + strategy + ": " + str(round(100 * \
								full_eq[k][l], 2)) + "%"

			BR = bestResponses(input_game, full_eq)
			print "best responses:"
			for role in input_game.roles:
				if len(BR[role][0]) == 0:
					continue
				r = regret(input_game, full_eq, role, deviation=BR[role][0][0])
				print "\t" + str(role) + ": " + list_repr(BR[role][0]) + \
						";\tgain =", (round(r, 4) if not isinf(r) else "?")


if __name__ == "__main__":
	print "command: " + list_repr(argv, sep=" ") + "\n"
	game, args = parse_args()
	main(game, args)
