#! /usr/bin/env python2.7

from BasicFunctions import leading_zeros
from RoleSymmetricGame import Game, payoff_data

import numpy.random as rnd

def uniform_zero_sum(S):
	roles = ["row", "column"]
	players = {r:1 for r in roles}
	strategies = {"row":["r" + leading_zeros(i,S) for i in range(S)], \
			"column":["c" + leading_zeros(i,S) for i in range(S)]}
	g = Game(roles, players, strategies)
	for prof in g.allProfiles():
		row_strat = prof["row"].keys()[0]
		row_val = rnd.uniform(-1,1)
		col_strat = prof["column"].keys()[0]
		p = {"row":[payoff_data(row_strat, 1, row_val)], \
				"column":[payoff_data(col_strat, 1, -row_val)]}
		g.addProfile(p)
	return g


def uniform_symmetric(N, S):
	roles = ["All"]
	players = {"All":N}
	strategies = {"All":["s" + leading_zeros(i,S) for i in range(S)]}
	g = Game(roles, players, strategies)
	for prof in g.allProfiles():
		payoffs = []
		for strat, count in prof["All"].items():
			payoffs.append(payoff_data(strat, count, rnd.uniform(-1,1)))
		g.addProfile({"All":payoffs})
	return g


from GameIO import toJSONstr
from argparse import ArgumentParser
import sys

def main():
	parser = ArgumentParser()
	parser.add_argument("type", choices=["uZS", "uSym"], help="Type of " +\
			"random game to generate. uZS = uniform zero sum. uSym = " +\
			"uniform symmetric.")
	parser.add_argument("count", type=int, help="Number of random games " +\
			"to create.")
	parser.add_argument("-output", type=str, default="", help=\
			"Output file. Defaults to stdout.")
	parser.add_argument("game_args", nargs="*", help="Additional arguments " +\
			"for game generator function.")
	args = parser.parse_args()
	if args.output != "":
		sys.stdout = open(a.output, "w")
	if args.type == "uZS":
		game_func = uniform_zero_sum
		game_args = [int(args.game_args[0])]
	elif args.type == "uSym":
		game_func = uniform_symmetric
		game_args = map(int, args.game_args[:2])
	games = [game_func(*game_args) for i in range(args.count)]
	if len(games) == 1:
		print toJSONstr(games[0])
	else:
		print toJSONstr(games)


if __name__ == "__main__":
	main()
