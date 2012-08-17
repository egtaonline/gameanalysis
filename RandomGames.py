#! /usr/bin/env python2.7

from BasicFunctions import leading_zeros
from RoleSymmetricGame import Game, PayoffData

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
		p = {"row":[PayoffData(row_strat, 1, row_val)], \
				"column":[PayoffData(col_strat, 1, -row_val)]}
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
			payoffs.append(PayoffData(strat, count, rnd.uniform(-1,1)))
		g.addProfile({"All":payoffs})
	return g


def congestion(N, f, r):
	roles = ["All"]
	players = {"All":N}
	strategies = 


from GameIO import to_JSON_str
from argparse import ArgumentParser
import sys

def parse_args():
	parser = io_parser(description="Generate random games.")
	parser.add_argument("type", choices=["uZS", "uSym", "CG"], help= \
			"Type of random game to generate. uZS = uniform zero sum. " +\
			"uSym = uniform symmetric. CG = congestion game.")
	parser.add_argument("count", type=int, help="Number of random games " +\
			"to create.")
	parser.add_argument("-output", type=str, default="", help=\
			"Output file. Defaults to stdout.")
	parser.add_argument("game_args", nargs="*", help="Additional arguments " +\
			"for game generator function.")
	if "-input" in sys.argv:
		sys.argv[sys.argv.index("-input")+1] = None
	else:
		sys.arg = sys.argv[:3] + ["-input", None] + sys.argv[3:]
	return parser.parse_args()


def main():
	args = parse_args()
	game_args = map(int, args.game_args)
	if args.type == "uZS":
		game_func = uniform_zero_sum
		assert len(game_args == 1), "one game_arg specifies strategy count"
	elif args.type == "uSym":
		game_func = uniform_symmetric
		assert len(game_args == 2), "game_args specify player and strategy "+\
				"counts"
		game_args = map(int, args.game_args[:2])
	elif args.type == "CG":
		game_func = congestion
		assert len(game_args == 2), "game_args specify player, facility, and"+\
				" required facility counts"
	games = [game_func(*game_args) for i in range(args.count)]
	if len(games) == 1:
		print to_JSON_str(games[0])
	else:
		print to_JSON_str(games)


if __name__ == "__main__":
	main()
