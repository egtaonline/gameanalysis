#! /usr/bin/env python2.7

import GameIO
import Nash
from dpr import DPR
from argparse import ArgumentParser
from os.path import join

def parse_args():
	parser = ArgumenParser()
	parser.add_argument("players", type=int)
	parser.add_argument("folder", type=str)
	return parser.parse_args()

def main():
	args = parse_args()
	for i in range(100):
		game_file = join(args.folder, str(i)+".json")
		game = DPR(GameIO.read(game_file), args.players)
		equilibria = [g.toProfile(eq) for eq in Nash.mixed_nash(g, \
												at_least_one=True)]
		eq_file = join(args.folder, str(i)+"_DPR-EQ.json")
		with open(eq_file,"w") as f:
			f.write(GameIO.to_JSON_str(equilibria))

if __name__ == "__main__":
	main()
