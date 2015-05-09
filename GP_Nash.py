#! /usr/bin/env python2.7

import GameIO
import Nash
from argparse import ArgumentParser
from os.path import join

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("devs", choices=["None", "player", "strat"])
	parser.add_argument("EVs", choices=["point", "sample", "DPR"])
	parser.add_argument("folder", type=str)
	parser.add_argument("--DPR_size", type=int, default=0)
	args = parser.parse_args()
	if args.EVs == "DPR":
		assert args.DPR_size > 0, "DPR requires a value for --DPR_size"
	return args

def main():
	args = parse_args()
	for i in range(100):
		game_file = join(args.folder, str(i)+"_GP-"+args.devs[0]+".pkl")
		game = GameIO.read(game_file)
		game.EVs = args.EVs
		game.DPR_size = args.DPR_size
		equilibria = [game.toProfile(eq) for eq in Nash.mixed_nash(game, \
													at_least_one=True)]
		eq_file = join(args.folder, str(i)+"_GP-"+args.devs[0]+\
						"_EQ-"+args.EVs[0]+(str(args.DPR_size) if \
						args.EVs == "DPR" else "")+".json")
		with open(eq_file,"w") as f:
			f.write(GameIO.to_JSON_str(equilibria))

if __name__ == "__main__":
	main()
