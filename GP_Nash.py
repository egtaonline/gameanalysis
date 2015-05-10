#! /usr/bin/env python2.7

import GameIO
import Nash
from argparse import ArgumentParser
from os.path import join
from LearnedModels import ZeroPredictor

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("diffs", choices=["None", "player", "strat"])
	parser.add_argument("EVs", choices=["point", "sample", "DPR"])
	parser.add_argument("folder", type=str)
	parser.add_argument("--no_mean", action="store_true")
	parser.add_argument("--DPR_size", type=int, default=0)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--stop", type=int, default=100)
	args = parser.parse_args()
	if args.EVs == "DPR":
		assert args.DPR_size > 0, "DPR requires a value for --DPR_size"
	return args

def main():
	args = parse_args()
	for i in range(start, stop):
		game_file = join(args.folder, str(i)+"_GP-"+args.diffs[0]+".pkl")
		game = GameIO.read(game_file)
		game.EVs = args.EVs
		game.DPR_size = args.DPR_size
		if args.no_mean:
			for role in game.roles:
				game.GPs[role][None] = ZeroPredictor()
		equilibria = [game.toProfile(eq) for eq in Nash.mixed_nash(game, \
													at_least_one=True)]
		eq_file = join(args.folder, str(i)+"_GP-"+args.diffs[0]+\
						"_EQ-"+args.EVs[0]+(str(args.DPR_size) if \
						args.EVs == "DPR" else "")+("d" if args.no_mean else \
						"")+".json")
		with open(eq_file,"w") as f:
			f.write(GameIO.to_JSON_str(equilibria))

if __name__ == "__main__":
	main()
