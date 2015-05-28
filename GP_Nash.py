#! /usr/bin/env python2.7

import GameIO
import Nash
from NormalizedLearning import GP_Game

import cPickle
from argparse import ArgumentParser
from os.path import join, exists

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("game_type", type=str, help="Input filename, "+
			"excluding index and extension. Example: CV-N")
	parser.add_argument("EVs", choices=["point", "sample", "DPR"])
	parser.add_argument("folder", type=str)
	parser.add_argument("--no_mean", action="store_true")
	parser.add_argument("--DPR_size", type=int, default=0)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--stop", type=int, default=100)
	args = parser.parse_args()
	if args.EVs == "DPR":
		assert args.DPR_size > 1, "--DPR_size must be specified"
	return args

def main():
	args = parse_args()
	for i in range(args.start, args.stop):
		in_file = join(args.folder, str(i) +"_"+ args.game_type + ".pkl")
		out_file = join(args.folder, str(i) +"_"+ args.game_type +"_EQ-"+
						args.EVs[0] + (str(args.DPR_size) if args.EVs ==
						"DPR" else "")+("d" if args.no_mean else "")+".json")
		if exists(out_file):
			continue
		with open(in_file) as f:
			game = cPickle.load(f)
		game.EVs = args.EVs
		equilibria = [game.toProfile(eq) for eq in Nash.mixed_nash(game, \
													at_least_one=True)]
		with open(out_file, "w") as f:
			f.write(GameIO.to_JSON_str(equilibria))

if __name__ == "__main__":
	main()
