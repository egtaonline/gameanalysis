#! /usr/bin/env python2.7

from os.path import exists
from argparse import ArgumentParser
from os.path import join, exists

import GameIO

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("folder", type=str)
	parser.add_argument("mix_file", type=str, help="File with mixtures for "+
			"which expected values should be calculated.")
	parser.add_argument("game_type", type=str, help="Input filename, "+
			"excluding index and extension. Example: CV-N")
	parser.add_argument("EVs_method", type=str, choices=["point", "sample",
			"DPR"], help="Choices: point, sample, DPR.")
	parser.add_argument("--DPR_size", type=int, default=0, help="Only "+
			"required if EVs_method=DPR.")
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--stop", type=int, default=100)
	args = parser.parse_args()
	if args.EVs_method == "DPR":
		assert args.DPR_size > 1, "--DPR_size must be specified"
	return args

def main():
	args = parse_args()
	mixtures = GameIO.read(args.mix_file)
	for index in range(args.start, args.stop):
		in_file = join(args.folder, str(index)+"_"+args.game_type+".pkl")
		out_file = join(args.folder, str(index)+"_"+args.game_type+"_EVs.json")
		if exists(out_file):
			EVs = GameIO.read(EVs_file)
			if len(EVs) == len(mixtures):
				continue
		else:
			EVs = []
		game = GameIO.read(join(args.folder, str(index) + "_" + args.game_type))
		game.DPR_size = args.DPR_size
		game.EVs = args.EVs_method
		for m in mixtures[len(EVs):]:
			EVs.append(game.expectedValues(game.toArray(m)))
			with open(args.EVs_file,"w") as f:
				f.write(GameIO.to_JSON_str(EVs))

if __name__ == "__main__":
	main()
