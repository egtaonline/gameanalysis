#! /usr/bin/env python2.7

from os.path import exists
from argparse import ArgumentParser
from os.path import join, exists

import GameIO
from dpr import DPR

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("DPR_size", type=int)
	parser.add_argument("folder", type=str)
	parser.add_argument("mix_file", type=str, help="File with mixtures for "+
			"which expected values should be calculated.")
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--stop", type=int, default=100)
	return parser.parse_args()

def main():
	args = parse_args()
	mixtures = GameIO.read(args.mix_file)
	for index in range(args.start, args.stop):
		in_file = join(args.folder, str(index)+".json")
		out_file = join(args.folder, str(index)+"_DPR-EVs.json")
		try:
			EVs = GameIO.read(out_file)
			if len(EVs) == len(mixtures):
				continue
		except:
			EVs = []
		game = DPR(GameIO.read(in_file), args.DPR_size)
		for m in mixtures[len(EVs):]:
			EVs.append(game.expectedValues(game.toArray(m)))
			if len(EVs) % 10 == 0 or len(EVs) == len(mixtures):
				with open(out_file,"w") as f:
					f.write(GameIO.to_JSON_str(EVs))

if __name__ == "__main__":
	main()
