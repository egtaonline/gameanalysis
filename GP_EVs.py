#! /usr/bin/env python2.7

from os.path import exists

import GameIO

def parse_args():
	parser = GameIO.io_parser()
	parser.add_argument("mix_file", type=str, help="File with mixtures for "+
			"which expected values should be calculated.")
	parser.add_argument("EVs_file", type=str, help="File to which output "+
			"should be written. Can contain partial output.")
	parser.add_argument("EVs_method", type=str, choices=["point", "sample",
			"DPR"] help="Choices: point, sample, DPR.")
	return parser.parse_args()

def main():
	args = parse_args()
	game = args.input
	game.EVs = args.EVs_method
	mixtures = GameIO.read(args.mix_file)
	if exists(args.EVs_file):
		EVs = GameIO.read(args.EVs_file)
	else:
		EVs = []
	for m in mixtures:
		EVs.append(game.expectedValues(game.toArray(m)))
		with open(args.EVs_file,"w") as f:
			f.write(GameIO.to_JSON_str(EVs))

if __name__ == "__main__":
	main()
