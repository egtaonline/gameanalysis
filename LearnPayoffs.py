#! /usr/bin/env python2.7

import cPickle, json
from os.path import exists, join
from argparse import ArgumentParser
from NormalizedLearning import GP_Game


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("folder", type=str)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--stop", type=int, default=100)
	return parser.parse_args()


def main():
	args = parse_args()
	for index in range(args.start, args.stop):
		out_file = join(args.folder, str(index) + "_GP-NN.pkl")
		if exists(out_file):
			continue
		with open(join(args.folder, str(index) + ".json")) as f:
			sample_game = json.load(f)
		game = GP_Game(sample_game)
		with open(out_file, "w") as f:
			cPickle.dump(game, f)


if __name__ == "__main__":
	main()
