#! /usr/bin/env python2.7

import cPickle
from os.path import exists, join
import GameIO
from LearnedModels import GP_Game


def parse_args():
	parser = GameIO.io_parser()
	parser.add_argument("diffs", choices=["None", "strat", "player"], help=
					"Set to strat or player to learn differences from "
					"profile-average payoffs; None learns payoffs directly.")
	parser.add_argument("--CV", action="store_true")
	parser.add_argument("folder", type=str)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--stop", type=int, default=100)
	args = parser.parse_args()
	if args.diffs == "None":
		args.diffs = None
	return args


def main():
	args = parse_args()
	for index in range(args.start, args.stop):
		out_file = join(args.folder, str(index) + ("_CV-" if args.CV else
									"_GP-") + args.diffs[0]+".pkl")
		if exists(out_file):
			continue
		with open(join(args.folder, str(index) + ".json")) as f:
			sample_game = GameIO.read(f)
		game = GP_Game(sample_game, args.diffs, args.CV)
		with open(out_file, "w") as f:
			cPickle.dump(game, f)


if __name__ == "__main__":
	main()
