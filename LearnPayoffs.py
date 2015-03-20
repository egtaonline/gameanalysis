#! /usr/bin/env python2.7

import cPickle
import GameIO

from LearnedModels import GP_Game


def parse_args():
	parser = GameIO.io_parser()
	parser.add_argument("--EVs", choices=["point","sample","DPR"], default=
						"point", help="default method for game.expectedValues")
	parser.add_argument("--diffs", choices=["None", "strat", "player"],
						default="None", help="set to strat or player to learn"+
						"differences from profile-average payoffs")
	parser.add_argument("--CV", action="store_true", help="cross-validation")
	parser.add_argument("--DPR_size", type=int, nargs="*", default=[], help=
						"Number of players per role (in alphabetical order) "+
						"to be used by default for DPR_EVs.")
	args = parser.parse_args()
	if args.diffs == "None":
		args.diffs = None
	return args


def main():
	a = parse_args()
	a.DPR_size = dict(zip(a.input.roles, a.DPR_size))
	g = GP_Game(a.input, a.CV, a.diffs, a.EVs, a.DPR_size)
	if a.output != "":
		sys.stdout.close()
		with open(a.output, "w") as f:
			cPickle.dump(g, f)
	else:
		print cPickle.dumps(g)


if __name__ == "__main__":
	main()
