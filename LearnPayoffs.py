#! /usr/bin/env python2.7

import cPickle
import GameIO

from LearnedModels import GP_Game


def parse_args():
	parser = GameIO.io_parser()
	parser.add_argument("diffs", choices=["None", "strat", "player"], help=
					"Set to strat or player to learn differences from "
					"profile-average payoffs; None learns payoffs directly.")
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
