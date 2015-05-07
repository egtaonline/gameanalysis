#! /usr/bin/env python2.7

import GameIO
import Nash
from dpr import DPR

def parse_args():
	parser = GameIO.io_parser()
	parser.add_argument("players", type=int)
	parser.add_argument("--one", action="store_true")
	return parser.parse_args()

def main():
	args = parse_args()
	g = DPR(args.input, args.players)
	equilibria = [g.toProfile(eq) for eq in Nash.mixed_nash(g, \
					at_least_one=args.one)]
	print GameIO.to_JSON_str(equilibria)

if __name__ == "__main__":
	main()
