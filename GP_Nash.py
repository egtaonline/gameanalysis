#! /usr/bin/env python2.7

import GameIO
import Nash

def parse_args():
	parser = GameIO.io_parser()
	parser.add_argument("EVs", choices=["point", "sample", "DPR"])
	parser.add_argument("--DPR_size", type=int, default=0)
	parser.add_argument("--one", action="store_true")
	args = parser.parse_args()
	if args.EVs == "DPR":
		assert args.DPR_size > 0, "DPR requires a value for --DPR_size"
	return args

def main():
	args = parse_args()
	g = args.input
	g.EVs = args.EVs
	g.DPR_size = args.DPR_size
	equilibria = [g.toProfile(eq) for eq in Nash.mixed_nash(g, \
					at_least_one=args.one)]
	print GameIO.to_JSON_str(equilibria)

if __name__ == "__main__":
	main()
