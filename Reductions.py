#! /usr/bin/env python2.7

from dpr import DPR, HR, twins_reduction
from GameIO import io_parser

def parse_args():
	parser = io_parser()
	parser.add_argument("type", choices=["DPR", "HR", "TR"], help="Type " + \
			"of reduction to perform.")
	parser.add_argument("players", type=int, default=[], nargs="*", help= \
			"Number of players in each reduced-game role.")
	return parser.parse_args()


def main():
	args = parse_args()
	game = args.input
	players = dict(zip(game.roles, args.players))
	if args.type == "DPR":
		print GameIO.to_JSON_str(DPR(game, players))
	elif args.type == "HR":
		print GameIO.to_JSON_str(HR(game, players))
	elif args.type == "TR":
		print GameIO.to_JSON_str(twins_reduction(game))


if __name__ == "__main__":
	main()
