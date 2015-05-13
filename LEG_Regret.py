#! /usr/bin/env python2.7

import json
from argparse import ArgumentParser
from os.path import join, exists

import Regret
import GameIO
import RoleSymmetricGame as RSG
import ActionGraphGame as AGG

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("game_type", type=str)
	parser.add_argument("LEG_folder", type=str)
	parser.add_argument("EQ_folder", type=str)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--stop", type=int, default=100)
	return parser.parse_args()

def main():
	args = parse_args()
	for index in range(args.start, args.stop):
		agg_file = join(args.LEG_folder, str(index) + ".json")
		eq_file = join(args.EQ_folder, str(index)+"_"+args.game_type+".json")
		out_file = join(args.EQ_folder, str(index) +"_"+ args.game_type +
													"_regret.json")
		if exists(out_file) or (not exists(eq_file)):
			continue
		with open(agg_file) as f:
			game = AGG.LEG_to_AGG(json.load(f))
		empty_game = RSG.Game(["All"], {"All":game.players},
								{"All":game.strategies})
		equilibria = map(empty_game.toArray, GameIO.read(eq_file))
		regrets = [game.regret(eq[0]) for eq in equilibria]
		with open(out_file, "w") as f:
			json.dump(regrets, f)

if __name__ == "__main__":
	main()
