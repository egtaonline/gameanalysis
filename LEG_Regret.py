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
	parser.add_argument("LEG_folder", type=str)
	parser.add_argument("EQ_folder", type=str)
	parser.add_argument("index", type=int)
	return parser.parse_args()

def main():
	args = parse_args()
	with open(join(args.LEG_folder, str(args.index) + ".json")) as f:
		game = AGG.LEG_to_AGG(json.load(f))
	empty_game = RSG.Game(["All"], {"All":game.players},
							{"All":game.strategies})
	for eq_type in ["_GP-N_EQ-D3","_GP-N_EQ-D5","_GP-N_EQ-p",
					"_GP-p_EQ-D3","_GP-p_EQ-D5","_GP-p_EQ-p",
					"_GP-s_EQ-D3","_GP-s_EQ-D5","_GP-s_EQ-D5",
					"_DPR-EQ"]:
		eq_fn = join(args.EQ_folder, str(args.index)+eq_type+".json")
		if exists(eq_fn):
			equilibria = map(empty_game.toArray, GameIO.read(eq_fn))
			regrets = [game.regret(eq[0]) for eq in equilibria]
			with open(join(args.EQ_folder, str(args.index)+eq_type+
									"_regret.json"), "w") as rf:
				json.dump(regrets, rf)

if __name__ == "__main__":
	main()
