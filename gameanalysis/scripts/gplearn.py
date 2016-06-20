"""Module for constructing game models by Gaussian process regression."""
import json
import pickle
import argparse

from gameanalysis import rsgame
from gameanalysis import gpgame

def main(args):
    game = gpgame.GPGame(rsgame.Game.from_json(json.load(args.input)))
    pickle.dump(game, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType("r"))
    parser.add_argument("output", type=argparse.FileType("wb"))
    main(parser.parse_args())
