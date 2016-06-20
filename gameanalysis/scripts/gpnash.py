"""Module for constructing game models by Gaussian process regression."""
import json
import pickle
import argparse

from gameanalysis import gpgame
from gameanalysis import nash

def main(args):
    game = pickle.load(args.input)
    equilibria = list(nash.mixed_nash(game, replicator=[]))
    json.dump(equilibria, args.output, default=lambda x: x.to_json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType("rb"))
    parser.add_argument("output", type=argparse.FileType("w"))
    main(parser.parse_args())
