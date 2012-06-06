#! /usr/bin/env python2.7

from itertools import product
from math import isinf

from RoleSymmetricGame import Profile
from Regret import regret
from Subgames import Subgame

def bestResponses(game, p, role=None, strategy=None):
	"""
	If role is unspecified, bestResponses returns a dict mapping each role
	all of its strategy-level results. If strategy is unspecified,
	bestResponses returns a dict mapping strategies to the set of best
	responses to the opponent-profile without that strategy.

	If conditional=True, bestResponses returns two sets: the known best
	responses, and the deviations whose value is unkown; otherwise it
	returns only the known best response set.
	"""
	if role == None:
		return {r: bestResponses(game, p, r, strategy) for r \
				in game.roles}
	if strategy == None and isinstance(p, Profile):
		return {s: bestResponses(game, p, role, s) for s in \
				p[role]}
	best_deviations = set()
	biggest_gain = float('-inf')
	unknown = set()
	for dev in game.strategies[role]:
		r = regret(game, p, role, strategy, dev)
		if isinf(r):
			unknown.add(dev)
		elif r > biggest_gain:
			best_deviations = {dev}
			biggest_gain = r
		elif r == biggest_gain:
			best_deviations.add(dev)
	return list(best_deviations), list(unknown)


def IteratedElimination(game, criterion, *args, **kwargs):
	"""
	iterated elimination of dominated strategies

	input:
	criterion = function to find dominated strategies
	"""
	reduced_game = criterion(game, *args, **kwargs)
	while game != reduced_game:
		game = reduced_game
		reduced_game = criterion(game, *args, **kwargs)
	return game


def NeverBestResponse(game, conditional=True):
	"""
	never-a-weak-best-response criterion for IEDS

	This criterion is very strong: it can eliminate strict Nash equilibria.
	"""
	best_responses = {r:set() for r in game.roles}
	for profile in game:
		for r in game.roles:
			for s in profile[r]:
				br, unknown = bestResponses(game, profile, r, s)
				best_responses[r].update(br)
				if conditional:
					best_responses[r].update(unknown)
	return Subgame(game, best_responses)


def PureStrategyDominance(game, conditional=True, weak=False):
	"""
	pure-strategy dominance criterion for IEDS

	conditional==0==False --> unconditional dominance
	conditional==1==True ---> conditional dominance
	conditional==2 ---------> extra-conservative conditional dominance
	"""
	undominated = {r:set(game.strategies[r]) for r in game.roles}
	for r in game.roles:
		for dominant, dominated in product(game.strategies[r], repeat=2):
			if dominant == dominated or dominated not in undominated[r]:
				continue
			dominance_proved = False
			for profile in game:
				if dominated in profile[r]:
					reg = regret(game, profile, r, dominated, dominant)
					if reg > 0 and not isinf(reg):
						dominance_proved = True
					elif (reg < 0) or (reg == 0 and not weak) or \
							(isinf(reg) and conditional):
						dominance_proved = False
						break
				elif dominant in profile[r] and conditional > 1:
					if profile.deviate(r, dominant, dominated) not in game:
						dominance_proved = False
						break
			if dominance_proved:
				undominated[r].remove(dominated)
	return Subgame(game, undominated)


def MixedStrategyDominance(game, conditional=True, weak=False):
	"""
	mixed-strategy dominance criterion for IEDS
	"""
	raise NotImplementedError("TODO")


from GameIO import readGame, toJSON, io_parser

def parse_args():
	parser = io_parser()
	parser.add_argument("-type", choices=["PSD", "MSD", "NBR"], default = \
			"PSD", help="Dominance criterion: PSD = pure-strategy dominance;"+\
			" MSD = mixed-strategy dominance; NBR = never-best-response.")
	parser.add_argument("--conditional", action="store_true")
	parser.add_argument("--weak", action="store_true")
	return parser.parse_args()


def main():
	args = parse_args()
	game = readGame(args.input)
	if args.type == "PSD":
		rgame = IteratedElimination(game, PureStrategyDominance, \
				conditional=args.conditional)
	elif args.type == "MSD":
		rgame = IteratedElimination(game, MixedStrategyDominance, \
				conditional=args.conditional, weak=args.weak)
	elif args.type == "NBR":
		rgame = IteratedElimination(game, NeverBestResponse, \
				conditional=args.conditional, weak=args.weak)
	print toJSON(rgame)


if __name__ == "__main__":
	main()

