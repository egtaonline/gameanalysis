#! /usr/bin/env python2.7

from itertools import product
from math import isinf

from RoleSymmetricGame import Profile
from Regret import regret
from Subgames import subgame

def best_responses(game, prof, role=None, strategy=None):
	"""
	If role is unspecified, bestResponses returns a dict mapping each role
	all of its strategy-level results. If strategy is unspecified,
	bestResponses returns a dict mapping strategies to the set of best
	responses to the opponent-profile without that strategy.
	"""
	if role == None:
		return {r: best_responses(game, prof, r, strategy) for r \
				in game.roles}
	if strategy == None and isinstance(prof, Profile):
		return {s: best_responses(game, prof, role, s) for s in \
				prof[role]}
	best_deviations = set()
	biggest_gain = float('-inf')
	unknown = set()
	for dev in game.strategies[role]:
		reg = regret(game, prof, role, strategy, dev)
		if isinf(reg):
			unknown.add(dev)
		elif reg > biggest_gain:
			best_deviations = {dev}
			biggest_gain = reg
		elif reg == biggest_gain:
			best_deviations.add(dev)
	return best_deviations, unknown


def iterated_elimination(game, criterion, *args, **kwargs):
	"""
	iterated elimination of dominated strategies

	input:
	criterion = function to find dominated strategies
	"""
	reduced_game = eliminate_strategies(game, criterion, *args, **kwargs)
	while len(reduced_game) < len(game):
		game = reduced_game
		reduced_game = eliminate_strategies(game, criterion, *args, **kwargs)
	return game


def eliminate_strategies(game, criterion, *args, **kwargs):
	eliminated = criterion(game, *args, **kwargs)
	return subgame(game, {r : set(game.strategies[r]) - eliminated[r] \
			for r in game.roles})


def never_best_response(game, conditional=True):
	"""
	never-a-weak-best-response criterion for IEDS

	This criterion is very strong: it can eliminate strict Nash equilibria.
	"""
	non_best_responses = {r:set(game.strategies[r]) for r in game.roles}
	for prof in game:
		for r in game.roles:
			for s in prof[r]:
				br, unknown = best_responses(game, prof, r, s)
				non_best_responses[r] -= set(br)
				if conditional:
					non_best_responses[r] -= unknown
	return non_best_responses


def pure_strategy_dominance(game, conditional=True, weak=False):
	"""
	pure-strategy dominance criterion for IEDS

	conditional==0==False --> unconditional dominance
	conditional==1==True ---> conditional dominance
	conditional==2 ---------> extra-conservative conditional dominance
	"""
	dominated_strategies = {r:set() for r in game.roles}
	for role in game.roles:
		for dominant, dominated in product(game.strategies[role], repeat=2):
			if dominant == dominated or \
					dominated in dominated_strategies[role] or \
					dominant in dominated_strategies[role]:
				continue
			if dominates(game, role, dominant, dominated, conditional, weak):
				dominated_strategies[role].add(dominated)
	return dominated_strategies


def dominates(game, role, dominant, dominated, conditional=True, weak=False):
	dominance_observed = False
	for prof in game:
		if dominated in prof[role]:
			reg = regret(game, prof, role, dominated, dominant)
			if reg > 0 and not isinf(reg):
				dominance_observed = True
			elif (reg < 0) or (reg == 0 and not weak) or \
					(isinf(reg) and conditional):
				return False
		elif conditional > 1 and dominant in prof[role] and \
				(prof.deviate(role, dominant, dominated) not in game):
				return False
	return dominance_observed



def mixed_strategy_dominance(game, conditional=True, weak=False):
	"""
	mixed-strategy dominance criterion for IEDS
	"""
	raise NotImplementedError("TODO")


from GameIO import to_JSON_str, io_parser

def parse_args():
	parser = io_parser()
	parser.add_argument("-o", metavar="OUTPUT FORMAT", choices=["game",\
			"strategies"], default="game", help="'game' outputs a JSON "+\
			"representation of the game after IEDS; 'strategies' outputs a "+\
			"mapping of roles to eliminated strategies. Default=strategies")
	parser.add_argument("-c", metavar="CRITERION", choices=["PSD","MSD","NBR"],\
			default="PSD", help="Dominance criterion: PSD = pure-strategy "+\
			"dominance; MSD = mixed-strategy dominance; NBR = never-best-"+\
			"response. Default=PSD")
	parser.add_argument("-m", metavar="MISSING", choices=[0,1,2], default=1, \
			help= "Method to handle missing data: 0 = unconditional "+\
			"dominance, 1 = conditional dominance, 2 = conservative. Default=1")
	parser.add_argument("--weak", action="store_true", help="If the 'weak' "+\
			"flag is set, strategies are eliminated even if they are only "+\
			"weakly dominated.")
	return parser.parse_args()


def main():
	args = parse_args()
	criteria = {"PSD":pure_strategy_dominance, "MSD":mixed_strategy_dominance,\
				"NBR":never_best_response}
	g = iterated_elimination(args.input, criteria[args.c], conditional=args.m)
	if args.o == "strategies":
		eliminated = {r:sorted(set(args.input.strategies[r]) - \
						set(g.strategies[r])) for r in g.roles}
		print to_JSON_str(eliminated)
	else:
		print to_JSON_str(g)


if __name__ == "__main__":
	main()

