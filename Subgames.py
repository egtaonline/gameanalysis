#! /usr/bin/env python2.7

from RoleSymmetricGame import Game, PayoffData

def translate(arr, source_game, target_game):
	"""
	Translates a mixture, profile, count, or payoff array between related
	games based on role/strategy indices.

	Useful for testing full-game regret of subgame equilibria.
	"""
	a = target_game.zeros()
	for role in target_game.roles:
		for strategy in source_game.strategies[role]:
			a[target_game.index(role), target_game.index(role, strategy)] = \
					arr[source_game.index(role), source_game.index(role, \
					strategy)]
	return a


def subgame(game, strategies={}):
	"""
	Creates a game with a subset each role's strategies.

	default settings result in a subgame with no strategies
	"""
	if not strategies:
		strategies = {r:[] for r in game.roles}
	sg = Game(game.roles, game.players, strategies)
	if sg.size <= len(game):
		for p in sg.allProfiles():
			if p in game:
				sg.addProfile({r:[PayoffData(s, p[r][s], \
						game.getPayoff(p,r,s)) for s in p[r]] for r in p})
	else:
		for p in game:
			if all([all([s in sg.strategies[r] for s in p[r]]) for r in p]):
				sg.addProfile({r:[PayoffData(s, p[r][s], \
						game.getPayoff(p,r,s)) for s in p[r]] for r in p})
	return sg


def subgame_available(game, strategies = {}):
	sg = Game(game.roles, game.players, strategies)
	for p in sg.allProfiles():
		if p not in game:
			return False
	return True


def is_subgame(small_game, big_game):
	if any((r not in big_game.roles for r in small_game.roles)):
		return False
	if any((small_game.players[r] != big_game.players[r] for r \
			in small_game.roles)):
		return False
	for r in small_game.roles:
		if any((s not in big_game.strategies[r] for s in \
				small_game.strategies[r])):
			return False
	return True


def cliques(full_game, known_subgames=set(), fast=False):
	"""
	Finds maximal subgames for which all profiles are known.

	input:
	subgames = known complete subgames to be expanded (any payoff data in
	the known subgames is ignored, so for faster loading, give only the
	header information).
	"""
	subgames = {subgame(full_game)}
	for g in known_subgames:
		sg = subgame(full_game, g.strategies)
		if sg.isComplete():
			subgames.add(sg)
	maximal_subgames = set()
	explored_subgames = set()
	while(subgames):
		sg = subgames.pop()
		explored_subgames.add(sg)
		maximal = True
		for role in full_game.roles:
			for s in set(full_game.strategies[role]) - \
					set(sg.strategies[role]):
				strategies = {r:list(sg.strategies[r]) + ([s] if r == role \
						else []) for r in full_game.roles}
				if not subgame_available(full_game, strategies):
					continue
				new_sg = Game(sg.roles, sg.players, strategies)
				maximal=False
				if new_sg in subgames or new_sg in explored_subgames:
					continue
				if fast and any([is_subgame(new_sg, g) for g in \
						subgames.union(maximal_subgames)]):
					continue
				subgames.add(new_sg)
		if maximal:
			sg = subgame(full_game, sg.strategies)
			if len(sg) > 0:
				maximal_subgames.add(sg)
	return sorted(maximal_subgames, key=len)


from GameIO import read, to_JSON_str, io_parser

def parse_args():
	parser = io_parser()
	parser.add_argument("-known", type=str, default="[]", help= \
			"File with known complete subgames. Improves runtime.")
	parser.add_argument("--fast", action="store_true", help="Speeds up " + \
			"subgame finding, especially if using known subgames. Can miss " + \
			"some complete subgames.")
	return parser.parse_args()


def main():
	args = parse_args()
	subgames = read(args.known)
	print to_JSON_str(cliques(args.input, subgames, args.fast))


if __name__ == "__main__":
	main()
