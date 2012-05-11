#! /usr/bin/env python2.7

from RoleSymmetricGame import Game, payoff_data

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


def Subgame(game, strategies={}):
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
				sg.addProfile({r:[payoff_data(s, p[r][s], \
						game.getPayoff(p,r,s)) for s in p[r]] for r in p})
	else:
		for p in game:
			if all([all([s in sg.strategies[r] for s in p[r]]) for r in p]):
				sg.addProfile({r:[payoff_data(s, p[r][s], \
						game.getPayoff(p,r,s)) for s in p[r]] for r in p})
	return sg


def SubgameAvailable(game, strategies = {}):
	sg = Game(game.roles, game.players, strategies)
	for p in sg.allProfiles():
		if p not in game:
			return False
	return True


def IsSubgame(small_game, big_game):
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


def Cliques(full_game, known_subgames=set()):
	"""
	Finds maximal subgames for which all profiles are known.

	input:
	subgames = known complete subgames to be expanded (any payoff data in
	the known subgames is ignored, so for faster loading, give only the
	header information).
	"""
	subgames = {Subgame(full_game)}
	for g in known_subgames:
		sg = Subgame(full_game, g.strategies)
		if sg.isComplete():
			subgames.add(sg)
	maximal_subgames = set()
	while(subgames):
		sg = subgames.pop()
		maximal = True
		for role in full_game.roles:
			for s in set(full_game.strategies[role]) - \
					set(sg.strategies[role]):
				strategies = {r:list(sg.strategies[r]) + ([s] if r == role \
						else []) for r in full_game.roles}
				if not SubgameAvailable(full_game, strategies):
					continue
				new_sg = Game(sg.roles, sg.players, strategies)
				maximal=False
				if new_sg in subgames or new_sg in maximal_subgames:
					continue
				if any([IsSubgame(new_sg, g) for g in \
							subgames.union(maximal_subgames)]):
					continue
				subgames.add(new_sg)
		if maximal:
			sg = Subgame(full_game, sg.strategies)
			if len(sg) > 0:
				maximal_subgames.add(sg)
	return sorted(maximal_subgames, key=len)


from GameIO import readGame, toJSON, io_parser
from json import dumps

def parse_args():
	parser = io_parser()
	parser.add_argument("-known", type=str, default="[]", help= \
			"File with known complete subgames. Improves runtime.")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	game = readGame(args.input)
	subgames = readGame(args.known)
	print dumps(toJSON(*Cliques(game, subgames)))

