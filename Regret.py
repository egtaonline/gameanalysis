#! /usr/bin/env python2.7

import numpy as np

from itertools import product, chain, combinations_with_replacement as CwR

from RoleSymmetricGame import Profile


def regret(game, p, role=None, strategy=None, deviation=None, bound=False):
	if role == None:
		return max([regret(game, p, r, strategy, deviation, bound) for r \
				in game.roles])
	if strategy == None and isinstance(p, Profile):
		return max([regret(game, p, role, s, deviation, bound) for s \
				in p[role]])
	if deviation == None:
		return max([regret(game, p, role, strategy, d, bound) for d \
				in game.strategies[role]])
	if isinstance(p, Profile):
		dp = p.deviate(role, strategy, deviation)
		if dp in game:
			return game.getPayoff(dp, role, deviation) - \
					game.getPayoff(p, role, strategy)
		else:
			return -float("inf") if bound else float("inf")
	elif isinstance(p, np.ndarray):
		if any(map(lambda prof: prof not in game, mixtureNeighbors(game, \
				p, role, deviation))):
			return -float("inf") if bound else float("inf")
		return game.expectedValues(p)[game.index(role), game.index( \
				role, deviation)] - game.getExpectedPayoff(p, role)
	raise TypeError("unrecognized argument type: " + type(p).__name__)


def neighbors(game, p, *args, **kwargs):
	if isinstance(p, Profile):
		return profileNeighbors(game, p, *args, **kwargs)
	elif isinstance(p, np.ndarray):
		return mixtureNeighbors(game, p, *args, **kwargs)
	raise TypeError("unrecognized argument type: " + type(p).__name__)


def profileNeighbors(game, profile, role=None, strategy=None, \
		deviation=None):
	if role == None:
		return list(chain(*[profileNeighbors(game, profile, r, strategy, \
				deviation) for r in game.roles]))
	if strategy == None:
		return list(chain(*[profileNeighbors(game, profile, role, s, \
				deviation) for s in profile[role]]))
	if deviation == None:
		return list(chain(*[profileNeighbors(game, profile, role, strategy, \
				d) for d in set(game.strategies[role]) - {strategy}]))
	return [profile.deviate(role, strategy, deviation)]


def mixtureNeighbors(game, mix, role=None, deviation=None):
	n = set()
	for profile in feasibleProfiles(game, mix):
		n.update(profileNeighbors(game, profile, role, deviation=deviation))
	return n


def feasibleProfiles(game, mix, thresh=1e-3):
	return [Profile({r:{s:p[game.index(r)].count(s) for s in set(p[ \
			game.index(r)])} for r in game.roles}) for p in product(*[ \
			CwR(filter(lambda s: mix[game.index(r), game.index(r,s)] >= \
			thresh, game.strategies[r]), game.players[r]) for r \
			in game.roles])]


def SymmetricProfileRegrets(game):
	assert len(game.roles) == 1, "game must be symmetric"
	role = game.roles[0]
	return {s: regret(game, Profile({role:{s:game.players[role]}})) for s \
			in game.strategies[role]}


def EquilibriumRegrets(game, eq):
	regrets = {}
	for role in game.roles:
		regrets[role] = {}
		for strategy in game.strategies[role]:
			regrets[role][strategy] = -regret(game, eq, deviation=strategy)
	return regrets


from GameIO import readProfile, toJSONstr, io_parser
from Subgames import translate

def parse_args():
	parser = io_parser()
	parser.add_argument("games", type=str, help="File with games from which " +\
			"equilibria were computed.")
	parser.add_argument("-base", type=str, default="", help= \
			"Base game file against which to compute regrets. If unspecified" +\
			" in-game regrets are computed instead.")
	return parser.parse_args()


def main():
	args = parse_args()
	games = read(args.games)
	if not isinstance(games, list):
		games = [games]
		equilibria = [args.input]
	else:
		equilibria = args.input
	if args.base != "":
		base_game = readGame(args.base)
	else:
		base_game = None
	regrets = []
	for g, eq_list in zip(games, equilibria):
		regrets.append([])
		for eq in eq_list:
			if base_game != None:
				regrets[-1].append(regret(base_game, translate(eq, g, \
						base_game)))
			else:
				regrets[-1].append(regret(g, eq))
	print toJSONstr(regrets)


if __name__ == "__main__":
	main()

