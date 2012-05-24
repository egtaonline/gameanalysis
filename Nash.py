#! /usr/bin/env python2.7

from numpy.linalg import norm

from RoleSymmetricGame import Profile, tiny
from Regret import regret

def PureNash(game, epsilon=0):
	"""
	Finds all pure-strategy epsilon-Nash equilibria.
	"""
	return filter(lambda profile: regret(game, profile) <= epsilon, game)


def MinRegretProfile(game):
	"""
	Finds the profile with the confirmed lowest regret.
	"""
	return min([(regret(game, profile), profile) for profile in game])[1]


def MixedNash(game, regret_thresh=1e-4, dist_thresh=1e-2, *RD_args, **RD_kwds):
	"""
	Runs replicator dynamics from multiple starting mixtures.
	"""
	equilibria = []
	for m in game.biasedMixtures() + [game.uniformMixture()]:
		eq = ReplicatorDynamics(game, m, *RD_args, **RD_kwds)
		distances = map(lambda e: norm(e-eq,2), equilibria)
		if regret(game, eq) <= regret_thresh and all([d >= dist_thresh \
				for d in distances]):
			equilibria.append(eq)
	return equilibria


def ReplicatorDynamics(game, mix, iters=10000, converge_thresh=1e-8, \
		verbose=False):
	"""
	Replicator dynamics.
	"""
	for i in range(iters):
		old_mix = mix
		mix = (game.expectedValues(mix) - game.minPayoffs + tiny) * mix
		mix = mix / mix.sum(1).reshape(mix.shape[0],1)
		if norm(mix - old_mix) <= converge_thresh:
			break
	if verbose:
		print i+1, "iterations ; mix =", mix, "; regret =", regret(game, mix)
	return mix


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


from GameIO import readGame, io_parser
from Regret import regret, neighbors
from Subgames import Subgame, translate
from Dominance import bestResponses

from copy import deepcopy
from math import isinf

def parse_args():
	parser = io_parser()
	parser.add_argument("-base", type=str, default="", help="Base game to " + \
			"be used for regret calculations. If unspecified, in-game " + \
			"regrets are calculated.")
	parser.add_argument("-r", metavar="REGRET", type=float, default=1e-3, \
			help="Max allowed regret for approximate Nash equilibria.")
	parser.add_argument("-d", metavar="DISTANCE", type=float, default=1e-3, \
			help="L2-distance threshold to consider equilibria distinct.")
	parser.add_argument("-c", metavar="CONVERGENCE", type=float, default=1e-8, \
			help="Replicator dynamics convergence thrshold.")
	parser.add_argument("-i", metavar="ITERATIONS", type=int, default=10000, \
			help="Max replicator dynamics iterations.")
	parser.add_argument("-s", metavar="SUPPORT", type=float, default=1e-3, \
			help="Min probability for a strategy to be considered in support.")
	parser.add_argument("--pure", action="store_true", help="compute " + \
			"pure-strategy Nash equilibria")
	parser.add_argument("--mixed", action="store_true", help="compute " + \
			"mixed-strategy Nash equilibria")
	args = parser.parse_args()
	games = readGame(args.input)
	if not isinstance(games, list):
		games = [games]
	try:
		base_game = readGame(args.b)
	except:
		base_game = None
	return games, base_game, args


def write_pure_eq(game, base_game, args):
	if base_game == None:
		base_game = game
	pure_equilibria = PureNash(game, args.r)
	l = len(pure_equilibria)
	if l > 0:
		print "\n" + str(len(pure_equilibria)), "pure strategy Nash equilibri" \
				+ ("um:" if l == 1 else "a:")
		for i, eq in enumerate(pure_equilibria):
			print str(i+1) + ". regret =", round(regret(base_game, eq), 4)
			for role in base_game.roles:
				print "    " + role + ":", ", ".join(map(lambda pair: \
						str(pair[1]) + "x " + str(pair[0]), eq[role].items()))
	else:
		print "\nno pure strategy Nash equilibria found."
		mrp = MinRegretProfile(game)
		print "regret =", regret(base_game, mrp)
		print "minimum regret pure strategy profile (regret = " + \
				str(round(regret(base_game, mrp), 4)) + "):"
		for role in base_game.roles:
			print "    " + role + ":", ", ".join(map(lambda pair: \
					str(pair[1]) + "x " + str(pair[0]), mrp[role].items()))



def write_mixed_eq(game, base_game, args):
	if base_game == None:
		base_game = game
	print "game "+str(i+1)+":\n", "\n".join(map(lambda x: x[0] + \
			":\n\t\t" + "\n\t\t".join(x[1]), sorted( \
			game.strategies.items()))).expandtabs(4)
	mixed_equilibria = MixedNash(game, args.r, args.d, iters=args.i, \
		converge_thresh=args.c)
	print "\n" + str(len(mixed_equilibria)), "approximate mixed strategy"+ \
			" Nash equilibri" + ("um:" if len(mixed_equilibria) == 1 \
			else "a:")
	for j, eq in enumerate(mixed_equilibria):
		full_eq = translate(eq, game, base_game)
		if all(map(lambda p: p in base_game, neighbors(base_game, \
				full_eq))):
			print str(j+1) + ". regret =", round(regret(base_game, \
					full_eq), 4)
		else:
			print str(j+1) + ". regret >=", round(regret(base_game,  \
					full_eq, bound=True), 4)

		support = {r:[] for r in base_game.roles}
		for k,role in enumerate(base_game.roles):
			print role + ":"
			for l,strategy in enumerate(base_game.strategies[role]):
				if full_eq[k][l] >= args.s:
					support[role].append(strategy)
					print "    " + strategy + ": " + str(round(100 * \
							full_eq[k][l], 2)) + "%"

		BR = bestResponses(base_game, full_eq)
		print "best responses:"
		for role in base_game.roles:
			deviation_support = deepcopy(support)
			deviation_support[role].extend(BR[role][0])
			if len(BR[role][0]) == 0:
				continue
			r = regret(base_game, full_eq, role, deviation=BR[role][0][0])
			print "\t" + str(role) + ": " + ", ".join(BR[role][0]) + \
					";\tgain =", (round(r, 4) if not isinf(r) else "?")
			if base_game != game:
				print "Deviation game " + ("explored." if Subgame( \
						base_game, deviation_support).isComplete() else \
						"UNEXPLORED!") + "\n"


def main():
	games, base_game, args = parse_args()
	for i, game in enumerate(games):
		if len(games) > 1:
			print "game", i+1, "=", game, "\n"
		else:
			print game, "\n"
		if args.pure:
			write_pure_eq(game, base_game, args)
		if args.mixed:
			write_mixed_eq(game, base_game, args)


if __name__ == "__main__":
	main()


