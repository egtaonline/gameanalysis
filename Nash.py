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


from GameIO import readGame, writeGames
from Regret import regret
from argparse import ArgumentParser
from sys import stdin, stdout

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("-g", metavar="GAMES", type=str, default="", help= \
			"File with game or games to be analyzed. Suported file types: " + \
			"EGAT symmetric XML, EGAT strategic XML, testbed role-symmetric" + \
			"JSON. Defaults to stdin.")
	parser.add_argument("-o", metavar="OUTPUT", type=str, default="", \
			help="File for writing output. Defaults to stdout.")
	parser.add_argument("-b", metavar="BASEGAME", type=str, default="", \
			help="Base game to be used for regret calculations. " + \
			"If unspecified, in-game regrets are calculated.")
	parser.add_argument("-r", metavar="REGRET", type=float, default=1e-3, \
			help="Max allowed regret for approximate Nash equilibria.")
	parser.add_argument("-d", metavar="DIST", type=float, default=1e-3, \
			help="L2-distance threshold to consider equilibria distinct.")
	parser.add_argument("-c", metavar="CONVERGE", type=float, default=1e-8, \
			help="Replicator dynamics convergence thrshold.")
	parser.add_argument("-i", metavar="ITERS", type=int, default=10000, \
			help="Max replicator dynamics iterations.")
	parser.add_argument("-s", metavar="SUPPORT", type=float, default=1e-3, \
			help="Min probability for a strategy to be considered in support.")
	parser.add_argument("--pure", action="store_true", help="compute " + \
			"pure-strategy Nash equilibria")
	parser.add_argument("--mixed", action="store_true", help="compute " + \
			"mixed-strategy Nash equilibria")
	args = parser.parse_args()
	if args.g == "":
		args.g = stdin
	games = readGame(args.g)
	if not isinstance(games, list):
		games = [games]
	try:
		out_file = open(args.o, "w")
	except IOError:
		out_file = stdout
	try:
		base_game = readGame(args.b)
	except:
		base_game = None
	return games, out_file, base_game, args


def write_pure_eq(game, out_file, base_game, regr_thresh):
	if base_game == None:
		base_game = game
	pure_equilibria = PureNash(game, regr_thresh)
	l = len(pure_equilibria)
	if l > 0:
		out_file.write(str(len(pure_equilibria)) + " pure strategy Nash " + \
				"equilibri" + ("um:" if l == 1 else "a:") + "\n")
		for i, eq in enumerate(pure_equilibria):
			out_file.write(str(i+1) + ". regret = " + str(round(regret( \
					base_game, eq), 4)) + "\n")
			for role in base_game.roles:
				out_file.write("    "+ role +": "+ ", ".join(map(lambda pair: \
						str(pair[1]) + "x " + str(pair[0]), eq[role].items())) \
						+ "\n")
		out_file.write("\n")
	else:
		out_file.write("no pure strategy Nash equilibria found.\n")
		mrp = MinRegretProfile(game)
		out_file.write("regret =" + str(regret(base_game, mrp)) + "\n")
		out_file.write("minimum regret pure strategy profile (regret = " + \
				str(round(regret(base_game, mrp), 4)) + "):\n")
		for role in base_game.roles:
			out_file.write("    " + role + ":" + ", ".join(map(lambda pair: \
					str(pair[1]) + "x " + str(pair[0]), mrp[role].items())))
		out_file.write("\n")



def write_mixed_eq(game, out_file, base_game, regr_thresh, dist_thresh, \
		iters, conv_thresh, supp_thresh):
	if base_game == None:
		base_game = game
	raise NotImplementedError("TODO")


if __name__ == "__main__":
	games, out_file, base_game, args = parse_args()
	for i, game in enumerate(games):
		print "game", i+1, "=", game
		if args.pure:
			write_pure_eq(game, out_file, base_game, args.r)
		if args.mixed:
			write_mixed_eq(game, out_file, base_game, args.r, args.d, args.i, \
					args.c, args.s)




