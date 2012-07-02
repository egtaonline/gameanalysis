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


from GameIO import io_parser, toJSONstr

def parse_args():
	parser = io_parser()
	parser.add_argument("-r", metavar="REGRET", type=float, default=1e-3, \
			help="Max allowed regret for approximate Nash equilibria. " + \
			"default=1e-3")
	parser.add_argument("-d", metavar="DISTANCE", type=float, default=1e-3, \
			help="L2-distance threshold to consider equilibria distinct. " + \
			"default=1e-3")
	parser.add_argument("-c", metavar="CONVERGENCE", type=float, default=1e-8, \
			help="Replicator dynamics convergence thrshold. default=1e-8")
	parser.add_argument("-i", metavar="ITERATIONS", type=int, default=10000, \
			help="Max replicator dynamics iterations. default=1e4")
	parser.add_argument("-s", metavar="SUPPORT", type=float, default=1e-3, \
			help="Min probability for a strategy to be considered in " + \
			"support. default=1e-3")
	parser.add_argument("--pure", action="store_true", help="compute " + \
			"pure-strategy Nash equilibria rather than mixed")
	args = parser.parse_args()
	games = args.input
	if not isinstance(games, list):
		games = [games]
	return games, args


def main():
	games, args = parse_args()
	if args.pure:
		equilibria = [PureNash(g, args.r) for g in games]
	else:
		equilibria = [[g.toProfile(eq, args.s) for eq in MixedNash(g, \
				args.r, args.d, iters=args.i, converge_thresh=args.c)] \
				for g in games]
	if len(equilibria) > 1:
		print toJSONstr(equilibria)
	else:
		print toJSONstr(equilibria[0])


if __name__ == "__main__":
	main()

