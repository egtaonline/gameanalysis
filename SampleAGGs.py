#! /usr/bin/env python2.7

import GameIO
import ActionGraphGame
import RoleSymmetricGame as RSG
from dpr import HR_profiles, DPR_profiles
from numpy import array
from numpy.random import multinomial


def sample_at_reduction(AGG, samples, reduction_profiles, players):
	"""
	AGG:		ActionGraphGame.Noisy_AGG object.
	samples:	number of samples per reduced-game profile.
	reduction_profiles:
				function that takes a number of players and generates a set of
				profiles; intended settings: DPR_profiles, HR_profiles
	players:	number of players in the reduced game; passed as an argument to
				reduction_profiles.

	RETURNS:	RoleSymmetricGame.SampleGame object with observations drawn
				from AGG; samples draws are taken at each profile generated
				by reduction_profiles.
	"""
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})
	for prof in reduction_profiles(g, {"All":players}):
		values = AGG.sample(prof["All"], samples)
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})
	return g


def sample_near_reduction(AGG, samples, reduction_profiles, players):
	"""
	AGG:		ActionGraphGame.Noisy_AGG object.
	samples:	number of samples per reduced-game profile; should be a
				multiple of 5.
	reduction_profiles:
				function that takes a number of players and generates a set of
				profiles; intended settings: DPR_profiles, HR_profiles
	players:	number of players in the reduced game; passed as an argument to
				reduction_profiles.

	RETURNS:	RoleSymmetricGame.SampleGame object with observations drawn
				from AGG; samples/5 draws are taken near each profile generated
				by reduction_profiles by treating the fraction of players in a
				profile playing each strategy as a distribution and drawing N
				samples from it; each resulting N-player profile gets sampled
				5 times from the noisy AGG.
	"""
	assert not (samples % 5)
	random_profiles = {}
	g = RSG.SampleGame(["All"], {"All":AGG.players}, {"All":AGG.strategies})
	for prof in reduction_profiles(g, {"All":players}):
		dist = array([prof["All"].get(s,0) for s in AGG.strategies],
														dtype=float)
		dist /= float(AGG.players)
		for _ in range(samples / 5):
			rp = multinomial(AGG.players, dist)
			rp = filter(lambda p:p[1], zip(AGG.strategies,rp))
			rp = RSG.Profile({"All":dict(rp)})
			random_profiles[rp] = random_profiles.get(rp,0) + 1

	for prof,count in random_profiles.iteritems():
		values = AGG.sample(prof["All"], 5*count)
		g.addProfile({"All":[RSG.PayoffData(s,c,values[s]) for \
								s,c in prof["All"].iteritems()]})

	return g


def parse_args():
	parser = GameIO.io_parser()
	parser.add_argument("players", type=int, help="Size of reduced game to "+
				"sample near.")
	parser.add_argument("samples", type=int, default=[], help="Number of "+
				"samples per profile. Should be a multiple of 5.")
	parser.add_argument("--HR", action="store_true", help="Set to sample near "+
				"hierarchical reduction profiles instead of deviation-"+
				"preserving reduction profiles.")
	parser.add_argument("--at", action="store_true", help="Set to gather "+
				"samples at reduced game profiles instead of near them.")
	args = parser.parse_args()
	return args


def main():
	a = parse_args()
	g = ActionGraphGame.LEG_to_AGG(a.input)
	if a.HR:
		reduction_profiles = HR_profiles
	else:
		reduction_profiles = DPR_profiles
	if a.at:
		sg = sample_at_reduction(g, a.samples, reduction_profiles, a.players)
	else:
		sg = sample_near_reduction(g, a.samples, reduction_profiles, a.players)
	print GameIO.to_JSON_str(sg)


if __name__ == "__main__":
	main()

