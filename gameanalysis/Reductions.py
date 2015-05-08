#! /usr/bin/env python2.7

from RoleSymmetricGame import Game, Profile, PayoffData, is_symmetric

def hierarchical_reduction(game, players={}):
	if not players:
		players = {r : game.players[r] / 2 for r in game.roles}
	elif len(game.roles) == 1 and isinstance(players, int):
		players = {game.roles[0]:players}
	elif isinstance(players, list):
		players = dict(zip(game.roles, players))
	HR_game = type(game)(game.roles, players, game.strategies)
	for reduced_profile in HR_game.allProfiles():
		try:
			full_profile = Profile({r:full_prof_sym(reduced_profile[r], \
					game.players[r]) for r in game.roles})
			HR_game.addProfile({r:[PayoffData(s, reduced_profile[r][s], \
					game.getPayoffData(full_profile, r, s)) for s in \
					full_profile[r]] for r in full_profile})
		except KeyError:
			continue
	return HR_game


HR = hierarchical_reduction


def full_prof_sym(HR_profile, N):
	"""
	Returns the symmetric full game profile corresponding to the given
	symmetric reduced game profile under hierarchical reduction.

	In the event that N isn't divisible by n, we first assign by rounding
	error and break ties in favor of more-played strategies. The final
	tie-breaker is alphabetical order.
	"""
	if N < 2:
		return HR_profile
	n = sum(HR_profile.values())
	full_profile = {s : (c * N / n) if n > 0 else N for s,c in \
					HR_profile.items()}
	if sum(full_profile.values()) == N:
		return full_profile

	#deal with non-divisible strategy counts
	rounding_error = {s : float(c * N) / n - full_profile[s] for \
						s,c in HR_profile.items()}
	strat_order = sorted(HR_profile.keys())
	strat_order.sort(key=HR_profile.get, reverse=True)
	strat_order.sort(key=rounding_error.get, reverse=True)
	for s in strat_order[:N - sum(full_profile.values())]:
		full_profile[s] += 1
	return full_profile


def full_prof_DPR(DPR_profile, role, strat, players):
	"""
	Returns the full game profile whose payoff determines that of strat
	in the reduced game profile.
	"""
	full_prof = {}
	for r in DPR_profile:
		if r == role:
			opp_prof = DPR_profile.asDict()[r]
			opp_prof[strat] -= 1
			full_prof[r] = full_prof_sym(opp_prof, players[r] - 1)
			full_prof[r][strat] += 1
		else:
			full_prof[r] = full_prof_sym(DPR_profile[r], players[r])
	return Profile(full_prof)


def deviation_preserving_reduction(game, players={}):
	if not players:
		return twins_reduction(game)
	elif len(game.roles) == 1 and isinstance(players, int):
		players = {game.roles[0]:players}
	elif isinstance(players, list):
		players = dict(zip(game.roles, players))

	DPR_game = type(game)(game.roles, players, game.strategies)
	for DPR_prof in DPR_game.allProfiles():
		try:
			role_payoffs = {}
			for r in game.roles:
				role_payoffs[r] = []
				for s in DPR_prof[r]:
					full_prof = full_prof_DPR(DPR_prof, r, s, game.players)
					role_payoffs[r].append(PayoffData(s,DPR_prof[r][s],\
							game.getPayoffData(full_prof, r, s)))
			DPR_game.addProfile(role_payoffs)
		except KeyError:
			continue
	return DPR_game


DPR = deviation_preserving_reduction


def twins_reduction(game):
	players = {r:min(2,p) for r,p in game.players.items()}
	return deviation_preserving_reduction(game, players)


def DPR_profiles(game, players={}):
	"""Returns the profiles from game that contribute to the DPR game."""
	if not players:
		players = {r:2 for r in game.roles}
	elif len(game.roles) == 1 and isinstance(players, int):
		players = {game.roles[0]:players}
	elif isinstance(players, list):
		players = dict(zip(game.roles, players))
	DPR_game = Game(game.roles, players, game.strategies)
	profiles = []
	for DPR_prof in DPR_game.allProfiles():
		for r in game.roles:
			for s in DPR_prof[r]:
				full_prof = full_prof_DPR(DPR_prof, r, s, game.players)
				profiles.append(full_prof)
	return profiles


from GameIO import to_JSON_str, io_parser

def parse_args():
	parser = io_parser()
	parser.add_argument("type", choices=["DPR", "HR", "TR"], help="Type " + \
			"of reduction to perform.")
	parser.add_argument("players", type=int, default=[], nargs="*", help= \
			"Number of players in each reduced-game role.")
	return parser.parse_args()


def main():
	args = parse_args()
	game = args.input
	players = dict(zip(game.roles, args.players))
	if args.type == "DPR":
		print to_JSON_str(deviation_preserving_reduction(game, players))
	elif args.type == "HR":
		print to_JSON_str(hierarchical_reduction(game, players))
	elif args.type == "TR":
		print to_JSON_str(twins_reduction(game))


if __name__ == "__main__":
	main()
