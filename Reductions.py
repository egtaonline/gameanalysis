#! /usr/bin/env python2.7

from RoleSymmetricGame import Game, Profile, PayoffData

def hierarchical_reduction(game, players={} ):
	if not players:
		players = {r : game.players[r] / 2 for r in game.roles}
	HR_game = Game(game.roles, players, game.strategies)
	for reduced_profile in HR_game.allProfiles():
		try:
			full_profile = Profile({r:full_game_profile(reduced_profile[r], \
					game.players[r]) for r in game.roles})
			HR_game.addProfile({r:[PayoffData(s, reduced_profile[r][s], \
					game.getPayoff(full_profile, r, s)) for s in \
					full_profile[r]] for r in full_profile})
		except KeyError:
			continue
	return HR_game


def full_game_profile(HR_profile, N):
	"""
	Returns the symmetric full game profile corresponding to the given
	symmetric reduced game profile.
	"""
	n = sum(HR_profile.values())
	full_profile = {s : c * N / n  for s,c in HR_profile.items()}
	while sum(full_profile.values()) < N:
		full_profile[max([(float(N) / n * HR_profile[s] \
				- full_profile[s], s) for s in full_profile])[1]] += 1
	return full_profile


def deviation_preserving_reduction(game, players={}):
	if not players:
		return twins_reduction(game)
	DPR_game = Game(game.roles, players, game.strategies)
	for reduced_profile in DPR_game.allProfiles():
		try:
			role_payoffs = {}
			for role in game.roles:
				role_payoffs[role] = []
				for s in reduced_profile[role]:
					full_profile = {}
					for r in game.roles:
						if r == role:
							opp_prof = reduced_profile.asDict()[r]
							opp_prof[s] -= 1
							full_profile[r] = full_game_profile(opp_prof, \
									game.players[r] - 1)
							full_profile[r][s] += 1
						else:
							full_profile[r] = full_game_profile( \
									reduced_profile[r], game.players[r])
					role_payoffs[r].append(PayoffData(s, reduced_profile[r\
							][s], game.getPayoff(Profile(full_profile), r, s)))
			DPR_game.addProfile(role_payoffs)
		except KeyError:
			continue
	return DPR_game


def twins_reduction(game):
	return deviation_preserving_reduction(game, {r:2 for r in game.roles})


def DPR_profiles(game, players={}):
	if not players:
		players = {r:2 for r in game.roles}
	DPR_game = Game(game.roles, players, game.strategies)
	profiles = []
	for reduced_profile in DPR_game.allProfiles():
		for role in game.roles:
			for s in reduced_profile[role]:
				full_profile = {}
				for r in game.roles:
					if r == role:
						opp_prof = reduced_profile.asDict()[r]
						opp_prof[s] -= 1
						full_profile[r] = full_game_profile(opp_prof, \
								game.players[r] - 1)
						full_profile[r][s] += 1
					else:
						full_profile[r] = full_game_profile(reduced_profile[\
								r], game.players[r])
				profiles.append(Profile(full_profile))
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
