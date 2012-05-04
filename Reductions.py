from RoleSymmetricGame import Game, Profile

def HierarchicalReduction(game, players={} ):
	if not players:
		players = {r : game.players[r] / 2 for r in game.roles}
	HR_game = Game(game.roles, players, game.strategies)
	for reduced_profile in HR_game.allProfiles():
		try:
			full_profile = Profile({r:FullGameProfile(reduced_profile[r], \
					game.players[r]) for r in game.roles})
			HR_game.addProfile({r:[payoff_data(s, reduced_profile[r][s], \
					game.getPayoff(full_profile, r, s)) for s in \
					full_profile[r]] for r in full_profile})
		except KeyError:
			continue
	return HR_game


def FullGameProfile(HR_profile, N):
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


def DeviationPreservingReduction(game, players={}):
	if not players:
		players = {r:2 for r in game.roles}
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
							full_profile[r] = FullGameProfile(opp_prof, \
									game.players[r] - 1)
							full_profile[r][s] += 1
						else:
							full_profile[r] = FullGameProfile(reduced_profile[\
									r], game.players[r])
					role_payoffs[r].append(payoff_data(s, reduced_profile[r\
							][s], game.getPayoff(Profile(full_profile), r, s)))
			DPR_game.addProfile(role_payoffs)
		except KeyError:
			continue
	return DPR_game


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
						full_profile[r] = FullGameProfile(opp_prof, \
								game.players[r] - 1)
						full_profile[r][s] += 1
					else:
						full_profile[r] = FullGameProfile(reduced_profile[\
								r], game.players[r])
				profiles.append(Profile(full_profile))
	return profiles

