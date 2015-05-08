#! /usr/bin/env python2.7

from HashableClasses import h_dict
from RoleSymmetricGame import SampleGame, PayoffData
from GameIO import read

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


def subgame(game, strategies={}, require_all=False):
	"""
	Creates a game with a subset each role's strategies.

	default settings result in a subgame with no strategies
	"""
	if not strategies:
		strategies = {r:[] for r in game.roles}
	sg = type(game)(game.roles, game.players, strategies)
	if sg.size <= len(game):
		for p in sg.allProfiles():
			if p in game:
				add_subgame_profile(game, sg, p)
			elif require_all:
				raise KeyError("Profile missing")
	elif require_all:
		raise KeyError("Profile missing")
	else:
		for p in game:
			if is_valid_profile(sg, p):
				add_subgame_profile(game, sg, p)
	return sg


def add_subgame_profile(game, subgame, prof):
	if isinstance(game, SampleGame):
		subgame.addProfile({role:[PayoffData(strat, prof[role][strat], \
				game.sample_values[game[prof]][game.index(role), game.index( \
				role, strat)]) for strat in prof[role]] for role in prof})
	else:
		subgame.addProfile({role:[PayoffData(strat, prof[role][strat], \
				game.getPayoff(prof, role, strat)) for strat in prof[role]] \
				for role in prof})


def is_valid_profile(game, prof):
	if set(prof.keys()) != set(game.roles):
		return False
	for role in prof:
		for strat in prof[role]:
			if strat not in game.strategies[role]:
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


def cliques(full_game, known_subgames=[]):
	"""
	Finds maximal subgames for which all profiles are known.

	input:
	subgames = known complete subgames to be expanded (any payoff data in
	the known subgames is ignored, so for faster loading, give only the
	header information).
	"""
	new_profiles = set(full_game.knownProfiles()) - set().union(*( \
			subgame(full_game, s).allProfiles() for s in known_subgames))
	new_strategies = {r:set() for r in full_game.roles}
	for prof in new_profiles:
		for role in full_game.roles:
			new_strategies[role] |= set(prof[role].keys())
	subgames = {subgame(full_game).strategies}
	for sg in known_subgames:
		try:
			sg = subgame(full_game, sg.strategies)
		except AttributeError:
			sg =  subgame(full_game, sg)
		if sg.isComplete():
			subgames.add(sg.strategies)
	maximal_subgames = set()
	explored_subgames = set()
	while(subgames):
		sg_strat = subgames.pop()
		empty_role = not all(map(len, sg_strat.values()))
		empty_game = not any(map(len, sg_strat.values()))
		explored_subgames.add(sg_strat)
		maximal = True
		for role in full_game.roles:
			if empty_role and len(sg_strat[role]) > 0:
				continue
			available_strategies = (new_strategies[role] if empty_game \
					else set(full_game.strategies[role])) - set(sg_strat[role])
			for s in available_strategies:
				strategies = h_dict({r : tuple(sorted(list(sg_strat[r]) + \
						([s] if r == role else []))) for r in full_game.roles})
				if strategies in explored_subgames:
					maximal=False
					continue
				try:
					new_sg = subgame(full_game, strategies, True)
				except KeyError:
					continue
				maximal=False
				subgames.add(new_sg.strategies)
		if maximal:
			sg = subgame(full_game, sg_strat)
			if len(sg) > 0:
				maximal_subgames.add(sg.strategies)
	return sorted(maximal_subgames, key=len)


from GameIO import to_JSON_str, io_parser


def main():
	parser = io_parser()
	parser.description = "Detect all complete subgames in a partial game or "+\
						"extract specific subgames."
	parser.add_argument("mode", choices=["detect","extract"], help="If mode "+\
			"is set to detect, all complete subgames will be found, and the "+\
			"output will be a JSON list of role:[strategies] maps "+\
			"enumerating the complete subgames. If mode is set to extract, "+\
			"then the output will be a JSON representation of a game or a "+\
			"list of games with the specified subsets of strategies.")
	parser.add_argument("-k", metavar="known_subgames", type=str, default="", \
			help="In 'detect' mode: file containing known complete subgames "+\
			"from a prior run. If available, this will often speed up the "+\
			"clique-finding algorithm.")
	parser.add_argument("--full", action="store_true", help="In 'detect' "+\
			"mode: setting this flag causes the script to output games "+\
			"instead of role:strategy maps.")
	parser.add_argument("-f", metavar="strategies file", type=str, default="", \
			help="In 'extract' mode: JSON file with role:[strategies] map(s) "+\
			"of subgame(s) to extract. The file should have the same format "+\
			"as the output of detect mode (or to extract just one subgame, "+\
			"a single map instead of a list of them).")
	parser.add_argument("-s", type=int, nargs='*', default=[], help="In "+\
			"'extract' mode: a list of strategy indices to extract. A "+\
			"strategy is specified by its zero-indexed position in a list "+\
			"of all strategies sorted alphabetically by role and sub-sorted "+\
			"alphabetically by strategy name. For example if role r1 has "+\
			"strategies s1,s2,s2 and role r2 has strategies s1,s2, then the "+\
			"subgame with all but the last strategy for each role is "+\
			"extracted by './Subgames.py extract -s 0 1 3'. Ignored if -f "+\
			"is also specified.")
	args = parser.parse_args()
	game = args.input

	if args.mode == "detect":
		if args.k != "":
			known = read(args.k)
		else:
			known = []
		subgames = cliques(game, known)
		if args.full:
			subgames = [subgame(game,s) for s in subgames]
	else:
		if args.f != "":
			strategies = read(args.f)
		elif len(args.s) > 0:
			strategies = {r:[] for r in game.roles}
			l = 0
			i = 0
			for r in game.roles:
				while i < len(args.s) and args.s[i] < l + \
									len(game.strategies[r]):
					strategies[r].append(game.strategies[r][args.s[i]-l])
					i += 1
				l += len(game.strategies[r])
			strategies = [strategies]
		else:
			raise IOError("Please specify either -f or -s for extract mode.")
		subgames = [subgame(game, s) for s in strategies]
		if len(subgames) == 1:
			subgames = subgames[0]

	print to_JSON_str(subgames)


if __name__ == "__main__":
	main()
