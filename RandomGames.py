#! /usr/bin/env python2.7

import GameIO as IO

from BasicFunctions import leading_zeros
from RoleSymmetricGame import Game, SampleGame, PayoffData, Profile

from functools import partial
from itertools import combinations
from bisect import bisect
from numpy.random import uniform as U, normal, multivariate_normal, beta, gumbel
from random import choice
from numpy import array, arange, zeros, fill_diagonal, cumsum
from sys import argv


def __make_asymmetric_game(N, S):
	"""
	N: number of players
	S: number of strategies
	"""
	roles = ["r" + leading_zeros(i,N) for i in range(N)]
	players = {r:1 for r in roles}
	strategies = {r:["s"+leading_zeros(i,S) for i in range(S)] for r in roles}
	return Game(roles, players, strategies)


def __make_symmetric_game(N, S):
	"""
	N: number of players
	S: number of strategies
	"""
	roles = ["All"]
	players = {"All":N}
	strategies = {"All":["s" + leading_zeros(i,S) for i in range(S)]}
	return Game(roles, players, strategies)


def independent(N, S, dstr=partial(U,-1,1)):
	"""
	All payoff values drawn independently according to specified distribution.

	N: number of players
	S: number of strategies
	dstr: distribution from which payoff values are independently drawn
	"""
	g = __make_asymmetric_game(N, S)
	for prof in g.allProfiles():
		g.addProfile({r:[PayoffData(prof[r].keys()[0], 1, dstr())] \
				for r in prof})
	return g


def covariant(N, S, mean_func=lambda:0, var=1, covar_func=partial(U,0,1)):
	"""
	Payoff values for each profile drawn according to multivariate normal.

	The multivariate normal for each profile has a constant mean-vector with 
	value drawn from mean_func, constant variance=var, and equal covariance 
	between all pairs of players, drawn from covar_func.

	N: number of players
	S: number of strategies
	mean_func: distribution from which mean payoff for each profile is drawn
	var: diagonal entries of covariance matrix
	covar_func: distribution from which the value of the off-diagonal 
				covariance matrix entries for each profile is drawn

	Both mean_func and covar_func should be numpy-style random number generators
	that can return an array.
	"""
	g = __make_asymmetric_game(N, S)
	mean = zeros(S)
	covar = zeros([S,S])
	for prof in g.allProfiles():
		mean.fill(mean_func())
		payoffs = multivariate_normal(mean, covar)
		covar.fill(covar_func())
		fill_diagonal(covar, var)
		g.addProfile({r:[PayoffData(prof[r].keys()[0], 1, payoffs[i])] \
				for i,r in enumerate(g.roles)})
	return g
	

def uniform_zero_sum(S, min_val=-1, max_val=1):
	"""
	2-player zero-sum game; player 1 payoffs drawn from a uniform distribution.

	S: number of strategies
	min_val: minimum for the uniform distribution
	max_val: maximum for the uniform distribution
	"""
	g = __make_asymmetric_game(2, S)
	for prof in g.allProfiles():
		row_strat = prof["r0"].keys()[0]
		row_val = U(min_val, max_val)
		col_strat = prof["r1"].keys()[0]
		p = {"r0":[PayoffData(row_strat, 1, row_val)], \
				"r1":[PayoffData(col_strat, 1, -row_val)]}
		g.addProfile(p)
	return g


def uniform_symmetric(N, S, min_val=-1, max_val=1):
	"""
	Symmetric game with each payoff value drawn from a uniform distribution.

	N: number of players
	S: number of strategies
	min_val: minimum for the uniform distribution
	max_val: maximum for the uniform distribution
	"""
	g = __make_symmetric_game(N, S)
	for prof in g.allProfiles():
		payoffs = []
		for strat, count in prof["All"].items():
			payoffs.append(PayoffData(strat, count, U(min_val, max_val)))
		g.addProfile({"All":payoffs})
	return g


def sym_2p2s(a=0, b=1, c=2, d=3, min_val=-1, max_val=1):
	"""
	Create a symmetric 2-player 2-strategy game of the specified form.

	Four payoff values get drawn from U(min_val, max_val), and then are assigned
	to profiles in order from smallest to largest according to the order 
	parameters as follows:

	   | s0  | s1  |
	---|-----|-----|
	s0 | a,a | b,c |
	s1 | c,b | d,d |
	---|-----|-----|
	
	So a=2,b=0,c=3,d=1 gives a prisoners' dilemma; a=0,b=3,c=1,d=2 gives a game
	of chicken.
	"""
	g = __make_symmetric_game(2, 2)
	payoffs = sorted(U(min_val, max_val, 4))
	g.addProfile({"All":[PayoffData(g.strategies["All"][0], 2, payoffs[a])]})
	g.addProfile({"All":[PayoffData(g.strategies["All"][0], 1, payoffs[b]), \
						PayoffData(g.strategies["All"][1], 1, payoffs[c])]})
	g.addProfile({"All":[PayoffData(g.strategies["All"][1], 2, payoffs[d])]})
	return g


def congestion(N, facilities, required):
	"""
	Generates random congestion games with N players and nCr(f,r) strategies.

	Congestion games are symmetric, so all players belong to role All. Each 
	strategy is a subset of size #required among the size #facilities set of 
	available facilities. Payoffs for each strategy are summed over facilities.
	Each facility's payoff consists of three components:

	-constant ~ U[0,#facilities]
	-linear congestion cost ~ U[-#required,0]
	-quadratic congestion cost ~ U[-1,0]
	"""
	roles = ["All"]
	players = {"All":N}
	strategies = {'+'.join(["f"+str(f) for f in strat]):strat for strat in \
			combinations(range(facilities), required)}
	facility_values = [array([U(facilities), U(-required), U(-1)]) for __ in \
			range(facilities)]
	g = Game(roles, players, {"All":strategies.keys()})
	for prof in g.allProfiles():
		payoffs = []
		useage = [0]*facilities
		for strat, count in prof["All"].items():
			for facility in strategies[strat]:
				useage[facility] += count
		for strat, count in prof["All"].items():
			payoffs.append(PayoffData(strat, count, [sum(useage[f]**arange(3) \
					* facility_values[f]) for f in strategies[strat]]))
		g.addProfile({"All":payoffs})
	return g


def local_effect(N, S):
	"""
	Generates random congestion games with N players and S strategies.

	Local effect games are symmetric, so all players belong to role All. Each
	strategy corresponds to a node in the G(N,2/S) local effect graph. Payoffs
	for each strategy consist of constant terms for each strategy, and
	interaction terms for the number of players choosing that strategy and each
	neighboring strategy.

	The one-strategy terms are drawn as follows:
	-constant ~ U[-N-S,N+S]
	-linear ~ U[-N,N]

	The neighbor strategy terms are drawn as follows:
	-linear ~ U[-S,S]
	-quadratic ~ U[-1,1]
	"""
	g = __make_symmetric_game(N, S)
	strategies = g.strategies["All"]
	local_effects = {s:{} for s in strategies}
	for s in strategies:
		for d in strategies:
			if s == d:
				local_effects[s][d] = [U(-N-S,N+S),U(-N,N)]
			elif U(0,S) > 2:
				local_effects[s][d] = [U(-S,S),U(-1,1)]
	for prof in g.allProfiles():
		payoffs = []
		for strat, count in prof["All"].items():
			value = local_effects[strat][strat][0] + \
					local_effects[strat][strat][1] * count
			for neighbor in local_effects[strat]:
				if neighbor not in prof["All"]:
					continue
				nc = prof["All"][neighbor]
				value += local_effects[strat][neighbor][0] * count
				value += local_effects[strat][neighbor][1] * count**2
			payoffs.append(PayoffData(strat, count, value))
		g.addProfile({"All":payoffs})
	return g


def polymatrix(N, S, matrix_game=partial(independent,2)):
	"""
	Creates a polymatrix game using the specified 2-player matrix game function.

	Each player's payoff in each profile is a sum over independent games played
	against each opponent. Each pair of players plays an instance of the
	specified random 2-player matrix game.

	N: number of players
	S: number of strategies
	matrix_game: a function of one argument (S) that returns 2-player, 
					S-strategy games.
	"""
	g = __make_asymmetric_game(N, S)
	matrices = {pair : matrix_game(S) for pair in combinations(g.roles, 2)}
	for prof in g.allProfiles():
		payoffs = {r:0 for r in g.roles}
		for role in g.roles:
			role_strat = prof[role].keys()[0]
			for other in g.roles:
				if role < other:
					m = matrices[(role, other)]
					p0 = sorted(m.players.keys())[0]
					p1 = sorted(m.players.keys())[1]
				elif role > other:
					m = matrices[(other, role)]
					p0 = sorted(m.players.keys())[1]
					p1 = sorted(m.players.keys())[0]
				else:
					continue
				other_strat = prof[other].keys()[0]
				s0 = m.strategies[p0][g.strategies[role].index(role_strat)]
				s1 = m.strategies[p1][g.strategies[other].index(other_strat)]
				m_prof = Profile({p0:{s0:1},p1:{s1:1}})
				payoffs[role] += m.getPayoff(m_prof, p0, s0)
		g.addProfile({r:[PayoffData(prof[r].keys()[0], 1, payoffs[r])] \
						for r in g.roles})
	return g


def add_noise(game, model, spread, samples):
	"""
	Generate sample game with random noise added to each payoff.
	
	game: a RSG.Game or RSG.SampleGame
	model: a 2-parameter function that generates mean-zero noise
	spread, samples: the parameters passed to the noise function
	"""
	sg = SampleGame(game.roles, game.players, game.strategies)
	for prof in game.knownProfiles():
		sg.addProfile({r:[PayoffData(s, prof[r][s], game.getPayoff(prof,r,s) + \
				model(spread, samples)) for s in prof[r]] for r in game.roles})
	return sg


def gaussian_mixture_noise(max_stdev, samples, modes=2, spread_mult=2):
	"""
	Generate Gaussian mixture noise to add to one payoff in a game.

	max_stdev: maximum standard deviation for the mixed distributions (also 
				affects how widely the mixed distributions are spaced)
	samples: numer of samples to take of every profile
	modes: number of Gaussians to mix
	spread_mult: multiplier for the spread of the Gaussians. Distance between
				the mean and the nearest distribution is drawn from 
				N(0,max_stdev*spread_mult).
	"""
	multipliers = arange(float(modes)) - float(modes-1)/2
	offset = normal(0, max_stdev * spread_mult)
	stdev = beta(2,1) * max_stdev
	return [normal(choice(multipliers)*offset, stdev) for _ in range(samples)]


normal_noise = partial(normal, 0)
unimodal_noise = partial(gaussian_mixture_noise, modes=1)
bimodal_noise = partial(gaussian_mixture_noise, modes=2)


def nonzero_gaussian_noise(max_stdev, samples, prob_pos=0.5, spread_mult=1):
	"""
	Generate Noise from a normal distribution centered up to one stdev from 0.

	With prob_pos=0.5, this implements the previous buggy output of 
	bimodal_noise.

	max_stdev: maximum standard deviation for the mixed distributions (also 
				affects how widely the mixed distributions are spaced)
	samples: numer of samples to take of every profile
	prob_pos: the probability that the noise mean for any payoff will be >0.
	spread_mult: multiplier for the spread of the Gaussians. Distance between
				the mean and the mean of the distribution is drawn from 
				N(0,max_stdev*spread_mult).
	"""
	offset = normal(0, max_stdev)*(1 if U(0,1) < prob_pos else -1)*spread_mult
	stdev = beta(2,1) * max_stdev
	return normal(offset, stdev, samples)


def uniform_noise(max_half_width, samples):
	"""
	Generate uniform random noise to add to one payoff in a game.

	max_range: maximum half-width of the uniform distribution
	samples: numer of samples to take of every profile
	"""
	hw = beta(2,1) * max_half_width
	return U(-hw, hw, samples)


def gumbel_noise(scale, samples):
	location = -0.5772*scale
	return gumbel(location, scale, samples)


def mix_models(models, rates, spread, samples):
	"""
	Generate SampleGame with noise drawn from several models.

	models: a list of 2-parameter noise functions to draw from
	rates: the probabilites with which a payoff will be drawn from each model
	spread, samples: the parameters passed to the noise functions
	"""
	cum_rates = cumsum(rates)
	m = models[bisect(cum_rates, U(0,1))]
	return m(spread, samples)


u80b20_noise = partial(mix_models, [unimodal_noise, bimodal_noise], [.8,.2])
u60b40_noise = partial(mix_models, [unimodal_noise, bimodal_noise], [.6,.4])
u40b60_noise = partial(mix_models, [unimodal_noise, bimodal_noise], [.4,.6])
u20b80_noise = partial(mix_models, [unimodal_noise, bimodal_noise], [.2,.8])


def rescale_payoffs(game, min_payoff=0, max_payoff=100):
	"""
	Rescale game's payoffs to be in the range [min_payoff, max_payoff].

	Modifies game.values in-place.
	"""
	game.makeArrays()
	min_val = game.values.min()
	max_val = game.values.max()
	game.values -= min_val
	game.values *= (max_payoff - min_payoff)
	game.values /= (max_val - min_val)
	game.values += min_payoff


def parse_args():
	parser = IO.io_parser(description="Generate random games.")
	parser.add_argument("type", choices=["uZS", "uSym", "CG", "LEG"], help= \
			"Type of random game to generate. uZS = uniform zero sum. " +\
			"uSym = uniform symmetric. CG = congestion game.")
	parser.add_argument("count", type=int, help="Number of random games " +\
			"to create.")
	parser.add_argument("-noise", choices=["none", "normal", \
			"gauss_mix"], default="None", help="Noise function.")
	parser.add_argument("-noise_args", nargs="*", default=[], \
			help="Arguments to be passed to the noise function.")
	parser.add_argument("-game_args", nargs="*", default=[], \
			help="Additional arguments for game generator function.")
	assert "-input" not in argv, "no input JSON required"
	argv = argv[:3] + ["-input", None] + argv[3:]
	return parser.parse_args()


def main():
	args = parse_args()

	if args.type == "uZS":
		game_func = uniform_zero_sum
		assert len(args.game_args) == 1, "game_args must specify strategy count"
		game_args = map(int, args.game_args)
	elif args.type == "uSym":
		game_func = uniform_symmetric
		assert len(args.game_args) == 2, "game_args must specify player and "+\
									"strategy counts"
	elif args.type == "CG":
		game_func = congestion
		assert len(args.game_args) == 3, "game_args must specify player, "+\
									"facility, and required facility counts"
		game_args = map(int, args.game_args)
	elif args.type == "LEG":
		game_func = local_effect
		assert len(args.game_args) == 2, "game_args must specify player and "+\
									"strategy counts"
		game_args = map(int, args.game_args)
	
	games = [game_func(*game_args) for __ in range(args.count)]

	if args.noise == "normal":
		assert len(args.noise_args) == 2, "noise_args must specify stdev "+\
											"and sample count"
		noise_args = [float(args.noise_args[0]), int(args.noise_args[1])]
		games = map(lambda g: normal_noise(g, *noise_args), games)	
	elif args.noise == "gauss_mix":
		assert len(args.noise_args) == 3, "noise_args must specify max "+\
									"stdev, sample count, and number of modes"
		noise_args = [float(args.noise_args[0]), int(args.noise_args[1]), \
						int(args.noise_args[2])]
		games = map(lambda g: gaussian_mixture_noise(g, *noise_args), games)	

	if len(games) == 1:
		print IO.to_JSON_str(games[0])
	else:
		print IO.to_JSON_str(games)


if __name__ == "__main__":
	main()
