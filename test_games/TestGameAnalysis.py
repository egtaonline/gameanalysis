#!/usr/local/bin/python2.7

import unittest

from os.path import dirname, join
from sys import path

import numpy as np

path.append(dirname(path[0]))

import GameIO as IO
import Dominance as D
import Nash as N
import Subgames as S
import Regret as R

class TestDominance(unittest.TestCase):
	def setUp(self):
		self.nbr = IO.readGame(join(path[0], "never_best_response.xml"))
		self.wpd = IO.readGame(join(path[0], "weak_pure_dominance.xml"))
		self.spd = IO.readGame(join(path[0], "strict_pure_dominance.xml"))

	def test_StrictPureDominance(self):
		"""
		iterated elimination of strictly pure-strategy dominated strategies
		"""
		self.assertEqual(len(D.IteratedElimination(self.nbr, \
				D.PureStrategyDominance, weak=False)), 6)
		self.assertEqual(len(D.IteratedElimination(self.wpd, \
				D.PureStrategyDominance, weak=False)), 9)
		self.assertEqual(len(D.IteratedElimination(self.spd, \
				D.PureStrategyDominance, weak=False)), 1)

	def test_WeakPureDominance(self):
		"""
		iterated elimination of weakly pure-strategy dominated strategies
		"""
		self.assertEqual(len(D.IteratedElimination(self.nbr, \
				D.PureStrategyDominance, weak=True)), 6)
		self.assertEqual(len(D.IteratedElimination(self.wpd, \
				D.PureStrategyDominance, weak=True)), 1)
		self.assertEqual(len(D.IteratedElimination(self.spd, \
				D.PureStrategyDominance, weak=True)), 1)

	def test_NeverBestResponse(self):
		"""
		iterated elimination of never weak best-response strategies
		"""
		self.assertEqual(len(D.IteratedElimination(self.nbr, \
				D.NeverBestResponse)), 4)
		self.assertEqual(len(D.IteratedElimination(self.wpd, \
				D.NeverBestResponse)), 1)
		self.assertEqual(len(D.IteratedElimination(self.spd, \
				D.NeverBestResponse)), 1)


class TestConditionalDominance(unittest.TestCase):
	def setUp(self):
		self.cd_bl = IO.readGame(join(path[0], "conditional_dominance_BL.xml"))
		self.cd_bc = IO.readGame(join(path[0], "conditional_dominance_BC.xml"))
		self.cd_br = IO.readGame(join(path[0], "conditional_dominance_BR.xml"))
		self.cd_bcr = IO.readGame(join(path[0],"conditional_dominance_BCR.xml"))

	def test_UnconditionalPureDominance(self):
		self.assertEqual(len(D.IteratedElimination(self.cd_bl, \
				D.PureStrategyDominance, conditional=False)), 1)
		self.assertEqual(len(D.IteratedElimination(self.cd_bc, \
				D.PureStrategyDominance, conditional=False)), 1)
		self.assertEqual(len(D.IteratedElimination(self.cd_br, \
				D.PureStrategyDominance, conditional=False)), 1)
		self.assertEqual(len(D.IteratedElimination(self.cd_bcr, \
				D.PureStrategyDominance, conditional=False)), 1)

	def test_ConditionalPureDominance(self):
		"""
		iterated elimination of conditionally strictly pure-strategy dominated
		strategies in games with partial data
		"""
		self.assertEqual(len(D.IteratedElimination(self.cd_bl, \
				D.PureStrategyDominance, conditional=True)), 1)
		self.assertEqual(len(D.IteratedElimination(self.cd_bc, \
				D.PureStrategyDominance, conditional=True)), 8)
		self.assertEqual(len(D.IteratedElimination(self.cd_br, \
				D.PureStrategyDominance, conditional=True)), 1)
		self.assertEqual(len(D.IteratedElimination(self.cd_bcr, \
				D.PureStrategyDominance, conditional=True)), 1)

	def test_ConservativePureDominance(self):
		self.assertEqual(len(D.IteratedElimination(self.cd_bl, \
				D.PureStrategyDominance, conditional=2)), 5)
		self.assertEqual(len(D.IteratedElimination(self.cd_bc, \
				D.PureStrategyDominance, conditional=2)), 8)
		self.assertEqual(len(D.IteratedElimination(self.cd_br, \
				D.PureStrategyDominance, conditional=2)), 8)
		self.assertEqual(len(D.IteratedElimination(self.cd_bcr, \
				D.PureStrategyDominance, conditional=2)), 5)

	def test_ConditionalNeverBestResponse(self):
		"""
		iterated elimination of conditionally never weak best-response
		strategies in games with partial data
		"""
		self.assertEqual(len(D.IteratedElimination(self.cd_bl, \
				D.NeverBestResponse, conditional=True)), 5)
		self.assertEqual(len(D.IteratedElimination(self.cd_bl, \
				D.NeverBestResponse, conditional=False)), 1)

		self.assertEqual(len(D.IteratedElimination(self.cd_bc, \
				D.NeverBestResponse, conditional=True)), 5)
		self.assertEqual(len(D.IteratedElimination(self.cd_bc, \
				D.NeverBestResponse, conditional=False)), 1)

		self.assertEqual(len(D.IteratedElimination(self.cd_br, \
				D.NeverBestResponse, conditional=True)), 8)
		self.assertEqual(len(D.IteratedElimination(self.cd_br, \
				D.NeverBestResponse, conditional=False)), 1)

		self.assertEqual(len(D.IteratedElimination(self.cd_bcr, \
				D.NeverBestResponse, conditional=True)), 7)
		self.assertEqual(len(D.IteratedElimination(self.cd_bcr, \
				D.NeverBestResponse, conditional=False)), 1)


class TestNash(unittest.TestCase):
	def setUp(self):
		self.pd_sym = IO.readGame(join(path[0], "PD_sym.xml"))
		self.pd_str = IO.readGame(join(path[0], "PD_str.xml"))
		self.rps_sym = IO.readGame(join(path[0], "RPS_sym.xml"))
		self.rps_str = IO.readGame(join(path[0], "RPS_str.xml"))

	def test_PureNash(self):
		self.assertEqual(len(N.PureNash(self.pd_sym)), 1)
		self.assertEqual(N.PureNash(self.pd_sym)[0], {"All":{"d":2}})
		self.assertEqual(len(N.PureNash(self.pd_str)), 1)
		self.assertEqual(N.PureNash(self.pd_str)[0], {"Alice":{"d":1}, \
				"Bob":{"d":1}})
		self.assertEqual(len(N.PureNash(self.rps_sym)), 0)
		self.assertEqual(len(N.PureNash(self.rps_str)), 0)

	def test_ExpectedValues(self):
		pd_str_mix = np.array([[0.1,0.9],[0.2,0.8]])
		pd_str_EVs = np.array([[0.4,1.4],[0.2,1.2]])
		self.assertTrue(np.allclose(self.pd_str.expectedValues(\
				pd_str_mix), pd_str_EVs))

		pd_sym_mix = np.array([[0.3,0.7]])
		pd_sym_EVs = np.array([[0.6,1.6]])
		self.assertTrue(np.allclose(self.pd_sym.expectedValues(\
				pd_sym_mix), pd_sym_EVs))

		rps_str_mix = np.array([[0.1,0.2,0.7],[0.3,0.4,0.3]])
		rps_str_EVs = np.array([[0.1,0.0,-0.1],[-0.5,0.6,-0.1]])
		self.assertTrue(np.allclose(self.rps_str.expectedValues(\
				rps_str_mix), rps_str_EVs))

		rps_sym_mix = np.array([[0.,0.4,0.6]])
		rps_sym_EVs = np.array([[-0.2,0.6,-0.4]])
		self.assertTrue(np.allclose(self.rps_sym.expectedValues(\
				rps_sym_mix), rps_sym_EVs))

	def test_MixedNash(self):
		expected_eq = np.array([[0.,1.]]*2)
		found_eq = N.MixedNash(self.pd_str)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[0.,1.]])
		found_eq = N.MixedNash(self.pd_sym)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[1./3]*3]*2)
		found_eq = N.MixedNash(self.rps_str)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[1./3]*3])
		found_eq = N.MixedNash(self.rps_sym)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))


class TestSparseGame(unittest.TestCase):
	def setUp(self):
		self.cliques_full = IO.readGame(join(path[0], "cliques_full.json"))
		self.cliques_1 = IO.readGame(join(path[0], "cliques_HLRR.json"))
		self.cliques_2 = IO.readGame(join(path[0], "cliques_HLRR_HLAA.json"))
		self.cliques_4 = IO.readGame(join(path[0], "cliques_all_sym.json"))
		self.ss = IO.readGame(join(path[0], "sparse_symmetric.xml"))

	def test_Cliques(self):
		self.assertEqual(map(len, S.Cliques(self.cliques_full)), [9])
		self.assertEqual(map(len, S.Cliques(self.cliques_1)), [3,3,3])
		self.assertEqual(map(len, S.Cliques(self.cliques_2)), [3,3])
		self.assertEqual(map(len, S.Cliques(self.cliques_4)), [])
		self.assertEqual(map(len, S.Cliques(self.ss)), [3])

	def test_SparseRegret(self):
		clique = S.Cliques(self.ss)[0]
		clique_eq = N.MixedNash(clique)[0]
		full_candidate = S.translate(clique_eq, clique, self.ss)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="A"), 0)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="B"), 0)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="C"), 1)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="D"), -1)
		self.assertEqual(R.regret(self.ss, full_candidate), 1)


class TestDegenerateGame(unittest.TestCase):
	def setUp(self):
		self.one_player = IO.readGame(join(path[0], "one_player.xml"))
		self.one_strategy = IO.readGame(join(path[0], "one_strategy.xml"))
		self.one_profile = IO.readGame(join(path[0], "one_profile.xml"))

	def test_IEDS(self):
		self.assertEqual(len(D.IteratedElimination(self.one_player, \
				D.PureStrategyDominance)), 1)
		self.assertEqual(len(D.IteratedElimination(self.one_strategy, \
				D.PureStrategyDominance)), 1)
		self.assertEqual(len(D.IteratedElimination(self.one_profile, \
				D.PureStrategyDominance)), 1)

	def test_Cliques(self):
		self.assertEqual(map(len, S.Cliques(self.one_player)), [2])
		self.assertEqual(map(len, S.Cliques(self.one_strategy)), [1])
		self.assertEqual(map(len, S.Cliques(self.one_profile)), [1])

	def test_PureNash(self):
		self.assertEqual(N.PureNash(self.one_player)[0], {"All":{"s2":1}})
		self.assertEqual(N.PureNash(self.one_strategy)[0], {"All":{"s1":2}})
		self.assertEqual(len(N.PureNash(self.one_profile)), 0)

	def test_MixedNash(self):
		expected_eq = np.array([[0.,1.]])
		found_eq = N.MixedNash(self.one_player)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[1.]])
		found_eq = N.MixedNash(self.one_strategy)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		found_eq = N.MixedNash(self.one_profile)
		self.assertEqual(len(found_eq), 0)


if __name__ == '__main__':
    unittest.main(verbosity=1)

