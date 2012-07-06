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
import RoleSymmetricGame as RSG


class TestProfileDetection(unittest.TestCase):
	def setUp(self):
		self.cliques = IO.read(join(path[0], "cliques_full.json"))

	def test_profile_array_detection(self):
		self.assertTrue(RSG.is_profile_array(self.cliques.counts[0]))
		self.assertFalse(RSG.is_mixture_array(self.cliques.counts[0]))
		self.assertFalse(RSG.is_pure_profile(self.cliques.counts[0]))
		self.assertFalse(RSG.is_mixed_profile(self.cliques.counts[0]))

	def test_pure_profile_detection(self):
		self.assertFalse(RSG.is_profile_array(self.cliques.toProfile( \
				self.cliques.counts[0])))
		self.assertFalse(RSG.is_mixture_array(self.cliques.toProfile( \
				self.cliques.counts[0])))
		self.assertTrue(RSG.is_pure_profile(self.cliques.toProfile( \
				self.cliques.counts[0])))
		self.assertFalse(RSG.is_mixed_profile(self.cliques.toProfile( \
				self.cliques.counts[0])))

	def test_mixture_array_detection(self):
		self.assertFalse(RSG.is_profile_array(self.cliques.uniformMixture()))
		self.assertTrue(RSG.is_mixture_array(self.cliques.uniformMixture()))
		self.assertFalse(RSG.is_pure_profile(self.cliques.uniformMixture()))
		self.assertFalse(RSG.is_mixed_profile(self.cliques.uniformMixture()))

	def test_mixed_profile_detection(self):
		self.assertFalse(RSG.is_profile_array(self.cliques.toProfile( \
				self.cliques.uniformMixture())))
		self.assertFalse(RSG.is_mixture_array(self.cliques.toProfile( \
				self.cliques.uniformMixture())))
		self.assertFalse(RSG.is_pure_profile(self.cliques.toProfile( \
				self.cliques.uniformMixture())))
		self.assertTrue(RSG.is_mixed_profile(self.cliques.toProfile( \
				self.cliques.uniformMixture())))


class TestProfileRegret(unittest.TestCase):
	def setUp(self):
		self.cd_bl = IO.read(join(path[0], "conditional_dominance_BL.xml"))
		self.spd = IO.read(join(path[0], "strict_pure_dominance.xml"))
		self.cliques = IO.read(join(path[0], "cliques_full.json"))
		self.BC = RSG.Profile({"Column":{"Center":1},"Row":{"Bottom":1}})
		self.AAHL = RSG.Profile({"buyers":{"accept":2}, "sellers":{"high":1, \
				"low":1}})

	def test_dev_regret(self):
		self.assertEqual(R.regret(self.spd, self.BC, "Row", "Bottom", \
				"Middle"), 1)
		self.assertEqual(R.regret(self.spd, self.BC, "Column", "Center", \
				"Right"), -9)
		self.assertEqual(R.regret(self.spd, self.BC, "Column", "Center", \
				"Center"), 0)
		self.assertEqual(R.regret(self.cliques, self.AAHL, "sellers", \
				"low", "high"), 2)
		self.assertEqual(R.regret(self.cliques, self.AAHL, "sellers", \
				"high", "low"), 4)
		self.assertEqual(R.regret(self.cliques, self.AAHL, "buyers", \
				"accept", "reject"), -4)
		self.assertRaises(KeyError, R.regret, self.cliques, self.AAHL, \
				"buyers", "reject", "accept")

	def test_strat_regret(self):
		self.assertEqual(R.regret(self.spd, self.BC, "Row", "Bottom"), 1)
		self.assertEqual(R.regret(self.cliques, self.AAHL, "buyers", \
				"accept"), -4)

	def test_role_regret(self):
		self.assertEqual(R.regret(self.spd, self.BC, "Row"), 1)
		self.assertEqual(R.regret(self.cliques, self.AAHL, "buyers"), -4)
		self.assertEqual(R.regret(self.cliques, self.AAHL, "sellers"), 4)

	def test_profile_regret(self):
		self.assertEqual(R.regret(self.spd, self.BC), 1)
		self.assertEqual(R.regret(self.cliques, self.AAHL), 4)

	def test_missing_regret(self):
		self.assertEqual(R.regret(self.cd_bl, self.BC, "Column", \
				"Center", "Left"), float('inf'))
		self.assertEqual(R.regret(self.cd_bl, self.BC, "Column", \
				"Center"), float('inf'))
		self.assertEqual(R.regret(self.cd_bl, self.BC, "Column"), float('inf'))
		self.assertEqual(R.regret(self.cd_bl, self.BC), float('inf'))

	def test_regret_bound(self):
		self.assertEqual(R.regret(self.cd_bl, self.BC, "Column", \
				"Center", "Left", True), float('-inf'))
		self.assertEqual(R.regret(self.cd_bl, self.BC, "Column", \
				"Center", bound=True), -9)
		self.assertEqual(R.regret(self.cd_bl, self.BC, "Column", \
				bound=True), -9)
		self.assertEqual(R.regret(self.cd_bl, self.BC, bound=True), 1)


class TestMixtureRegret(unittest.TestCase):
	pass


class TestDominates(unittest.TestCase):
	def setUp(self):
		self.spd = IO.read(join(path[0], "strict_pure_dominance.xml"))
		self.cd_br = IO.read(join(path[0], "conditional_dominance_BR.xml"))
		self.cd_bc = IO.read(join(path[0], "conditional_dominance_BC.xml"))
		self.wpd = IO.read(join(path[0], "weak_pure_dominance.xml"))

	def test_weak_dominance(self):
		self.assertFalse(D.dominates(self.wpd, "Column", "Center", "Left", \
				weak=False))
		self.assertTrue(D.dominates(self.wpd, "Column", "Center", "Left", \
				weak=True))

	def test_unconditional_dominance(self):
		self.assertTrue(D.dominates(self.spd, "Column", "Center", "Right", 0))
		self.assertFalse(D.dominates(self.spd, "Row", "Middle", "Bottom", 0))
		self.assertTrue(D.dominates(self.cd_br, "Column", "Center", "Right", \
				conditional=False))
		self.assertTrue(D.dominates(self.cd_br, "Row", "Middle", "Bottom", \
				conditional=False))
		self.assertTrue(D.dominates(self.cd_bc, "Column", "Center", "Right", \
				conditional=False))
		self.assertFalse(D.dominates(self.cd_bc, "Row", "Middle", "Bottom", \
				conditional=False))

	def test_conditional_dominance(self):
		self.assertTrue(D.dominates(self.spd, "Column", "Center", "Right", 1))
		self.assertFalse(D.dominates(self.spd, "Row", "Middle", "Bottom", 1))
		self.assertTrue(D.dominates(self.cd_br, "Column", "Center", "Right", \
				conditional=True))
		self.assertTrue(D.dominates(self.cd_br, "Row", "Middle", "Bottom", \
				conditional=True))
		self.assertFalse(D.dominates(self.cd_bc, "Column", "Center", "Right", \
				conditional=True))
		self.assertFalse(D.dominates(self.cd_bc, "Row", "Middle", "Bottom", \
				conditional=True))

	def test_conservative_dominance(self):
		self.assertTrue(D.dominates(self.spd, "Column", "Center", "Right", 2))
		self.assertFalse(D.dominates(self.spd, "Row", "Middle", "Bottom", 2))
		self.assertFalse(D.dominates(self.cd_br, "Column", "Center", "Right", \
				conditional=2))
		self.assertFalse(D.dominates(self.cd_br, "Row", "Middle", "Bottom", \
				conditional=2))
		self.assertFalse(D.dominates(self.cd_bc, "Column", "Center", "Right", \
				conditional=2))
		self.assertFalse(D.dominates(self.cd_bc, "Row", "Middle", "Bottom", \
				conditional=2))


class Testiterated_elimination(unittest.TestCase):
	def setUp(self):
		self.nbr = IO.read(join(path[0], "never_best_response.xml"))
		self.wpd = IO.read(join(path[0], "weak_pure_dominance.xml"))
		self.spd = IO.read(join(path[0], "strict_pure_dominance.xml"))

	def test_IE_SPD(self):
		"""
		iterated elimination of strictly pure-strategy dominated strategies
		"""
		self.assertEqual(len(D.iterated_elimination(self.nbr, \
				D.pure_strategy_dominance, weak=False)), 6)
		self.assertEqual(len(D.iterated_elimination(self.wpd, \
				D.pure_strategy_dominance, weak=False)), 9)
		self.assertEqual(len(D.iterated_elimination(self.spd, \
				D.pure_strategy_dominance, weak=False)), 1)

	def test_IE_WPD(self):
		"""
		iterated elimination of weakly pure-strategy dominated strategies
		"""
		self.assertEqual(len(D.iterated_elimination(self.nbr, \
				D.pure_strategy_dominance, weak=True)), 6)
		self.assertEqual(len(D.iterated_elimination(self.wpd, \
				D.pure_strategy_dominance, weak=True)), 1)
		self.assertEqual(len(D.iterated_elimination(self.spd, \
				D.pure_strategy_dominance, weak=True)), 1)

	def test_IE_NBR(self):
		"""
		iterated elimination of never weak best-response strategies
		"""
		self.assertEqual(len(D.iterated_elimination(self.nbr, \
				D.never_best_response)), 4)
		self.assertEqual(len(D.iterated_elimination(self.wpd, \
				D.never_best_response)), 1)
		self.assertEqual(len(D.iterated_elimination(self.spd, \
				D.never_best_response)), 1)


class TestConditionaliterated_elimination(unittest.TestCase):
	def setUp(self):
		self.cd_bl = IO.read(join(path[0], "conditional_dominance_BL.xml"))
		self.cd_bc = IO.read(join(path[0], "conditional_dominance_BC.xml"))
		self.cd_br = IO.read(join(path[0], "conditional_dominance_BR.xml"))
		self.cd_bcr = IO.read(join(path[0],"conditional_dominance_BCR.xml"))

	def test_IE_unconditional_PD(self):
		self.assertEqual(len(D.iterated_elimination(self.cd_bl, \
				D.pure_strategy_dominance, conditional=False)), 1)
		self.assertEqual(len(D.iterated_elimination(self.cd_bc, \
				D.pure_strategy_dominance, conditional=False)), 1)
		self.assertEqual(len(D.iterated_elimination(self.cd_br, \
				D.pure_strategy_dominance, conditional=False)), 1)
		self.assertEqual(len(D.iterated_elimination(self.cd_bcr, \
				D.pure_strategy_dominance, conditional=False)), 1)

	def test_IE_conditional_PD(self):
		"""
		iterated elimination of conditionally strictly pure-strategy dominated
		strategies in games with partial data
		"""
		self.assertEqual(len(D.iterated_elimination(self.cd_bl, \
				D.pure_strategy_dominance, conditional=True)), 1)
		self.assertEqual(len(D.iterated_elimination(self.cd_bc, \
				D.pure_strategy_dominance, conditional=True)), 8)
		self.assertEqual(len(D.iterated_elimination(self.cd_br, \
				D.pure_strategy_dominance, conditional=True)), 1)
		self.assertEqual(len(D.iterated_elimination(self.cd_bcr, \
				D.pure_strategy_dominance, conditional=True)), 1)

	def test_conservative_PD(self):
		self.assertEqual(len(D.iterated_elimination(self.cd_bl, \
				D.pure_strategy_dominance, conditional=2)), 5)
		self.assertEqual(len(D.iterated_elimination(self.cd_bc, \
				D.pure_strategy_dominance, conditional=2)), 8)
		self.assertEqual(len(D.iterated_elimination(self.cd_br, \
				D.pure_strategy_dominance, conditional=2)), 8)
		self.assertEqual(len(D.iterated_elimination(self.cd_bcr, \
				D.pure_strategy_dominance, conditional=2)), 5)

	def test_IE_CNBR(self):
		"""
		iterated elimination of conditionally never weak best-response
		strategies in games with partial data
		"""
		self.assertEqual(len(D.iterated_elimination(self.cd_bl, \
				D.never_best_response, conditional=True)), 5)
		self.assertEqual(len(D.iterated_elimination(self.cd_bl, \
				D.never_best_response, conditional=False)), 1)

		self.assertEqual(len(D.iterated_elimination(self.cd_bc, \
				D.never_best_response, conditional=True)), 5)
		self.assertEqual(len(D.iterated_elimination(self.cd_bc, \
				D.never_best_response, conditional=False)), 1)

		self.assertEqual(len(D.iterated_elimination(self.cd_br, \
				D.never_best_response, conditional=True)), 8)
		self.assertEqual(len(D.iterated_elimination(self.cd_br, \
				D.never_best_response, conditional=False)), 1)

		self.assertEqual(len(D.iterated_elimination(self.cd_bcr, \
				D.never_best_response, conditional=True)), 7)
		self.assertEqual(len(D.iterated_elimination(self.cd_bcr, \
				D.never_best_response, conditional=False)), 1)


class TestNash(unittest.TestCase):
	def setUp(self):
		self.pd_sym = IO.read(join(path[0], "PD_sym.xml"))
		self.pd_str = IO.read(join(path[0], "PD_str.xml"))
		self.rps_sym = IO.read(join(path[0], "RPS_sym.xml"))
		self.rps_str = IO.read(join(path[0], "RPS_str.xml"))

	def test_pure_nash(self):
		self.assertEqual(len(N.pure_nash(self.pd_sym)), 1)
		self.assertEqual(N.pure_nash(self.pd_sym)[0], {"All":{"d":2}})
		self.assertEqual(len(N.pure_nash(self.pd_str)), 1)
		self.assertEqual(N.pure_nash(self.pd_str)[0], {"Alice":{"d":1}, \
				"Bob":{"d":1}})
		self.assertEqual(len(N.pure_nash(self.rps_sym)), 0)
		self.assertEqual(len(N.pure_nash(self.rps_str)), 0)

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

	def test_mixed_nash(self):
		expected_eq = np.array([[0.,1.]]*2)
		found_eq = N.mixed_nash(self.pd_str)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[0.,1.]])
		found_eq = N.mixed_nash(self.pd_sym)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[1./3]*3]*2)
		found_eq = N.mixed_nash(self.rps_str)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[1./3]*3])
		found_eq = N.mixed_nash(self.rps_sym)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))


class TestSparseGame(unittest.TestCase):
	def setUp(self):
		self.cliques_full = IO.read(join(path[0], "cliques_full.json"))
		self.cliques_1 = IO.read(join(path[0], "cliques_HLRR.json"))
		self.cliques_2 = IO.read(join(path[0], "cliques_HLRR_HLAA.json"))
		self.cliques_4 = IO.read(join(path[0], "cliques_all_sym.json"))
		self.ss = IO.read(join(path[0], "sparse_symmetric.xml"))

	def test_cliques(self):
		self.assertEqual(map(len, S.cliques(self.cliques_full)), [9])
		self.assertEqual(map(len, S.cliques(self.cliques_1)), [3,3,3])
		self.assertEqual(map(len, S.cliques(self.cliques_2)), [3,3])
		self.assertEqual(map(len, S.cliques(self.cliques_4)), [])
		self.assertEqual(map(len, S.cliques(self.ss)), [3])

	def test_SparseRegret(self):
		clique = S.cliques(self.ss)[0]
		clique_eq = N.mixed_nash(clique)[0]
		full_candidate = S.translate(clique_eq, clique, self.ss)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="A"), 0)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="B"), 0)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="C"), 1)
		self.assertEqual(R.regret(self.ss, full_candidate, deviation="D"), -1)
		self.assertEqual(R.regret(self.ss, full_candidate), 1)


class TestDegenerateGame(unittest.TestCase):
	def setUp(self):
		self.one_player = IO.read(join(path[0], "one_player.xml"))
		self.one_strategy = IO.read(join(path[0], "one_strategy.xml"))
		self.one_profile = IO.read(join(path[0], "one_profile.xml"))

	def test_IEDS(self):
		self.assertEqual(len(D.iterated_elimination(self.one_player, \
				D.pure_strategy_dominance)), 1)
		self.assertEqual(len(D.iterated_elimination(self.one_strategy, \
				D.pure_strategy_dominance)), 1)
		self.assertEqual(len(D.iterated_elimination(self.one_profile, \
				D.pure_strategy_dominance)), 1)

	def test_cliques(self):
		self.assertEqual(map(len, S.cliques(self.one_player)), [2])
		self.assertEqual(map(len, S.cliques(self.one_strategy)), [1])
		self.assertEqual(map(len, S.cliques(self.one_profile)), [1])

	def test_pure_nash(self):
		self.assertEqual(N.pure_nash(self.one_player)[0], {"All":{"s2":1}})
		self.assertEqual(N.pure_nash(self.one_strategy)[0], {"All":{"s1":2}})
		self.assertEqual(len(N.pure_nash(self.one_profile)), 0)

	def test_mixed_nash(self):
		expected_eq = np.array([[0.,1.]])
		found_eq = N.mixed_nash(self.one_player)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		expected_eq = np.array([[1.]])
		found_eq = N.mixed_nash(self.one_strategy)
		self.assertEqual(len(found_eq), 1)
		self.assertTrue(np.allclose(expected_eq, found_eq[0]))

		found_eq = N.mixed_nash(self.one_profile)
		self.assertEqual(len(found_eq), 0)


if __name__ == '__main__':
    unittest.main(verbosity=1)

