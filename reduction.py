"""Module for doing reductions while quiessing"""
# pylint: disable=relative-import

import copy

import analysis

import GameIO as gameio
import Reductions as reductions

class no_reduction(object):
    """A reduction where nothing is changed"""

    def reduce_game_data(self, game_summary):
        """Returns the same data that passed in"""
        return game_summary

    def expand_profile(self, profile):
        """Returns an generator of the singleton profile"""
        yield profile

class dpr_reduction(object):
    """A DPR reduction

    reduced counts is a dictionary mapping role to reduced counts for that
    role

    """
    def __init__(self, full_counts, reduced_counts):
        self.full_counts = full_counts
        self.reduced_counts = reduced_counts

    def reduce_game_data(self, game_summary):
        """Returns the DPR reduced version of a game summary"""
        # pylint: disable=protected-access
        return gameio.to_JSON_obj(reductions.deviation_preserving_reduction(
            gameio.read_JSON(game_summary), self.reduced_counts))

    def expand_profile(self, profile):
        """Returns a generator for all full game profiles necessary to create the
        reduction

        """

        for role, strats in profile.iteritems():
            for strat in strats:
                # pylint: disable=cell-var-from-loop pylint is wrong about what's happening
                # Copy and monkey patch to get function to pass and meet mutable invariant
                monkey_prof = analysis.profile(copy.deepcopy(profile))
                monkey_prof.asDict = lambda: monkey_prof
                yield analysis.profile(reductions.full_prof_DPR(
                    monkey_prof, role, strat, self.full_counts))
