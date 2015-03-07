#!/usr/bin/env python
import argparse
import requests
import json
import time
import itertools
import copy
from collections import Counter

import egtaonlineapi as egta

parser = argparse.ArgumentParser(description='Quiesce a generic scheduler on egtaonline.')
parser.add_argument('-a', '--auth', metavar='auth_token', help='An authorization token to allow access to egtaonline.')
parser.add_argument('-s', '--scheduler', metavar='scheduler', help='The name or id of the scheduler to quiesce.')
parser.add_argument('-g', '--game', metavar='game_id', type=int, default=None, help='The id of the game used to indicate how to schedule. If not provided, this will try and determine the game, but may fail.')
parser.add_argument('-n', '--max-profiles', metavar='num-profiles', type=int, default=10000, help='Maximum number of profiles to ever have scheduled at a time. Defaults to 10000.')
parser.add_argument('-t', '--sleep-time', metavar='delta', type=int, default=600, help='Time to wait in seconds between checking egtaonline for job completion. Defaults to 300.')


class subgame(dict):
    """Modification of a dict with convenience methods and verification"""
    
    def __init__(self, *args, **kwargs):
        super(subgame, self).__init__(*args, **kwargs)
        self.validate()

    def validate(self):
        """Validate representation of this subgame

        Throws an assertion error is incorrect

        """
        for role, strats in self.iteritems():
            assert isinstance(role, basestring) , "roles must be strings"
            assert isinstance(strats, set) , "strategies must be in sets"
            for strat in strats:
                assert isinstance(strat, basestring) , "strategies must be strings"

    def get_subgame_profiles(self, role_counts):
        """Returns an iterable over all subgame profiles"""
        # Compute the product of assignments by role and turn back into dictionary
        return (egta.profile(rs) for rs in itertools.product(
            # Iterate through all strategy allocations per role and compute
            # every allocation of agents
            *([(r, Counter(sprof)) for sprof
               in itertools.combinations_with_replacement(ss, role_counts[r])]
              # For each role
              for r, ss in self.iteritems())))

    def get_deviation_profiles(self, full_game, role_counts):
        """Returns an iterable over all deviations from every subgame profile"""
        for role, strats in self.iteritems():
            deviation_counts = role_counts.copy()
            deviation_counts[role] -= 1
            for prof in self.get_subgame_profiles(deviation_counts):
                for deviation in full_game[role].difference(strats):
                    deviation_prof = prof.copy()
                    deviation_prof[role] = deviation_prof[role].copy()
                    deviation_prof[role][deviation] = 1
                    yield deviation_prof

    def __repr__(self):
        return "subgame(%s)" % super(subgame, self).__repr__()

class quieser(object):
    """Class to manage quiesing of a scheduler"""
    def __init__(self, scheduler, auth_token, game=None, max_profiles=10000,
                 sleep_time=300):
        # Get api and access to standard objects
        self.api = egta.egtaonline(auth_token)
        self.scheduler = self.api.get_scheduler(scheduler, verbose=True)
        self.simulator = self.api.get_simulator(self.scheduler.simulator_id)
        sim_inst_id = self.api.get_scheduler(self.scheduler.id).simulator_instance_id
        if game is None:
            # Hack to find game with same simulator instance id
            game = egta._only(g for g in self.api.get_games()
                if g.simulator_instance_id == sim_inst_id).id
        self.game = self.api.get_game(game, granularity='summary')

        # Set other game information
        self.role_counts = {r['name']: r['count'] for r in self.game.roles}
        self.full_game = subgame(
            (r['name'], set(r['strategies'])) for r in self.game.roles)

        # Set initial subgames
        #
        # TODO Initial subgames might not want to include all pure profiles
        self.subgames = [subgame(rs) for rs in itertools.product(
            *([(r, {s}) for s in ss] for r, ss in self.full_game.iteritems()))]

        # Set useful variables
        self.obs_count = self.scheduler.default_observation_requirement
        self.max_profiles = max_profiles
        self.sleep_time = sleep_time

    def quiesce(self):
        """Starts the process of quiescing

        No writes happen until this method is called

        """
        # Initially schedule all pure profiles and all deviations
        self.schedule_profiles(itertools.chain(
            # pure profiles
            itertools.chain.from_iterable(
                sg.get_subgame_profiles(self.role_counts) for sg in self.subgames),
            # deviations
            itertools.chain.from_iterable(
                sg.get_deviation_profiles(self.full_game, self.role_counts)
                for sg in self.subgames)))
                               

    def schedule_profiles(self, profiles):
        """Schedules an interable of profiles

        Makes sure not to exceed max_profiles, and to only query the state of
        the simulator every sleep_time seconds when blocking on simulation
        execution

        """
        # Number of running profiles
        # 
        # This is an overestimate because checking is expensive
        count = self.scheduler.num_running_profiles()
        for prof in profiles:
            # First, we check / block until we can schedule another profile

            # Sometimes scheduled profiles already exist, and so even though we
            # increment our count, the global count doesn't increase. This
            # stops up from waiting a long time if we hit this threshold by
            # accident
            if count >= self.max_profiles:
                count = self.scheduler.num_running_profiles()
            # Wait until we can schedule more profiles
            while count >= self.max_profiles:
                time.sleep(self.sleep_time)
                count = self.scheduler.num_running_profiles()
                
            count += 1
            self.api.add_profile(self.scheduler.id, prof, self.obs_count)

        # Like before, but we block until everything is finished
        if count > 0:
            count = self.scheduler.num_running_profiles()
        while count > 0:
            time.sleep(self.sleep_time)
            count = self.scheduler.num_running_profiles()


if __name__ == '__main__':
    args = parser.parse_args()

    quies = quieser(
        int(args.scheduler) if args.scheduler.isdigit() else args.scheduler,
        args.auth,
        args.game and (int(args.game) if args.game.isdigit() else args.game),
        args.max_profiles,
        args.sleep_time)

    quies.quiesce()
    print '[[ done ]]'
