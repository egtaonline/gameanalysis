#!/usr/bin/env python
"""Python script for quiessing a game"""
# pylint: disable=relative-import

import argparse
import time
import itertools
import logging
import sys

import BasicFunctions as funcs

import egtaonlineapi as egta
import analysis
import containers

PARSER = argparse.ArgumentParser(description='Quiesce a generic scheduler on egtaonline.')
PARSER.add_argument('-a', '--auth', metavar='auth_token', required=True,
                    help='An authorization token to allow access to egtaonline.')
PARSER.add_argument('-s', '--scheduler', metavar='scheduler', required=True,
                    help='The name or id of the scheduler to quiesce.')
PARSER.add_argument('-g', '--game', metavar='game_id', type=int, default=None,
                    help='''The id of the game used to indicate how to schedule.
                    If not provided, this will try and determine the game, but may fail.
                    ''')
PARSER.add_argument('-m', '--max-profiles', metavar='num-profiles', type=int,
                    default=10000, help='''Maximum number of profiles to ever have
                    scheduled at a time. Defaults to 10000.''')
PARSER.add_argument('-t', '--sleep-time', metavar='delta', type=int, default=600,
                    help='''Time to wait in seconds between checking egtaonline for
                    job completion. Defaults to 300.''')
PARSER.add_argument('-n', '--max-subgame-size', metavar='n', type=int,
                    default=3,
                    help='Maximum subgame size to require exploration. Defaults to 3')
PARSER.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

def max_strategies(subgame, **_):
    """Max number of strategies per role in subgame"""
    return max(len(strats) for strats in subgame.values())

def sum_strategies(subgame, **_):
    """Sum of all strategies in each role in subgame"""
    return sum(len(strats) for strats in subgame.values())

def num_profiles(subgame, role_counts, **_):
    """Returns the number of profiles in a subgame"""
    return funcs.prod(funcs.game_size(role_counts[role], len(strats))
                      for role, strats in subgame.iteritems())

class quieser(object):
    """Class to manage quiesing of a scheduler"""
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=star-args

    # TODO keep track up most recent set of data, and have schedule update it
    # after it finishes blocking
    def __init__(self, scheduler, auth_token, game=None, max_profiles=10000,
                 sleep_time=300, subgame_limit=None, verbosity=0):
        # pylint: disable=too-many-arguments
        # Get api and access to standard objects
        self.api = egta.egtaonline(auth_token)
        self.scheduler = self.api.get_scheduler(scheduler, verbose=True)
        self.simulator = self.api.get_simulator(self.scheduler.simulator_id)
        sim_inst_id = self.api.get_scheduler(self.scheduler.id).simulator_instance_id
        if game is None:
            # Hack to find game with same simulator instance id
            game = analysis.only(g for g in self.api.get_games()
                                 if g.simulator_instance_id == sim_inst_id).id
            self.game = self.api.get_game(game, granularity='summary')

        # Set other game information
        self.role_counts = {r['name']: r['count'] for r in self.game.roles}
        self.full_game = analysis.subgame(
            (r['name'], set(r['strategies'])) for r in self.game.roles)

        # Set up progress containers
        # Set initial subgames: currently this is all pure profiles
        # The subgames to necessary explore to consider quiessed
        self.necessary = containers.priorityqueue(
            (0, analysis.subgame(rs)) for rs in itertools.product(
                *([(r, {s}) for s in ss] for r, ss in self.full_game.iteritems())))
        # Subgames to try only if no equilibria have been found. Priority is a
        # tuple first indicating if it was a best response then indicating the
        # regret
        self.backup = containers.priorityqueue()
        # Subgames we've already explored
        self.explored = analysis.subgame_set()
        self.confirmed_equilibria = set()

        # Set useful quiesing variables
        self.obs_count = self.scheduler.default_observation_requirement
        self.max_profiles = max_profiles
        self.sleep_time = sleep_time
        self.subgame_limit = subgame_limit
        self.subgame_size = sum_strategies # TODO allow other functions

        # Logging
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(40 - verbosity * 10)
        self.log.addHandler(logging.StreamHandler(sys.stderr))

    def quiesce(self):
        """Starts the process of quiescing

        No writes happen until this method is called

        """

        # TODO could be changed to allow multiple stopping conditions
        # TODO could be changed to be parallel
        while not self.confirmed_equilibria or self.necessary:
            # Get next subgame to explore
            _, subgame = self.necessary.pop() if self.necessary else self.backup.pop()
            self.log.debug(">>> Exploring subgame:\t%s", subgame)
            if not self.explored.add(subgame):  # already explored
                self.log.debug("--- Already Explored Subgame:\t%s", subgame)
                continue

            # Schedule subgame
            self.schedule_profiles(subgame.get_subgame_profiles(self.role_counts))
            game_data = self.get_data()

            # Find equilibria in the subgame
            equilibria = game_data.equilibria(eq_subgame=subgame)
            for equilibrium in equilibria:
                self.log.debug("Found equilibrium:\t%s", equilibrium)

            # Schedule all deviations from found equilibria
            self.schedule_profiles(itertools.chain.from_iterable(
                eq.support().get_deviation_profiles(self.full_game, self.role_counts)
                for eq in equilibria))
            game_data = self.get_data()

            # Confirm equilibria and add beneficial deviating subgames to
            # future exploration
            for equilibrium in equilibria:
                self.queue_deviations(equilibrium, game_data)

        self.log.info("Confirmed Equilibria:")
        for i, equilibrium in enumerate(self.confirmed_equilibria):
            self.log.info("%d:\t%s", (i + 1), equilibrium)

    def queue_deviations(self, equilibrium, game_data):
        """Queues deviations to an equilibrium"""
        responses = game_data.responses(equilibrium)
        self.log.debug("Responses:\t%s", responses)
        if not responses:  # Found equilibrium
            self.confirmed_equilibria.add(equilibrium)
            self.log.info("!!! Confirmed Equilibrium:\t%s", equilibrium)
        else: # Queue up next subgames
            supp = equilibrium.support()
            # If it's a large subgame, best responses should not be necessary
            large_subgame = self.subgame_size(supp, role_counts=self.role_counts) \
                            >= self.subgame_limit

            for role, rresps in responses.iteritems():
                ordered = sorted(rresps.iteritems(), key=lambda x: -x[1])
                strat, gain = ordered[0]  # best response
                if large_subgame:
                    # Large, so add to backup with priority 0 (highest)
                    self.backup.append(((0, -gain), supp.with_deviation(role, strat)))
                else:
                    # Best response becomes necessary to explore
                    self.necessary.append((-gain, supp.with_deviation(role, strat)))
                # All others become backups if we run out without finding one
                # These all have priority 1 (lowest)
                for strat, gain in ordered[1:]:
                    self.backup.append(((1, -gain), supp.with_deviation(role, strat)))

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
        profile_ids = set()
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
            profile_ids.add(self.api.add_profile(self.scheduler.id, prof, self.obs_count))

        # Check that all scheduled profiles are finished executing
        active_profiles = self.scheduler.are_profiles_still_active(profile_ids)
        while active_profiles:
            time.sleep(self.sleep_time)
            active_profiles = self.scheduler.are_profiles_still_active(profile_ids)

    def get_data(self):
        """Gets current game data"""
        return analysis.game_data(self.api.get_game(self.game.id, 'summary'))


def main():
    """Main function, declared so it doesn't have global scope"""
    args = PARSER.parse_args()

    quies = quieser(
        int(args.scheduler) if args.scheduler.isdigit() else args.scheduler,
        auth_token=args.auth,
        game=args.game and (int(args.game) if args.game.isdigit() else args.game),
        max_profiles=args.max_profiles,
        sleep_time=args.sleep_time,
        subgame_limit=args.max_subgame_size,
        verbosity=args.verbose)

    quies.quiesce()


if __name__ == '__main__':
    main()
