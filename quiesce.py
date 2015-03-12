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
import reduction

PARSER = argparse.ArgumentParser(description='Quiesce a generic scheduler on egtaonline.')
PARSER.add_argument('-g', '--game', metavar='game-id', type=int, required=True,
                    help='The id of the game to pull data from / to quiesce')
PARSER.add_argument('-a', '--auth', metavar='auth_token', required=True,
                    help='An authorization token to allow access to egtaonline.')
# Ideally this will just create a new scheduler and won't require and argument
PARSER.add_argument('-s', '--scheduler', metavar='generic-scheduler-id', type=int,
                    default=None,
                    help='''The id of the generic scheduler to quiesce. If not provided,
                    this will attempt to find a matching generic scheduler''')
PARSER.add_argument('-p', '--max-profiles', metavar='max-num-profiles', type=int,
                    default=10000, help='''Maximum number of profiles to ever have
                    scheduled at a time. Defaults to 10000.''')
PARSER.add_argument('-t', '--sleep-time', metavar='sleep-time', type=int, default=600,
                    help='''Time to wait in seconds between checking egtaonline for
                    job completion. Defaults to 300 (5 minutes).''')
PARSER.add_argument('-m', '--max-subgame-size', metavar='max-subgame-size', type=int,
                    default=3,
                    help='Maximum subgame size to require exploration. Defaults to 3')
PARSER.add_argument('-n', '--num-subgames', metavar='num-subgames', type=int,
                    default=1,
                    help='Maximum subgame size to require exploration. Defaults to 3')
PARSER.add_argument('--dpr', nargs="+", metavar="role-or-count", default=(),
                    help='''If specified, does a dpr reduction with role strategy counts.
                    e.g. --dpr role1 1 role2 2 ...''')
PARSER.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

# These are methods to measure the size of a game
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
    def __init__(self, game, auth_token, scheduler=None, max_profiles=10000,
                 sleep_time=300, subgame_limit=None, num_subgames=1, dpr=None,
                 verbosity=0):
        # pylint: disable=too-many-arguments

        # Get api and access to standard objects
        self.api = egta.egtaonline(auth_token)
        self.game = self.api.get_game(game, granularity='summary')
        if not scheduler:
            # If not specified we need to find the scheduler id. First find the
            # games simulator instance id, and then match against all generic
            # schedulers
            sim_inst_id = self.api.get_game(self.game.id).simulator_instance_id
            scheduler = analysis.only(gs for gs in self.api.get_generic_schedulers()
                                      if gs.simulator_instance_id == sim_inst_id).id
        self.scheduler = self.api.get_scheduler(scheduler, verbose=True)
        self.simulator = self.api.get_simulator(self.scheduler.simulator_id)

        # Set other game information
        self.role_counts = {r['name']: r['count'] for r in self.game.roles}
        self.full_game = analysis.subgame(
            (r['name'], set(r['strategies'])) for r in self.game.roles)

        # Set up reduction
        if dpr:
            self.reduction = reduction.dpr_reduction(self.role_counts, dpr)
            self.role_counts = dpr
        else:
            self.reduction = reduction.no_reduction()

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

        # Logging
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(40 - verbosity * 10)
        self.log.addHandler(logging.StreamHandler(sys.stderr))

        # Set useful quiesing variables
        self.obs_count = self.scheduler.default_observation_requirement
        self.max_profiles = max_profiles
        self.sleep_time = sleep_time
        self.subgame_limit = subgame_limit
        self.subgame_size = sum_strategies # TODO allow other functions
        self.num_subgames = num_subgames

    def quiesce(self):
        """Starts the process of quiescing

        No writes happen until this method is called

        """

        # TODO could be changed to allow multiple stopping conditions
        # TODO could be changed to be parallel
        while not self.confirmed_equilibria or self.necessary:
            # Get next subgames to explore
            subgames = self.get_next_subgames()
            self.log.debug(">>> Exploring subgames:\t%s", subgames)

            # Schedule subgames
            self.schedule_profiles(itertools.chain.from_iterable(
                sg.get_subgame_profiles(self.role_counts) for sg in subgames))
            game_data = self.get_data()

            # Find equilibria in the subgame
            equilibria = list(itertools.chain.from_iterable(
                game_data.equilibria(eq_subgame=subgame) for subgame in subgames))
            self.log.debug("Found equilibria:\t%s", equilibria)

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

    def get_next_subgames(self):
        """Gets a list of subgames to explore next"""
        subgames = []
        # This loop essentially says keep dequing subgames as long as you
        # haven't exceeded the threshold and either there's more necessary
        # subgames, or there are more backup subgames, you've scheduled no
        # subgames currently, and you still haven't found an equilibrium
        while (len(subgames) < self.num_subgames and (
                self.necessary or (
                    not subgames and self.backup and not self.confirmed_equilibria))):
            _, subgame = self.necessary.pop() if self.necessary else self.backup.pop()
            if not self.explored.add(subgame):  # already explored
                self.log.debug("--- Already Explored Subgame:\t%s", subgame)
            else:
                subgames.append(subgame)
        return subgames

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

        # Iterate through full game profiles
        for prof in itertools.chain.from_iterable(
                self.reduction.expand_profile(p) for p in profiles):
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
        return analysis.game_data(self.reduction.reduce_game_data(
            self.api.get_game(self.game.id, 'summary')))


def parse_dpr(dpr_list):
    """Turn list of role counts into dictionary"""
    return {dpr_list[2*i]: int(dpr_list[2*i+1]) for i in xrange(len(dpr_list)//2)}

def main():
    """Main function, declared so it doesn't have global scope"""
    args = PARSER.parse_args()

    quies = quieser(
        game=args.game,
        auth_token=args.auth,
        scheduler=args.scheduler,
        max_profiles=args.max_profiles,
        sleep_time=args.sleep_time,
        subgame_limit=args.max_subgame_size,
        num_subgames=args.num_subgames,
        dpr=parse_dpr(args.dpr),
        verbosity=args.verbose)

    quies.quiesce()


if __name__ == '__main__':
    main()
