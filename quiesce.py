#!/usr/bin/env python
'''Python script for quiessing a game'''
# pylint: disable=relative-import

import argparse
import time
import itertools
import logging
import sys
import collections
import json

import egtaonlineapi as egta
import analysis
import containers
import reduction
import utils

PARSER = argparse.ArgumentParser(description='Quiesce a generic scheduler on egtaonline.')
PARSER.add_argument('-g', '--game', metavar='game-id', type=int, required=True,
                    help='The id of the game to pull data from / to quiesce')
PARSER.add_argument('-a', '--auth', metavar='auth-token', required=True,
                    help='An authorization token to allow access to egtaonline.')
PARSER.add_argument('-p', '--max-profiles', metavar='max-num-profiles', type=int,
                    default=1000, help='''Maximum number of profiles to ever have
                    scheduled at a time. Defaults to 1000.''')
PARSER.add_argument('-t', '--sleep-time', metavar='sleep-time', type=int, default=600,
                    help='''Time to wait in seconds between checking egtaonline for
                    job completion. Defaults to 300 (5 minutes).''')
PARSER.add_argument('-m', '--max-subgame-size', metavar='max-subgame-size', type=int,
                    default=3,
                    help='Maximum subgame size to require exploration. Defaults to 3')
PARSER.add_argument('-n', '--num-subgames', metavar='num-subgames', type=int,
                    default=1,
                    help='''Maximum number of subgames to explore simultaneously.  Defaults to 1''')
PARSER.add_argument('--dpr', nargs='+', metavar='role-or-count', default=(),
                    help='''If specified, does a dpr reduction with role strategy counts.  e.g.
                    --dpr role1 1 role2 2 ...''')
PARSER.add_argument('-v', '--verbose', action='count', default=0,
                    help='''Verbosity level. Two for confirmed equilibria,
                    three for everything. Logging is output to standard
                    error''')
SCHED_GROUP = PARSER.add_argument_group(
    'Scheduler parameters', description='Parameters for the scheduler')
SCHED_GROUP.add_argument('-y', '--memory', metavar='process-memory', type=int,
                         default=4096, help='''The process memory to schedule
                         jobs with in MB. Defaults to 4096''')
SCHED_GROUP.add_argument('-o', '--observation-time', metavar='observation-time', type=int,
                         default=600,
                         help='The time to allow for each observation in seconds.  Defaults to 600')
SCHED_GROUP.add_argument('--obs-per-sim', metavar='observations-per-simulation', type=int,
                         default=10,
                         help='The number of observations to run per simulation. Defaults to 10')
SCHED_GROUP.add_argument('--default-obs-req', metavar='default-observation-requirement',
                         type=int, default=10,
                         help='The default observation requirement. Defaults to 10')
SCHED_GROUP.add_argument('--nodes', metavar='nodes', type=int, default=1,
                         help='Number of nodes to run the simulation on. Defaults to 1')

# These are methods to measure the size of a game
def max_strategies(subgame, **_):
    '''Max number of strategies per role in subgame'''
    return max(len(strats) for strats in subgame.values())

def sum_strategies(subgame, **_):
    '''Sum of all strategies in each role in subgame'''
    return sum(len(strats) for strats in subgame.values())

def num_profiles(subgame, role_counts, **_):
    '''Returns the number of profiles in a subgame'''
    return utils.prod(utils.game_size(role_counts[role], len(strats))
                      for role, strats in subgame.iteritems())

# For output of general objects
def _json_default(obj):
    '''Function for json default kwargs'''
    if isinstance(obj, collections.Mapping):
        return dict(obj)
    elif isinstance(obj, collections.Iterable):
        return list(obj)
    raise TypeError('Can\'t serialize %s of type %s' % (obj, type(obj)))

def _to_json_str(obj):
    '''Converts general objects into nice output string'''
    return json.dumps(obj, indent=2, default=_json_default)


# Main class
class quieser(object):
    '''Class to manage quiesing of a scheduler'''
    # pylint: disable=too-many-instance-attributes

    def __init__(self, game, auth_token, max_profiles=10000,
                 sleep_time=300, subgame_limit=None, num_subgames=1, dpr=None,
                 scheduler_options=containers.frozendict(), verbosity=0):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        # Logging
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(40 - verbosity * 10)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self.log.addHandler(handler)

        # Get api and access to standard objects
        self.api = egta.egtaonline(auth_token)
        self.game = self.api.get_game(game, granularity='summary')
        self.scheduler = self._create_scheduler(**scheduler_options)
        self.scheduler = self.api.get_scheduler(self.scheduler['id'],
                                                verbose=True)
        self.scheduler.update(active=1) # Make scheduler active
        self.simulator = self.api.get_simulator(self.scheduler['simulator_id'])

        # Set other game information
        self.role_counts = {r['name']: r['count'] for r in self.game['roles']}
        self.full_game = analysis.subgame(
            (r['name'], set(r['strategies'])) for r in self.game['roles'])

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
            (0, subg) for subg in self.full_game.pure_subgames())
        # Subgames to try only if no equilibria have been found. Priority is a
        # tuple first indicating if it was a best response then indicating the
        # regret
        self.backup = containers.priorityqueue()
        # Subgames we've already explored
        self.explored = analysis.subgame_set()
        self.confirmed_equilibria = set()

        # Set useful quiesing variables
        self.obs_count = self.scheduler['default_observation_requirement']
        self.max_profiles = max_profiles
        self.sleep_time = sleep_time
        self.subgame_limit = subgame_limit
        self.subgame_size = sum_strategies # TODO allow other functions
        self.num_subgames = num_subgames

    def _create_scheduler(self, process_memory=4096,
                          observation_time=600, obs_req=10, obs_per_sim=10,
                          nodes=1):
        '''Creates a generic scheduler with the appropriate parameters'''
        # pylint: disable=too-many-arguments

        # Find simulator by matching on fullname
        sim_id = utils.only(s for s in self.api.get_simulators()
                            if '%s-%s' % (s['name'], s['version']) == \
                            self.game['simulator_fullname'])['id']
        # Generate a random name
        name = '%s_generic_quiesce_%s' % (self.game['name'],
                                          utils.random_string(6))
        size = self.api.get_game(self.game['id'], 'structure')['size']
        sched = self.api.create_generic_scheduler(
            simulator_id=sim_id,
            name=name,
            active=1,
            process_memory=process_memory,
            size=size,
            time_per_observation=observation_time,
            observations_per_simulation=obs_per_sim,
            nodes=nodes,
            default_observation_requirement=obs_req,
            configuration=dict(self.game['configuration']))

        # Add roles and counts to scheduler
        for role in self.game['roles']:
            sched.add_role(role['name'], role['count'])

        self.log.debug('Created scheduler %d', sched['id'])
        return sched

    def quiesce(self):
        '''Starts the process of quiescing

        No writes happen until this method is called

        '''

        # TODO could be changed to allow multiple stopping conditions
        while not self.confirmed_equilibria or self.necessary:
            # Get next subgames to explore
            subgames = self.get_next_subgames()
            self.log.debug('>>> Exploring subgames <<<\n%s',
                           _to_json_str(subgames))

            # Schedule subgames
            self.schedule_profiles(itertools.chain.from_iterable(
                sg.subgame_profiles(self.role_counts) for sg in subgames))
            game_data = self.get_data()
            self.log.debug('Finished exploring subgames')

            # Find equilibria in the subgame
            equilibria = list(itertools.chain.from_iterable(
                game_data.equilibria(eq_subgame=subgame)
                for subgame in subgames))
            self.log.debug('Found candidate equilibria\n%s',
                           _to_json_str(equilibria))

            # Schedule all deviations from found equilibria
            self.schedule_profiles(itertools.chain.from_iterable(
                eq.support().deviation_profiles(self.full_game, self.role_counts)
                for eq in equilibria))
            game_data = self.get_data()
            self.log.debug('Finished exploring equilibria deviations')

            # Confirm equilibria and add beneficial deviating subgames to
            # future exploration
            for equilibrium in equilibria:
                self.queue_deviations(equilibrium, game_data)

        self.log.info('Finished quiescing\nConfirmed equilibria:\n%s',
                      _to_json_str(self.confirmed_equilibria))

    def get_next_subgames(self):
        '''Gets a list of subgames to explore next'''
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
                self.log.debug('--- Already explored subgame ---\n%s',
                               _to_json_str(subgame))
            else:
                subgames.append(subgame)
        return subgames

    def queue_deviations(self, equilibrium, game_data):
        '''Queues deviations to an equilibrium'''
        responses = game_data.responses(equilibrium)
        self.log.debug('Responses\n%s', _to_json_str(responses))
        if not responses:  # Found equilibrium
            self.confirmed_equilibria.add(equilibrium)
            self.log.info('!!! Confirmed equilibrium !!!\n%s',
                          _to_json_str(equilibrium))
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
        '''Schedules an interable of profiles

        Makes sure not to exceed max_profiles, and to only query the state of
        the simulator every sleep_time seconds when blocking on simulation
        execution

        '''
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
            profile_ids.add(self.api.add_profile(self.scheduler.scheduler_id, prof,
                                                 self.obs_count))

        # Check that all scheduled profiles are finished executing
        active_profiles = self.scheduler.are_profiles_still_active(profile_ids)
        while active_profiles:
            time.sleep(self.sleep_time)
            active_profiles = self.scheduler.are_profiles_still_active(profile_ids)

    def get_data(self):
        '''Gets current game data'''
        return analysis.game_data(self.reduction.reduce_game_data(
            self.api.get_game(self.game['id'], 'summary')))

    def delete_scheduler(self):
        '''Deletes the scheduler'''
        self.scheduler.delete()

def parse_dpr(dpr_list):
    '''Turn list of role counts into dictionary'''
    return {dpr_list[2*i]: int(dpr_list[2*i+1]) for i in xrange(len(dpr_list)//2)}


def main():
    '''Main function, declared so it doesn't have global scope'''
    args = PARSER.parse_args()

    quies = quieser(
        game=args.game,
        auth_token=args.auth,
        max_profiles=args.max_profiles,
        sleep_time=args.sleep_time,
        subgame_limit=args.max_subgame_size,
        num_subgames=args.num_subgames,
        dpr=parse_dpr(args.dpr),
        scheduler_options={
            'process_memory': args.memory,
            'observation_time': args.observation_time,
            'obs_per_sim': args.obs_per_sim,
            'obs_req': args.default_obs_req,
            'nodes': args.nodes
        },
        verbosity=args.verbose)

    try:
        quies.quiesce()
    finally:
        quies.log.debug('Deleting scheduler %d', quies.scheduler['id'])
        quies.delete_scheduler()


if __name__ == '__main__':
    main()
