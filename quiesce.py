#!/usr/bin/env python
'''Python script for quiessing a game'''
# pylint: disable=relative-import

import argparse
import time
import itertools
import logging
from logging import handlers
import sys
import collections
import json
import traceback

import egtaonlineapi as egta
import analysis
import containers
import reduction
import utils
import smtplib

PARSER = argparse.ArgumentParser(description='''Quiesce a generic scheduler on
egtaonline.''')
PARSER.add_argument('-g', '--game', metavar='game-id', type=int, required=True,
                    help='''The id of the game to pull data from / to
                    quiesce''')
PARSER.add_argument('-a', '--auth', metavar='auth-token', required=True,
                    help='''An authorization token to allow access to
                    egtaonline.''')
PARSER.add_argument('-p', '--max-profiles', metavar='max-num-profiles',
                    type=int, default=500, help='''Maximum number of profiles
                    to ever have scheduled at a time. Defaults to 500.''')
PARSER.add_argument('-t', '--sleep-time', metavar='sleep-time', type=int,
                    default=300, help='''Time to wait in seconds between
                    checking egtaonline for job completion. Defaults to 300 (5
                    minutes).''')
PARSER.add_argument('-m', '--max-subgame-size', metavar='max-subgame-size',
                    type=int, default=3, help='''Maximum subgame size to
                    require exploration. Defaults to 3''')
PARSER.add_argument('--dpr', nargs='+', metavar='role-or-count', default=(),
                    help='''If specified, does a dpr reduction with role
                    strategy counts.  e.g.  --dpr role1 1 role2 2 ...''')
PARSER.add_argument('-v', '--verbose', action='count', default=0,
                    help='''Verbosity level. Two for confirmed equilibria,
                    three for everything. Logging is output to standard
                    error''')
PARSER.add_argument('-e', '--email_verbosity', action='count', default=0,
                    help='''Verbosity level for email. Two for confirmed
                    equilibria, three for everything''')
PARSER.add_argument('-r', '--recipient', action='append', default=[],
                    help='''Specify an email address to receive email logs
                    at. Can specify multiple email addresses.''')


SCHED_GROUP = PARSER.add_argument_group('Scheduler parameters',
                                        description='''Parameters for the
                                        scheduler.''')
SCHED_GROUP.add_argument('-y', '--memory', metavar='process-memory', type=int,
                         default=4096, help='''The process memory to schedule
                         jobs with in MB. Defaults to 4096''')
SCHED_GROUP.add_argument('-o', '--observation-time',
                         metavar='observation-time', type=int, default=600,
                         help='''The time to allow for each observation in
                         seconds.  Defaults to 600''')
SCHED_GROUP.add_argument('--obs-per-sim',
                         metavar='observations-per-simulation', type=int,
                         default=10, help='''The number of observations to run
                         per simulation. Defaults to 10''')
SCHED_GROUP.add_argument('--default-obs-req',
                         metavar='default-observation-requirement', type=int,
                         default=10, help='''The default observation
                         requirement. Defaults to 10''')
SCHED_GROUP.add_argument('--nodes', metavar='nodes', type=int, default=1,
                         help='''Number of nodes to run the simulation
                         on. Defaults to 1''')


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


# For output of general objects e.g. logging
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


def _get_logger(name, level, email_level, recipients, game_id):
    '''Returns an appropriate logger'''
    log = logging.getLogger(name)
    log.setLevel(40 - level * 10)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        '%%(asctime)s (%d) %%(message)s' % game_id))
    log.addHandler(handler)

    # Email Logging
    if recipients:
        email_subject = "EGTA Online Quiesce Status for Game %d" % game_id
        smtp_host = "localhost"

        # XXX - we need to do this to match the from address to the local host name
        # XXX - otherwise, email logging will not work
        server = smtplib.SMTP(smtp_host) # must get correct hostname to send mail
        smtp_fromaddr = "EGTA Online <egta_online@" + server.local_hostname + ">"
        server.quit() # dummy server is now useless

        email_handler = handlers.SMTPHandler(smtp_host, smtp_fromaddr,
                                             recipients, email_subject)
        email_handler.setLevel(40 - email_level * 10)
        log.addHandler(email_handler)

    return log


# Main class
class quieser(object):
    '''Class to manage quiesing of a scheduler'''
    # pylint: disable=too-many-instance-attributes

    def __init__(self, game_id, auth_token, max_profiles=10000,
                 sleep_time=300, subgame_limit=None, num_subgames=1, dpr=None,
                 scheduler_options=containers.frozendict(), verbosity=0,
                 email_verbosity=0, recipients=[]):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        # Get api and access to standard objects
        self._log = _get_logger(self.__class__.__name__, verbosity,
                                email_verbosity, recipients, game_id)
        self._api = egta.egtaonline(auth_token)
        self._game_id = game_id
        game = self._api.get_game(game_id, granularity='summary')

        # Set other game information
        self._role_counts = {r['name']: r['count'] for r in game['roles']}
        self._full_game = analysis.subgame(
            (r['name'], set(r['strategies'])) for r in game['roles'])

        # Set up scheduler
        self._scheduler = self._create_scheduler(game, **scheduler_options)

        # Set up reduction
        if dpr:
            self._reduction = reduction.dpr_reduction(self._role_counts, dpr)
            self._role_counts = dpr
        else:
            self._reduction = reduction.no_reduction()

        # Set useful quiesing variables
        self._obs_count = self._scheduler['default_observation_requirement']
        self._max_profiles = max_profiles
        self._sleep_time = sleep_time
        self._subgame_limit = subgame_limit
        self._subgame_size = sum_strategies  # TODO allow other functions

    def _create_scheduler(self, game, process_memory=4096,
                          observation_time=600, obs_req=10, obs_per_sim=10,
                          nodes=1):
        '''Creates a generic scheduler with the appropriate parameters'''
        # pylint: disable=too-many-arguments

        # Is to check that the number of players in each role match, it's a
        # hacky solution, but egta doesn't expose the necessary information.
        candidate_name = '%s_generic_quiesce_%s' % (
            game['name'],
            '_'.join('%s_%d' % rc for rc in self._role_counts.iteritems()))
        sim_inst_id = sim_inst_id = self._api.get_game(
            self._game_id)['simulator_instance_id']
        schedulers = [gs for gs in self._api.get_generic_schedulers() if
                      gs['simulator_instance_id'] == sim_inst_id and
                      gs['process_memory'] == process_memory and
                      gs['time_per_observation'] == observation_time and
                      gs['default_observation_requirement'] == obs_req and
                      gs['observations_per_simulation'] == obs_per_sim and
                      gs['nodes'] == nodes and
                      gs['name'].startswith(candidate_name)]

        if len(schedulers) > 0:
            # found at least one exact match so use it
            sched = schedulers[0]
            sched.update(active=1)
            self._log.info('Using scheduler %d ' +
                           '(http://egtaonline.eecs.umich.edu/' +
                           'generic_schedulers/%d)',
                           sched['id'], sched['id'])
            return sched

        # Find simulator by matching on fullname
        sim_id = utils.only(s for s in self._api.get_simulators() if '%s-%s' %
                            (s['name'], s['version']) ==
                            game['simulator_fullname'])['id']
        # Generate a random name
        name = '%s_%s' % (candidate_name, utils.random_string(6))
        size = self._api.get_game(self._game_id, 'structure')['size']
        sched = self._api.create_generic_scheduler(
            simulator_id=sim_id,
            name=name,
            active=1,
            process_memory=process_memory,
            size=size,
            time_per_observation=observation_time,
            observations_per_simulation=obs_per_sim,
            nodes=nodes,
            default_observation_requirement=obs_req,
            configuration=dict(game['configuration']))

        # Add roles and counts to scheduler
        for role in game['roles']:
            sched.add_role(role['name'], role['count'])

        self._log.info('Created scheduler %d ' +
                       '(http://egtaonline.eecs.umich.edu/' +
                       'generic_schedulers/%d)',
                       sched['id'], sched['id'])
        return sched

    def quiesce(self):
        '''Starts the process of quiescing

        No writes happen until this method is called

        '''

        # This handles scheduling all of the profiles
        sched = profile_scheduler(self._log, self._scheduler, self._reduction,
                                  self._max_profiles, self._full_game,
                                  self._role_counts, self._obs_count,
                                  self._full_game.pure_subgames())

        pending = []
        # Instead of using a set here, using an equilibrium set
        # (e.g. equilibria that are almost the same are counted arbitrarily as
        # exactly the same should probably be used)
        confirmed_equilibria = set()
        backup = containers.priorityqueue()

        while not confirmed_equilibria or sched.not_done() or pending:
            # Schedule as many as we can, and update pending list
            pending.extend(sched.schedule_more())

            # See what's finished
            try:
                game_data = self._get_game()
            except Exception, e:
                # Sometimes getting game data fails. Just wait and try again
                self._log.debug('Encountered error getting game data: (%s) %s\n',
                                e.__class__.__name__, e)
                time.sleep(self._sleep_time)
                continue
            running = self._scheduler.running_profiles()

            def check((item, ids)):
                if ids.intersection(running):
                    return True
                # Item is complete
                if isinstance(item, analysis.subgame):  # Subgame
                    self._analyze_subgame(game_data, item, sched)
                else:  # Equilibria
                    self._analyze_equilibrium(game_data, item,
                                              confirmed_equilibria,
                                              sched, backup)
                # This item was processed, so remove from pending
                return False

            # Process and complete pending jobs and remove from pending
            pending = filter(check, pending)

            if pending:
                # We're still waiting for jobs to complete, so take a break
                #
                # Note, this isn't the best way to do this. In principle we
                # should just check if any were immediately resolved, in which
                # case we try to schedule again. The current implementation has
                # slightly worse performance when requiescing something.
                time.sleep(self._sleep_time)
            elif not confirmed_equilibria and not sched.not_done():
                # We've finished all the required stuff, but still haven't
                # found an equilibrium, so pop a backup off
                sched.append(backup.pop()[1])

        self._log.info('Finished quiescing\nConfirmed equilibria:\n%s',
                       _to_json_str(confirmed_equilibria))

    def _get_game(self):
        '''Get the game data'''
        return analysis.game_data(self._reduction.reduce_game_data(
            self._api.get_game(self._game_id, 'summary')))

    def _analyze_subgame(self, game_data, subgame, sched):
        '''Computes subgame equilibrium and queues them to be scheduled'''
        equilibria = list(game_data.equilibria(eq_subgame=subgame))
        self._log.debug('Found candidate equilibria:\n%s\nin subgame:\n%s\n',
                        _to_json_str(equilibria), _to_json_str(subgame))
        if not equilibria:
            self._log.info('Found no equilibria in subgame:\n%s\n',
                           _to_json_str(subgame))
        sched.extend(equilibria)

    def _analyze_equilibrium(self, game_data, equilibrium,
                             confirmed_equilibria, sched, backup):
        '''Analyzes responses to an equilibrium and book keeps accordingly'''
        responses = game_data.responses(equilibrium)
        self._log.debug('Responses:\n%s\nto candidate equilibrium:\n%s\n',
                        _to_json_str(responses),
                        _to_json_str(equilibrium))

        if not responses:  # Found equilibrium
            if equilibrium not in confirmed_equilibria:
                confirmed_equilibria.add(equilibrium)
                self._log.info('Confirmed equilibrium:\n%s\n',
                               _to_json_str(equilibrium))

        else:  # Queue up next subgames
            supp = equilibrium.support()
            # If it's a large subgame, best responses should not be necessary
            large_subgame = self._subgame_size(supp, role_counts=self._role_counts) \
                >= self._subgame_limit

            for role, rresps in responses.iteritems():
                ordered = sorted(rresps.iteritems(), key=lambda x: -x[1])
                strat, gain = ordered[0]  # best response
                if large_subgame:
                    # Large, so add to backup with priority 0 (highest)
                    backup.append(
                        ((0, -gain), supp.with_deviation(role, strat)))
                else:
                    # Best response becomes necessary to explore
                    sched.append(supp.with_deviation(role, strat))
                # All others become backups if we run out without
                # finding one These all have priority 1 (lowest)
                for strat, gain in ordered[1:]:
                    backup.append(
                        ((1, -gain), supp.with_deviation(role, strat)))

    def delete_scheduler(self):
        '''Deletes the scheduler'''
        self._scheduler.delete()

    def deactivate(self):
        '''Deactivates the scheduler'''
        self._scheduler.update(active=0)


class profile_scheduler(object):
    '''Class that handles scheduling profiles'''
    def __init__(self, log, scheduler, reduction, max_profiles, full_game,
                 role_counts, obs_count, init_items=()):
        self._scheduler = scheduler
        self._log = log

        self._max_profiles = max_profiles
        self._full_game = full_game
        self._role_counts = role_counts
        self._reduction = reduction
        self._obs_count = obs_count

        self._item = None  # None if there are no more profiles
        self._profs = None
        self._profile_ids = None

        self._necessary = list(self._full_game.pure_subgames())
        self._explored_subgames = analysis.subgame_set()

    def not_done(self):
        '''Returns True if there is nothing else this can schedule'''
        return self._necessary or self._item

    def _set_active_item(self):
        '''Sets item and profs if not defined and possible'''
        while not self._item and self._necessary:
            self._item = self._necessary.pop()
            self._profile_ids = set()

            if isinstance(self._item, analysis.subgame):  # Subgame
                if not self._explored_subgames.add(self._item):
                    # already explored
                    self._log.debug('Already explored subgame:\n%s\n',
                                    _to_json_str(self._item))
                    self._item = None
                    continue
                self._log.debug('Exploring subgame:\n%s\n',
                                _to_json_str(self._item))
                self._profs = self._item.subgame_profiles(self._role_counts)

            else:  # Equilibria
                self._log.debug('Exploring equilibrium deviations:\n%s\n',
                                _to_json_str(self._item))
                self._profs = self._item.support().deviation_profiles(
                    self._full_game, self._role_counts)

            # Un-reduce profiles
            self._profs = itertools.chain.from_iterable(
                self._reduction.expand_profile(p) for p in self._profs)

    def schedule_more(self):
        '''Schedules as many profiles as possible

        Returns a generator of item, profile id set pairs for every item that
        became fully scheduled.

        '''
        count = self._scheduler.num_running_profiles()

        # Loop over necessary that we can schedule
        while count < self._max_profiles and (self._item or self._necessary):
            self._set_active_item()

            # Try to schedule profiles in last "item"
            while count < self._max_profiles and self._item:
                try:
                    # Schedule more profiles
                    for _ in xrange(count, self._max_profiles):
                        prof_id = self._scheduler.add_profile(
                            next(self._profs),
                            self._obs_count)
                        self._profile_ids.add(prof_id)
                except StopIteration:
                    # This set is gone, get the next task yo schedule
                    yield self._item, self._profile_ids
                    self._item = None

                # Update count
                count = self._scheduler.num_running_profiles()

    def append(self, item):
        '''Add one more item to get scheduled'''
        self._necessary.append(item)

    def extend(self, items):
        '''Add an iterator of items to get scheduled'''
        self._necessary.extend(items)


def _parse_dpr(dpr_list):
    '''Turn list of role counts into dictionary'''
    return {dpr_list[2*i]: int(dpr_list[2*i+1])
            for i in xrange(len(dpr_list)//2)}


def main():
    '''Main function, declared so it doesn't have global scope'''
    args = PARSER.parse_args()

    quies = quieser(
        game_id=args.game,
        auth_token=args.auth,
        max_profiles=args.max_profiles,
        sleep_time=args.sleep_time,
        subgame_limit=args.max_subgame_size,
        dpr=_parse_dpr(args.dpr),
        scheduler_options={
            'process_memory': args.memory,
            'observation_time': args.observation_time,
            'obs_per_sim': args.obs_per_sim,
            'obs_req': args.default_obs_req,
            'nodes': args.nodes
        },
        verbosity=args.verbose,
        email_verbosity=args.email_verbosity,
        recipients=args.recipient)

    try:
        quies.quiesce()
    except Exception, e:
        quies._log.error('Caught exception: (%s) %s\nWith traceback:\n%s\n',
                         e.__class__.__name__, e, traceback.format_exc())
    finally:
        quies.deactivate()


if __name__ == '__main__':
    main()
