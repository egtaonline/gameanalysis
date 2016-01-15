'''Python package to handle python interface to egta online api'''
import requests
import json
import logging
import sys

# pylint: disable=relative-import
import analysis
import utils
import containers


def _encode_data(data):
    '''Takes data in nested dictionary form, and converts it for egta

    All dictionary keys must be strings. This call is non destructive.

    '''
    encoded = {}
    for k, val in data.iteritems():
        if isinstance(val, dict):
            for inner_key, inner_val in _encode_data(val).iteritems():
                encoded['%s[%s]' % (k, inner_key)] = inner_val
        else:
            encoded[k] = val
    return encoded


class scheduler(dict):
    '''Represents information about a scheduler'''

    def __init__(self, *args, **kwargs):
        api = kwargs.pop('_api')
        super(scheduler, self).__init__(*args, **kwargs)
        self._api = api
        self.scheduler_id = self['id']

    def num_running_profiles(self):
        '''Get the number of currently running profiles'''
        # Update info
        update = self._api.get_scheduler(self.scheduler_id, verbose=True)
        # Sum profiles that aren't complete
        return sum(req['current_count'] < req['requirement']
                   for req in update['scheduling_requirements'] or ())

    def running_profiles(self):
        '''Get a set of the active profile ids'''
        # Update info
        update = self._api.get_scheduler(self.scheduler_id, verbose=True)
        # Sum profiles that aren't complete
        return {req['profile_id']
                for req in update['scheduling_requirements'] or ()
                if req['current_count'] < req['requirement']}

    def are_profiles_still_active(self, profiles):
        '''Returns true if any of the profile ids in profiles are active'''
        profiles = set(profiles)
        # Update info
        update = self._api.get_scheduler(self.scheduler_id, verbose=True)
        for req in update['scheduling_requirements'] or ():
            if (req['profile_id'] in profiles
                    and req['current_count'] < req['requirement']):
                return True
        return False

    def add_profile(self, profile_desc, update=False):
        '''Add a profile to this simulator

        See egtaonline.remove_profile for more documentation

        '''
        return self._api.add_profile(self.scheduler_id, profile_desc, update)

    def remove_profile(self, profile_desc):
        '''Removes a specific profile from this simulator

        See egtaonline.remove_profile for more documentation

        '''
        self._api.remove_profile(self.scheduler_id, profile_desc)

    def remove_all_profiles(self):
        '''Remove all profiles from this scheduler'''
        self._api.remove_all_profiles(self.scheduler_id)

    def add_role(self, role, count):
        '''Add a role and count to this scheduler

        Only works if scheduler is a generic scheduler

        '''
        self._api.generic_scheduler_add_role(self.scheduler_id, role, count)

    def remove_role(self, role):
        '''Remove a role from this scheduler

        Only works if scheduler is a generic scheduler

        '''
        self._api.generic_scheduler_remove_role(self.scheduler_id, role)

    def update(self, **kwargs):
        '''Updates scheduler parameters.

        Only works if scheduler is a generic scheduler. See
        create_generic_scheduler for possible arguments for kwargs

        '''
        return self._api.update_generic_scheduler(self.scheduler_id, **kwargs)

    def delete(self):
        '''Delete generic scheduler

        This will fail if scheduler is not generic

        '''
        self._api.delete_generic_scheduler(self.scheduler_id)


class egtaonline(object):
    '''Class to wrap egtaonline api'''
    def __init__(self, auth_token, domain='egtaonline.eecs.umich.edu',
                 logLevel=0):
        self.domain = domain
        self.auth = auth_token
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(40 - logLevel * 10)
        self.log.addHandler(logging.StreamHandler(sys.stderr))

    def request(self, verb, api, data=containers.frozendict()):
        '''Convenience method for making requests'''
        true_data = {'auth_token': self.auth}
        true_data.update(data)
        true_data = _encode_data(true_data)
        url = 'https://%s/api/v3/%s' % (self.domain, api)
        self.log.info('%s request to %s with data %s', verb, url, true_data)
        return requests.request(verb, url, data=true_data)

    def get_generic_schedulers(self):
        '''Get a generator of all known generic schedulers'''
        resp = self.request('get', 'generic_schedulers')
        resp.raise_for_status()
        return (scheduler(x, _api=self) for x
                in json.loads(resp.text)['generic_schedulers'])

    def get_scheduler(self, scheduler_name, verbose=False):
        '''Get a scheduler by name or id

        If scheduler is an int, then this will get the scheduler by id (more
        efficient and works for all schedulers) if it is a string, then this
        will get the generic scheduler by name

        '''
        if isinstance(scheduler_name, int):
            data = {'granularity': 'with_requirements'} if verbose else {}
            resp = self.request('get', 'schedulers/%d.json' % scheduler_name,
                                data)
            resp.raise_for_status()
            return scheduler(json.loads(resp.text), _api=self)

        named = scheduler(utils.only(s for s in self.get_generic_schedulers()
                                     if s['name'] == scheduler_name))
        if verbose:
            return self.get_scheduler(named.scheduler_id, verbose=True)
        return named

    def create_generic_scheduler(self, simulator_id, name, active,
                                 process_memory, size, time_per_observation,
                                 observations_per_simulation, nodes,
                                 default_observation_requirement,
                                 configuration):
        # pylint: disable=too-many-arguments
        '''Creates a generic scheduler and returns it's id

        simulator_id   - The id of the simulator for which you wish to make a
                         scheduler.
        name           - The name for the scheduler.
        active         - True or false, specifying whether the scheduler is
                         initially active.
        process_memory - The amount of memory in MB that your simulations need.
        size           - The number of players for the scheduler.
        time_per_observation - The time you require to take a single
                         observation.
        observations_per_simulation - The number of observations to take per
                         simulation run.
        nodes          - The number of nodes required to run one of your
                         simulations.
        default_observation_requirement - The number of observations to take
                         of a profile in the absence of a specific request.
        configuration  - A dictionary representation that sets all the
                         run-time parameters for this scheduler.

        '''
        resp = self.request('post', 'generic_schedulers',
                            data={'scheduler': {
                                'simulator_id': simulator_id,
                                'name': name,
                                'active': active,
                                'process_memory': process_memory,
                                'size': size,
                                'time_per_observation': time_per_observation,
                                'observations_per_simulation':
                                observations_per_simulation,
                                'nodes': nodes,
                                'default_observation_requirement':
                                default_observation_requirement,
                                'configuration': configuration
                            }})
        resp.raise_for_status()
        return scheduler(json.loads(resp.text), _api=self)

    def update_generic_scheduler(self, scheduler_id, **kwargs):
        '''Update the parameters of a given scheduler

        kwargs are any of the mandatory arguments for create_generic_scheduler

        '''
        resp = self.request('put', 'generic_schedulers/%d.json' % scheduler_id,
                            data={'scheduler': kwargs})
        resp.raise_for_status()

    def generic_scheduler_add_role(self, scheduler_id, role, count):
        '''Add a role with specific count to the scheduler'''
        resp = self.request(
            'post', 'generic_schedulers/%d/add_role.json' % scheduler_id,
            data={'role': role, 'count': count})
        resp.raise_for_status()

    def generic_scheduler_remove_role(self, scheduler_id, role):
        '''Remove a role from the scheduler'''
        resp = self.request(
            'post', 'generic_schedulers/%d/remove_role.json' % scheduler_id,
            data={'role': role})
        resp.raise_for_status()

    def delete_generic_scheduler(self, scheduler_id):
        '''Delete a generic scheduler'''
        resp = self.request('delete',
                            'generic_schedulers/%d.json' % scheduler_id)
        resp.raise_for_status()

    def get_simulators(self):
        '''Get all known simulators'''
        resp = self.request('get', 'simulators')
        resp.raise_for_status()
        return json.loads(resp.text)['simulators']

    def get_simulator(self, simulator, version=None):
        '''Return a simulator based on it's id or name and version

        If simulator is an int, this attempts to return the simulator with that
        id (most efficient). If this is a string, this attempts to find the
        simulator with that name. An error is thrown if more than one
        exists. Optionally you can specify a specific version for simulators
        with more than one.

        '''
        if isinstance(simulator, int):
            resp = self.request('get', 'simulators/%d.json' % simulator)
            resp.raise_for_status()
            return json.loads(resp.text)
        elif version:
            return utils.only(
                sim for sim in self.get_simulators()
                if sim['name'] == simulator and sim['version'] == version)
        else:
            return utils.only(
                sim for sim in self.get_simulators()
                if sim['name'] == simulator)

    def get_games(self):
        '''Get a generator of all of the game structures'''
        resp = self.request('get', 'games')
        resp.raise_for_status()
        return json.loads(resp.text)['games']

    def get_game(self, game_name, granularity='structure'):
        '''Gets game data from egta

        granularity can be one of:

        structure    - returns the game information but no profile information.
        summary      - returns the game information and profiles with
                       aggregated payoffs.
        observations - returns the game information and profiles with data
                       aggregated at the observation level.
        full         - returns the game information and profiles with complete
                       observation information

        '''
        if isinstance(game_name, int):
            # Int implies we have a game id
            #
            # This call breaks convention because the api is broken, so we use
            # a different api.
            resp = requests.get(
                'https://egtaonline.eecs.umich.edu/games/%d.json' % game_name,
                data={'granularity': granularity, 'auth_token': self.auth})
            resp.raise_for_status()
            return (json.loads(json.loads(resp.text))
                    if granularity == 'structure'
                    else json.loads(resp.text))

        named = utils.only(g for g in self.get_games() if g.name == game_name)
        if granularity == 'structure':
            return named
        return self.get_game(named['id'], granularity=granularity)

    def get_profile(self, profile_id):
        '''Get a profile given it's id

        Profile ids can be found by using get_scheduler with the verbose
        flag

        '''
        resp = self.request('get', 'profiles/%d.json' % profile_id)
        resp.raise_for_status()
        return json.loads(resp.text)

    def add_profile(self, scheduler_id, profile_desc, count, update=False):
        '''Adds a profile with a given count to the scheduler

        If a profile with the same symmetry groups is already scheduled then
        this will have no effect.

        Setting update to true will cause a full scan through the profiles and
        remove the one that matches this one first. Only useful if updating the
        requested count of a profile.

        returns the profile id of the added profile

        '''
        if update:
            self.remove_profile(scheduler_id, profile_desc)

        resp = self.request(
            'post',
            'generic_schedulers/%d/add_profile.json' % scheduler_id,
            data={
                'assignment': str(analysis.profile(profile_desc)),
                'count': count
            })
        resp.raise_for_status()
        return json.loads(resp.text)['id']

    def remove_profile(self, scheduler_id, profile_desc):
        '''Removes a profile from a scheduler

        profile_desc can be an id (fast) or a profile object (slow). Does
        nothing if a matching profile description doesn't exist.

        '''
        if isinstance(profile_desc, int):
            resp = self.request(
                'post',
                'generic_schedulers/%d/remove_profile.json' % scheduler_id,
                data={'profile_id': profile_desc})
            resp.raise_for_status()
        else:
            # Iterates through all scheduled profiles for a match, and removes
            # one if found
            for prof in self.get_scheduler(
                    scheduler_id, True)['scheduling_requirements']:
                prof_id = prof['profile_id']
                symgrps = self.get_profile(prof_id).symmetry_groups
                if profile_desc == analysis.profile(symgrps):
                    self.remove_profile(scheduler_id, prof_id)
                    break

    def remove_all_profiles(self, scheduler_id):
        '''Removes all profiles from a scheduler'''
        for prof in self.get_scheduler(
                scheduler_id, True)['scheduling_requirements']:
            self.remove_profile(scheduler_id, prof['profile_id'])
