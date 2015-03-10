import requests
import json
import logging

import analysis

class adict(dict):
    # pylint: disable=too-few-public-methods
    """A dictionary that has attribute access to its keys"""
    def __init__(self, *args, **kwargs):
        super(adict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class scheduler(adict):
    """Represents information about a scheduler"""

    def num_running_profiles(self):
        """Get the number of currently running profiles"""
        # Update info
        update = self._api.get_scheduler(self.id, verbose=True)
        # Sum profiles that aren't complete
        return sum(req['current_count'] < req['requirement']
                   for req in update.scheduling_requirements or [])

    def add_profile(self, profile_desc, update=False):
        """Add a profile to this simulator

        See egtaonline.remove_profile for more documentation

        """
        self._api.add_profile(self.id, profile_desc, update)

    def remove_profile(self, profile_desc):
        """Removes a specific profile from this simulator

        See egtaonline.remove_profile for more documentation

        """
        self._api.remove_profile(self.id, profile_desc)

    def remove_all_profiles(self):
        """Remove all profiles from this scheduler"""
        self._api.remove_all_profiles(self.id)

class egtaonline(object):
    """Class to wrap egtaonline api"""
    def __init__(self, auth_token, domain='egtaonline.eecs.umich.edu',
                 logLevel=0):
        self.domain = domain
        self.auth = auth_token
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logLevel)

    def _request(self, verb, api, data={}):
        """Convenience method for making requests"""
        true_data = {'auth_token': self.auth}
        true_data.update(data)
        url = 'http://%s/api/v3/%s' % (self.domain, api)
        self.log.info('%s request to %s with data %s', verb, url, true_data)
        return requests.request(verb, url, data=true_data)

    def get_generic_schedulers(self):
        """Get all known generic schedulers"""
        resp = self._request('get', 'generic_schedulers')
        resp.raise_for_status()
        return [adict(x) for x
                in json.loads(resp.text)['generic_schedulers']]

    def get_scheduler(self, scheduler_name, verbose=False):
        """Get a scheduler by name or id

        If scheduler is an int, then this will get the scheduler by id (more
        efficient and works for all schedulers) if it is a string, then this
        will get the generic scheduler by name

        """
        if isinstance(scheduler_name, int):
            data = {'granularity': 'with_requirements'} if verbose else {}
            resp = self._request('get', 'schedulers/%d.json' % scheduler_name, data)
            resp.raise_for_status()
            return scheduler(json.loads(resp.text), _api=self)

        named = scheduler(analysis.only(s for s in self.get_generic_schedulers()
                                        if s.name == scheduler_name))
        if verbose:
            return self.get_scheduler(named.id, verbose=True)
        return named

    def get_simulators(self):
        """Get all known simulators"""
        resp = self._request('get', 'simulators')
        resp.raise_for_status()
        return [adict(x) for x in json.loads(resp.text)['simulators']]

    def get_simulator(self, simulator, version=None):
        """Return a simulator based on it's id or name and version

        If simulator is an int, this attempts to return the simulator with that
        id (most efficient). If this is a string, this attempts to find the
        simulator with that name. An error is thrown if more than one
        exists. Optionally you can specify a specific version for simulators
        with more than one.

        """
        if isinstance(simulator, int):
            resp = self._request('get', 'simulators/%d.json' % simulator)
            resp.raise_for_status()
            return adict(json.loads(resp.text))
        elif version:
            return adict(analysis.only(
                s for s in self.get_simulators()
                if s['name'] == simulator and s['version'] == version))
        else:
            return adict(analysis.only(
                s for s in self.get_simulators()
                if s['name'] == simulator))

    def get_games(self):
        """Get a list of all of the games"""
        resp = self._request('get', 'games')
        resp.raise_for_status()
        return [adict(x) for x in json.loads(resp.text)['games']]

    def get_game(self, game_name, granularity='structure'):
        """Gets game data from egta

        granularity can be one of:

        structure    - returns the game information but no profile information.
        summary      - returns the game information and profiles with aggregated
                       payoffs.
        observations - returns the game information and profiles with data
                       aggregated at the observation level.
        full         - returns the game information and profiles with complete
                       observation information

        """
        if isinstance(game_name, int):
            # Int implies we have a game id
            #
            # This call is a little funny, because a path is returned instead of
            # the data, so a second request has to be made to actually get the data
            resp = self._request('get', 'games/%d.json' % game_name,
                                 data={'granularity': granularity})
            resp.raise_for_status()
            url = 'http://%s%s' % (self.domain, resp.text[resp.text.find('/public/') + 7:])
            resp = requests.get(url)
            resp.raise_for_status()
            return adict(json.loads(json.loads(resp.text))
                         if granularity == 'structure'
                         else json.loads(resp.text))

        named = analysis.only(g for g in self.get_games() if g.name == game_name)
        if granularity == 'structure':
            return named
        return self.get_game(named.id, granularity=granularity)


    def get_profile(self, profile_id):
        """Get a profile given it's id

        Profile ids can be found by using get_scheduler with the verbose
        flag

        """
        resp = self._request('get', 'profiles/%d.json' % profile_id)
        resp.raise_for_status()
        return adict(json.loads(resp.text))

    def add_profile(self, scheduler_id, profile_desc, count, update=False):
        """Adds a profile with a given count to the scheduler

        If a profile with the same symmetry groups is already scheduled then
        this will have no effect.

        Setting update to true will cause a full scan through the profiles and
        remove the one that matches this one first. Only useful if updating the
        requested count of a profile.

        """
        if update:
            self.remove_profile(scheduler_id, profile_desc)

        resp = self._request(
            'post',
            'generic_schedulers/%d/add_profile.json' % scheduler_id,
            data = {
                'assignment': str(analysis.profile(profile_desc)),
                'count': count
            })
        resp.raise_for_status()

    def remove_profile(self, scheduler_id, profile_desc):
        """Removes a profile from a scheduler

        profile_desc can be an id (fast) or a profile object (slow). Does
        nothing if a matching profile description doesn't exist.

        """
        if isinstance(profile_desc, int):
            resp = self._request(
                'post',
                'generic_schedulers/%d/remove_profile.json' % scheduler_id,
                data = {'profile_id': profile_desc})
            resp.raise_for_status()
        else:
            # Iterates through all scheduled profiles for a match, and removes
            # one if found
            for prof in self.get_scheduler(scheduler_id, True).scheduling_requirements:
                prof_id = prof['profile_id']
                symgrps = self.get_profile(prof_id).symmetry_groups
                if profile_desc == analysis.profile(symgrps):
                    self.remove_profile(scheduler_id, prof_id)
                    break


    def remove_all_profiles(self, scheduler_id):
        """Removes all profiles from a scheduler"""
        for prof in self.get_scheduler(scheduler_id, True).scheduling_requirements:
            self.remove_profile(scheduler_id, prof['profile_id'])

