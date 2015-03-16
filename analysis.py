#!/usr/bin/env python
#
# This package defined all of the data objects that are used for analysis. It
# wraps a lot of bryces methods to make things much easier.
'''A python module that contains analysis tools. Most are a wrapper around
Bryce's methods

'''

# pylint: disable=relative-import
import itertools
from collections import Counter as counter

import GameIO as gameio
import Subgames as subgames
import Nash as nash

import containers

class game_data(object):
    '''Game object, that just has wrapper convenience methods around Bryce's
    scripts

    '''
    def __init__(self, *args, **kwargs):
        self._analysis_game = gameio.read_JSON(dict(*args, **kwargs))

    def subgames(self, known_subgames=()):
        '''Find maximal subgames'''
        return (subgame(sg) for sg
                in subgames.cliques(self._analysis_game, known_subgames))

    def equilibria(self, eq_subgame=None, support_threshold=1e-3, **nash_args):
        '''Finds the equilibria of a subgame'''
        eq_subgame = eq_subgame or self._analysis_game.strategies
        analysis_subgame = subgames.subgame(self._analysis_game, eq_subgame)
        eqs = nash.mixed_nash(analysis_subgame,
                              dist_thresh=support_threshold, **nash_args)
        return (mixture(analysis_subgame.toProfile(
            e, supp_thresh=support_threshold)) for e in eqs)

    def responses(self, mix, gain_threshold=1e-3):
        '''Returns the gain for deviation by role and strategy

        Return value is {role: {strategy: gain}} where gain must be greater
        than gain_threshold and there is never a role pointing to am empty set
        of strategies.

        '''
        mix = self._analysis_game.toArray(mix)
        payoffs = self._analysis_game.expectedValues(mix)
        role_payoffs = (payoffs * mix).sum(axis=1)
        regrets = payoffs - role_payoffs[:, None]
        return {role: stratgains for role, stratgains
                in ((role, {s: g for s, g in zip(strats, gains) if g > gain_threshold})
                    for (role, strats), gains
                    in zip(self._analysis_game.strategies.iteritems(), regrets))
                if stratgains}


class profile(dict):
    '''A representation of a game profile'''
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], basestring):
            # String representation
            #
            # Split on appropriate delimiters and pass generator of tuples to
            # standard dictionary constructor
            super(profile, self).__init__(
                (r, {s: int(c) for c, s in (y.split(' ') for y in cs.split(', '))})
                for r, cs in (x.split(': ') for x in args[0].split('; ')))

        elif (len(args) == 1 and isinstance(args[0], list)
              and all(isinstance(g, dict)
                      and {'role', 'strategy', 'count'}.issubset(g.iterkeys())
                      for g in args[0])):
            # Symmetry group representation
            super(profile, self).__init__()
            for syg in args[0]:
                self.setdefault(syg['role'], {})[syg['strategy']] = syg['count']

        else:
            # Standard representation
            super(profile, self).__init__(*args, **kwargs)
            self.validate()

    def validate(self):
        '''Validates the profile, throws an error if structure is incorrect'''
        # TODO validate that all counts are equal
        for role, strats in self.iteritems():
            assert isinstance(role, basestring), 'role must be a string'
            assert isinstance(strats, dict), 'strategy counts must be in a dict'
            for strat, count in strats.iteritems():
                assert isinstance(strat, basestring), 'strategies must be strings'
                assert isinstance(count, int), 'strategy counts must be ints'

    def symmetry_groups(self):
        '''Convert profile to symmetry groups representation'''
        return list(itertools.chain.from_iterable(
            ({'role':r, 'strategy':s, 'count':c} for s, c in cs.iteritems())
            for r, cs in self.iteritems()))

    def profile_string(self):
        '''Convert profile to an egta string'''
        return '; '.join(
            '%s: %s' % (r, ', '.join(
                '%d %s' % (c, s) for s, c in cs.iteritems()))
            for r, cs in self.iteritems())

    def __str__(self):
        return self.profile_string()

    def __repr__(self):
        return 'profile(%s)' % super(profile, self).__repr__()


class subgame(containers.frozendict):
    '''Modification of a dict with convenience methods and validation'''

    def __init__(self, *args, **kwargs):
        temp = dict(*args, **kwargs)
        super(subgame, self).__init__((r, frozenset(s)) for r, s in temp.iteritems())

    def get_subgame_profiles(self, role_counts):
        '''Returns an iterable over all subgame profiles'''
        # pylint: disable=star-args
        # Compute the product of assignments by role and turn back into dictionary
        return (profile(rs) for rs in itertools.product(
            # Iterate through all strategy allocations per role and compute
            # every allocation of agents
            *([(r, counter(sprof)) for sprof
               in itertools.combinations_with_replacement(ss, role_counts[r])]
              # For each role
              for r, ss in self.iteritems())))

    def get_deviation_profiles(self, full_game, role_counts):
        '''Returns an iterable over all deviations from every subgame profile'''
        for role, strats in self.iteritems():
            deviation_counts = role_counts.copy()
            deviation_counts[role] -= 1
            for prof in self.get_subgame_profiles(deviation_counts):
                for deviation in full_game[role].difference(strats):
                    deviation_prof = prof.copy()
                    deviation_prof[role] = deviation_prof[role].copy()
                    deviation_prof[role][deviation] = 1
                    yield deviation_prof

    def with_deviation(self, role, strat):
        '''Returns a new subgame that includes the deviation'''
        assert role in self.iterkeys(), \
            'Can\'t have role %s deviate when roles are %s' % (role, self.keys())
        return subgame((r, itertools.chain(ss, [strat])) if r == role else (r, ss)
                       for r, ss in self.iteritems())

    def issubgame(self, other_subgame):
        '''Returns True if this is a subgame of other_subgame

        Throws an error if their roles don't match

        '''
        return self <= other_subgame

    def issupergame(self, other_subgame):
        '''Returns True if this is a supergame of other_subgame

        Throws an error if their roles don't match

        '''
        return self >= other_subgame

    def asset(self):
        '''Returns a view of the subgame as a set of role strategy pairs'''
        return frozenset(self.iterrolestrats())

    def iterrolestrats(self):
        '''Returns an iterable over role strategy pairs in this subgame'''
        return itertools.chain.from_iterable(
            ((r, s) for s in ss) for r, ss in self.iteritems())

    def __le__(self, other):
        '''is subgame'''
        assert self.keys() == other.keys(), 'subgames must have the same roles'
        return all(s <= other[r] for r, s in self.iteritems())

    def __ge__(self, other):
        '''is supergame'''
        assert self.keys() == other.keys(), 'subgames must have the same roles'
        return all(s >= other[r] for r, s in self.iteritems())

    def __lt__(self, other):
        '''is strict subgame'''
        assert self.keys() == other.keys(), 'subgames must have the same roles'
        return all(s < other[r] for r, s in self.iteritems())

    def __gt__(self, other):
        '''is strict supergame'''
        assert self.keys() == other.keys(), 'subgames must have the same roles'
        return all(s > other[r] for r, s in self.iteritems())

    def __repr__(self):
        return 'subgame(%s)' % super(subgame, self).__repr__()


class mixture(containers.frozendict):
    '''Representation of an mixture, not tied to an particular game'''
    # pylint: disable=too-few-public-methods

    def __init__(self, *args, **kwargs):
        temp = dict(*args, **kwargs)
        super(mixture, self).__init__(
            (r, containers.frozendict(
                (s, p) for s, p in ss.iteritems()))
            for r, ss in temp.iteritems())

    def support(self):
        '''Returns the subgame where this mixture has support'''
        return subgame((r, {s for s, p in ss.iteritems() if p > 0})
                       for r, ss in self.iteritems())


class subgame_set(object):
    '''Set of subgames, supports relevant operations on such a set'''
    # pylint: disable=too-few-public-methods

    # This class uses an inverted index from a role strategy tuple to every
    # subgame that contains that role strategy tuple.
    def __init__(self, iterable=()):
        self.inverted_index = {}
        for added_subgame in iterable:
            self.add(added_subgame)

    def add(self, added_subgame):
        '''Adds a subgame to the set

        Returns True if the set was modified

        '''
        # If dominated, don't add
        if added_subgame in self:
            return False

        # Otherwise, add and remove all subgames
        for key in added_subgame.iterrolestrats():
            bucket = self.inverted_index.setdefault(key, set())
            # Copy bucket to avoid concurrent modification
            for current_subgame in list(bucket):
                if added_subgame > current_subgame:
                    # Game in bucket is a subgame
                    bucket.remove(current_subgame)
            bucket.add(added_subgame)
        return True

    def __contains__(self, check_subgame):
        # Because every role strat key in the inverted index points to every
        # subgame that has that role strategy in it's support, we only need to
        # look in the bucket of an arbitrary role or strategy.
        key = next(check_subgame.iterrolestrats())
        bucket = self.inverted_index.get(key, set())
        return any(check_subgame <= game for game in bucket)

    def __repr__(self):
        # XXX Expensive!
        if not self.inverted_index:
            return '{}'
        else:
            return '{' + repr(set.union(*self.inverted_index.values()))[5:-2] + '}'
