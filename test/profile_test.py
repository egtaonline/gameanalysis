import numpy.random as rand

from gameanalysis import collect
from gameanalysis import gamegen
from gameanalysis import profile
from test import testutils


@testutils.apply([
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (1, 2, 2),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 1),
    (2, 2, 2),
    (2, [1, 2], 2),
    (2, 2, [1, 2]),
    (2, [1, 2], [1, 2]),
    (2, [3, 4], [2, 3]),
])
def simplex_project_test(roles, players, strategies):
    game = gamegen.role_symmetric_game(roles, players, strategies)
    for non_mixture in rand.uniform(-1, 1, (100, game.num_role_strats)):
        new_mix = profile.simplex_project(game, non_mixture)
        assert game.verify_array_mixture(new_mix), \
            "simplex project did not create a valid mixture"


def to_json_test():
    orig = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile(orig)
    assert orig == prof.to_json(), \
        "profile to_json didn't return correct dictionary"


def profile_from_json_test():
    orig = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    mapped = profile.Profile.from_json(orig).to_json()
    assert orig == mapped, \
        "from_json -> to_json wasn't an identity mapping"


def mixture_from_json_test():
    orig = {'r0': {'foo': .4, 'bar': .6}, 'r1': {'baz': 1}}
    mapped = profile.Mixture.from_json(orig).to_json()
    assert orig == mapped, \
        "from_json -> to_json wasn't an identity mapping"


def symmetry_groups_test():
    sym_grps = [
        dict(role='r0', strategy='foo', count=5),
        dict(role='r0', strategy='bar', count=6),
        dict(role='r1', strategy='baz', count=3),
    ]
    expected = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile.from_symmetry_groups(sym_grps)
    assert prof == expected, \
        "from_symmetry_groups did not make correct profile"
    new_sym_grps = prof.to_symmetry_groups()
    assert set(map(collect.frozendict, sym_grps)) == \
        set(map(collect.frozendict, new_sym_grps)), \
        "to / from sym groups wasn't identity"


def profile_from_input_profile_test():
    inp = {
        'r0': [('foo', 5, []), ('bar', 6, [])],
        'r1': [('baz', 3, [])],
    }
    expected = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile.from_input_profile(inp)
    assert prof == expected, \
        "from_input_profile didn't produce correct profile"


def profile_string_test():
    inp = 'r0: 5 foo, 6 bar; r1: 3 baz'
    expected = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile.from_profile_string(inp)
    assert prof == expected, \
        "from_profile_string didn't produce correct profile"

    def canonize(prof_str):
        return set(''.join(x for x in prof_str
                           if x.isalnum() or x.isspace()).split())

    new_str = prof.to_profile_string()
    assert canonize(new_str) == canonize(inp), \
        "to_profile_string didn't contain the correct 'words'"
    assert prof == profile.Profile.from_profile_string(new_str), \
        "to / from profile_string was not identity"


def remove_test():
    orig = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile(orig).remove('r0', 'foo')
    expected = {'r0': {'foo': 4, 'bar': 6}, 'r1': {'baz': 3}}
    assert prof == expected, \
        "Remove didn't properly remove a player"

    orig = {'r0': {'foo': 1, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile(orig).remove('r0', 'foo')
    expected = {'r0': {'bar': 6}, 'r1': {'baz': 3}}
    assert prof == expected, \
        "Remove didn't properly remove a strategy when players was 1"


def add_test():
    orig = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile(orig).add('r0', 'foo')
    expected = {'r0': {'foo': 6, 'bar': 6}, 'r1': {'baz': 3}}
    assert prof == expected, \
        "remove didn't properly add a player"


def deviation_test():
    orig = {'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}}
    prof = profile.Profile(orig).deviate('r0', 'foo', 'bar')
    expected = {'r0': {'foo': 4, 'bar': 7}, 'r1': {'baz': 3}}
    assert prof == expected, \
        "remove didn't properly deviate a player"


def profile_is_valid_test():
    players = {'r0': 11, 'r1': 3}
    strategies = {'r0': {'foo', 'bar', 'bizzle'}, 'r1': {'baz', 'boz'}}
    orig = profile.Profile({'r0': {'foo': 5, 'bar': 6}, 'r1': {'baz': 3}})
    assert orig.is_valid(players, strategies), \
        "didn't validate proper profile"
    assert not orig.add('r0', 'bar').is_valid(players, strategies), \
        "didn't invalidate profile for incorrect count"
    assert not orig.deviate('r0', 'bar', 'bz').is_valid(players, strategies), \
        "didn't invalidate profile for incorrect strategies"


def trim_support_test():
    mix = profile.Mixture({'r': {'1': 0.7, '2': 0.3}})
    not_trimmed = mix.trim_support(0.1)
    assert mix == not_trimmed, \
        "array got trimmed when it shouldn't"
    trimmed = mix.trim_support(0.4)
    assert {'r': {'1': 1}} == trimmed, \
        "array didn't get trimmed when it should"


def mixture_is_valid_test():
    strategies = {'r0': {'foo', 'bar', 'bizzle'}, 'r1': {'baz', 'boz'}}
    orig = profile.Mixture({'r0': {'foo': .6, 'bar': .4}, 'r1': {'baz': 1}})
    assert orig.is_valid(strategies), \
        "didn't validate proper mixture"
    invalid_sum = profile.Mixture(
        {'r0': {'foo': .6, 'bar': .41}, 'r1': {'baz': 1}})
    assert not invalid_sum.is_valid(strategies), \
        "didn't invalidate mixture for incorrect sum"
    invalid_strats = profile.Mixture(
        {'r0': {'foo': .6, 'bx': .4}, 'r1': {'baz': 1}})
    assert not invalid_strats.is_valid(strategies), \
        "didn't invalidate mixture for incorrect strategies"
