"""Functions for reading and writing gambit nfg files"""
import io
import itertools
import re
import warnings

import numpy as np

from gameanalysis import matgame
from gameanalysis import utils


def load(filelike):
    """Load a gambit game from a file"""
    return loads(filelike.read())


def loads(string):
    """Load a gambit game from a string"""
    match = _RE_NFG.match(string)
    if match:  # payoff form
        return _read_payoffs(match)
    match = _RE_NFGO.match(string)
    if match:  # outcome form
        return _read_outcomes(match)
    raise ValueError('failed to parse gambit format')


def dump(game, filelike):
    """Dump game to gambit file"""
    utils.check(game.is_complete(), 'gambit games must be complete')
    game = matgame.matgame_copy(game)
    filelike.write('NFG 1 R "gameanalysis game"\n{ ')

    for role in game.role_names:
        filelike.write('"')
        filelike.write(role.replace('"', '\\"'))
        filelike.write('" ')
    filelike.write('}\n{\n')
    for strats in game.strat_names:
        filelike.write('  { ')
        for strat in strats:
            filelike.write('"')
            filelike.write(strat.replace('"', '\\"'))
            filelike.write('" ')
        filelike.write('}\n')
    filelike.write('}\n\n{\n')

    perm = tuple(range(game.num_roles - 1, -1, -1)) + (game.num_roles,)
    pays = np.transpose(game.payoff_matrix(), perm)
    for outcome in pays.reshape((-1, game.num_roles)):
        filelike.write('  { "" ')
        filelike.write(', '.join(map(str, outcome)))
        filelike.write(' }\n')
    filelike.write('}\n1')
    for i in range(2, game.num_profiles + 1):
        filelike.write(' ')
        filelike.write(str(i))


def dumps(game):
    """Dump game as gambit string"""
    filelike = io.StringIO()
    dump(game, filelike)
    return filelike.getvalue()


def _read_payoffs(match):
    """Read gambit payoff format"""
    role_names = _string_list(match.group('roles'))
    num_strats = tuple(map(int, match.group('strats')[1:-1].split()))
    num_roles = len(num_strats)
    utils.check(
        len(role_names) == num_roles,
        "player names didn't match number of strategies")
    strats = utils.prefix_strings('s', sum(num_strats))
    strat_names = [list(itertools.islice(strats, n)) for n in num_strats]

    payoffs = list(map(float, match.group('payoffs').split()))
    matrix = np.empty(num_strats + (num_roles,))
    utils.check(
        len(payoffs) == matrix.size,
        'incorrect number of payoffs for strategies')
    inds = tuple(range(num_roles - 1, -1, -1)) + (num_roles,)
    np.transpose(matrix, inds).flat = payoffs

    return _normalize(role_names, strat_names, matrix)


def _read_outcomes(match):
    """Read gambit outcome format"""
    role_names = _string_list(match.group('roles'))
    num_roles = len(role_names)
    strat_names = [_string_list(m.group()) for m
                   in _RE_STRATS.finditer(match.group('strats')[1:-1])]
    utils.check(
        len(strat_names) == num_roles,
        'player names and strategies differed in length')
    num_strats = np.fromiter(map(len, strat_names), int, num_roles)

    outcomes = [np.zeros(num_roles)]
    for omatch in _RE_OUTCOME.finditer(match.group('outcomes')[1:-1]):
        outcome = omatch.group()[1:-1]
        pays = outcome[next(_RE_STR.finditer(outcome)).end():].split()
        utils.check(
            len(pays) == num_roles, 'outcome has wrong number of payoffs')
        outcomes.append(np.fromiter(  # pragma: no branch
            (float(s.rstrip(',')) for s in pays), float, num_roles))
    outcomes = np.stack(outcomes)

    inds = match.group('inds').split()
    utils.check(len(inds) == num_strats.prod(), 'wrong number of outcomes')
    inds = np.fromiter(map(int, inds), int, len(inds))

    matrix = np.empty(tuple(num_strats) + (num_roles,))
    tinds = tuple(range(num_roles - 1, -1, -1)) + (num_roles,)
    np.transpose(matrix, tinds).flat = outcomes[inds]
    return _normalize(role_names, strat_names, matrix)


def _dedup(lst):
    """Given a list of strings, modify inplace to remove duplicates"""
    dups = {}
    for i, string in enumerate(lst):
        dups.setdefault(string, []).append(i)
    for prefix, inds in dups.items():
        if len(inds) > 1:
            for i, new in zip(inds, utils.prefix_strings(prefix, len(inds))):
                lst[i] = new


def _normalize(role_names, strat_names, matrix):
    """Take gambit data and make it comply with gameanalysis standards"""
    num_roles = len(role_names)

    # Sort role names
    if not utils.is_sorted(role_names, strict=True):
        warnings.warn(
            "gambit player names aren't strictly sorted; modifying to comply "
            'with gameanalysis standards')
        _dedup(role_names)
        if not utils.is_sorted(role_names):
            order = sorted(range(num_roles), key=lambda i: role_names[i])
            role_names = [role_names[i] for i in order]
            strat_names = [strat_names[i] for i in order]
            shape = tuple(order) + (num_roles,)
            matrix = np.transpose(matrix, shape)[..., order]

    # Sort strat names
    if not all(utils.is_sorted(strats, strict=True) for strats in strat_names):
        warnings.warn(
            "gambit strategy names aren't strictly sorted; modifying to "
            'comply with gameanalysis standards')
        new_strats = []
        for role, strats in enumerate(strat_names):
            if not utils.is_sorted(strats, strict=True):
                _dedup(strats)
                if not utils.is_sorted(strats):
                    strats, order = zip(
                        *sorted((s, i) for i, s in enumerate(strats)))
                    shuffle = [slice(None)] * (num_roles + 1)
                    shuffle[role] = order
                    matrix = matrix[shuffle]
            new_strats.append(strats)
        strat_names = new_strats

    return matgame.matgame_names(role_names, strat_names, matrix)


def _string_list(lst):
    """Parse role names out of 'list'"""
    return [m.group()[1:-1].replace(r'\"', '"') for m
            in _RE_STR.finditer(lst[1:-1])]


def _list(element):
    """Regular expression for a gambit list"""
    return r'{(\s+' + element + r')+\s+}'


_STR = r'"(.|\n)*?(?<!\\)"'
_FLOAT = r'[-+]?(\d*\.?\d+|\d+\.\d*)([eE][-+]?\d+)?'
_OUTCOME = r'{{\s+{}(\s+{},?)+\s+}}'.format(_STR, _FLOAT)
_RE_STR = re.compile(_STR)
_RE_STRATS = re.compile(_list(_STR))
_RE_OUTCOME = re.compile(_OUTCOME)
_RE_NFG = re.compile((
    r'NFG\s+1\s+R\s+{}\s+(?P<roles>{})\s+(?P<strats>{})(\s+{})?'
    r'(?P<payoffs>(\s+{})+)\s*$').format(
        _STR, _list(_STR), _list(r'\d+'), _STR, _FLOAT))
_RE_NFGO = re.compile((
    r'NFG\s+1\s+R\s+{}\s+(?P<roles>{})\s+(?P<strats>{})(\s+{})?\s+'
    r'(?P<outcomes>{})(?P<inds>(\s+\d+)+)\s*$').format(
        _STR, _list(_STR), _list(_list(_STR)), _STR, _list(_OUTCOME)))
