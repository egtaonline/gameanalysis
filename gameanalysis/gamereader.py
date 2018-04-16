"""Module for loading an arbitrary game with its associated serializer"""
import contextlib
import json


def load(filelike):
    """Read a game from a file

    Parameters
    ----------
    filelike : file-like
        A file-like object to read the game from. The entire file will be
        consumed by this action.
    """
    return loads(filelike.read())


def loads(string):
    """Read a game from a string

    Parameters
    ----------
    string : str
        A string representation of the game.
    """
    with contextlib.suppress(json.JSONDecodeError):
        obj = json.loads(string)
        return loadj(obj)
    with contextlib.suppress(ValueError):
        from gameanalysis import gambit
        return gambit.loads(string)
    raise ValueError('no known format for game')


def loadj(obj): # pylint: disable=too-many-branches,too-many-return-statements
    """Read a game from serializable python objects

    Parameters
    ----------
    json : {...}
        The python object representation of a game encoded as json. Any valid
        game will be read and returned.
    """
    game_type, _ = obj.get('type', 'samplegame.').split('.', 1)
    if game_type == 'add':
        from gameanalysis import rsgame
        return rsgame.add_json(obj)
    elif game_type == 'aggfn':
        from gameanalysis import aggfn
        return aggfn.aggfn_json(obj)
    elif game_type == 'canon':
        from gameanalysis import canongame
        return canongame.canon_json(obj)
    elif game_type == 'const':
        from gameanalysis import rsgame
        return rsgame.const_json(obj)
    elif game_type == 'empty' or game_type == 'emptygame':
        from gameanalysis import rsgame
        return rsgame.empty_json(obj)
    elif game_type == 'game':
        from gameanalysis import paygame
        return paygame.game_json(obj)
    elif game_type == 'matrix':
        from gameanalysis import matgame
        return matgame.matgame_json(obj)
    elif game_type == 'neighbor':
        from gameanalysis import learning
        return learning.neighbor_json(obj)
    elif game_type == 'point':
        from gameanalysis import learning
        return learning.point_json(obj)
    elif game_type == 'rbf':
        from gameanalysis import learning
        return learning.rbfgame_json(obj)
    elif game_type == 'sample':
        from gameanalysis import learning
        return learning.sample_json(obj)
    elif game_type == 'samplegame':
        from gameanalysis import paygame
        return paygame.samplegame_json(obj)
    else:
        raise ValueError('unknown game type {}'.format(game_type))
