"""Module for loading an arbitrary game with its associated serializer"""
import json

from gameanalysis import aggfn
from gameanalysis import gambit
from gameanalysis import learning
from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import rsgame


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
    try:
        obj = json.loads(string)
        return loadj(obj)
    except json.JSONDecodeError:
        pass  # Try another thing
    try:
        return gambit.loads(string)
    except AssertionError:
        pass  # Try another thing
    assert False, "no known format for game"


def loadj(obj):
    """Read a game from serializable python objects

    Parameters
    ----------
    json : {...}
        The python object representation of a game encoded as json. Any valid
        game will be read and returned.
    """
    readers = {
        'emptygame': rsgame.emptygame_json,
        'game': paygame.game_json,
        'samplegame': paygame.samplegame_json,
        'aggfn': aggfn.aggfn_json,
        'matrix': matgame.matgame_json,
        'rbf': learning.rbfgame_json,
        'sample': learning.sample_json,
        'point': learning.point_json,
        'neighbor': learning.neighbor_json,
    }
    game_type = obj.get('type', 'samplegame.').split('.', 1)[0]
    return readers[game_type](obj)
