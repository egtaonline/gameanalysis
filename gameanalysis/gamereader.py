"""Module for loading an arbitrary game with its associated serializer"""
from gameanalysis import serialize
from gameanalysis import aggfn
from gameanalysis import matgame


_TYPE_MAP = {
    'game.1': serialize.read_game,
    'samplegame.1': serialize.read_samplegame,
    'aggfn.1': aggfn.read_aggfn,
    'matrix.1': matgame.read_matgame,
}


def read(json):
    """Read a game and its serializer in any form

    Parameters
    ----------
    json : {...}
        The python object representation of a game encoded as json. Any valid
        game will be read and returned with its appropriate serializer.
    """
    return _TYPE_MAP[json.get('type', 'samplegame.1')](json)
