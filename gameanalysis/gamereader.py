"""Module for loading an arbitrary game with its associated serializer"""
from gameanalysis import aggfn
from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import rsgame


_TYPE_MAP = {
    'emptygame.1': rsgame.emptygame_json,
    'game.1': paygame.game_json,
    'samplegame.1': paygame.samplegame_json,
    'aggfn.1': aggfn.aggfn_json,
    'matrix.1': matgame.matgame_json,
}


def read(json):
    """Read a game in any form

    Parameters
    ----------
    json : {...}
        The python object representation of a game encoded as json. Any valid
        game will be read and returned.
    """
    return _TYPE_MAP[json.get('type', 'samplegame.1')](json)
