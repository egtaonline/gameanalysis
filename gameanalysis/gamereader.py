"""Module for loading an arbitrary game with its associated serializer"""
from gameanalysis import aggfn
from gameanalysis import matgame
from gameanalysis import paygame
from gameanalysis import rsgame
from gameanalysis import learning


def read(json):
    """Read a game in any form

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
    game_type = json.get('type', 'samplegame.').split('.', 1)[0]
    return readers[game_type](json)
