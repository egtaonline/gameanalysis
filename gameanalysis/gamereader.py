"""Module for loading an arbitrary game with its associated serializer"""
import contextlib
import json
import logging


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
        logging.info("loading game with gambit format")
        from gameanalysis import gambit

        return gambit.loads(string)
    raise ValueError("no known format for game")


def loadj(obj):  # pylint: disable=too-many-branches,too-many-return-statements
    """Read a game from serializable python objects

    Parameters
    ----------
    json : {...}
        The python object representation of a game encoded as json. Any valid
        game will be read and returned.
    """
    game_type, _ = obj.get("type", "samplegame.").split(".", 1)
    if game_type == "add":
        logging.info("loading game with added game format")
        from gameanalysis import rsgame

        return rsgame.add_json(obj)
    elif game_type == "aggfn":
        logging.info("loading game with aggfn format")
        from gameanalysis import aggfn

        return aggfn.aggfn_json(obj)
    elif game_type == "canon":
        logging.info("loading game with canonical format")
        from gameanalysis import canongame

        return canongame.canon_json(obj)
    elif game_type == "const":
        logging.info("loading game with const format")
        from gameanalysis import rsgame

        return rsgame.const_json(obj)
    elif game_type == "empty" or game_type == "emptygame":
        logging.info("loading game with empty")
        from gameanalysis import rsgame

        return rsgame.empty_json(obj)
    elif game_type == "game":
        logging.info("loading game with payoff game format")
        from gameanalysis import paygame

        return paygame.game_json(obj)
    elif game_type == "matrix":
        logging.info("loading game with matrix format")
        from gameanalysis import matgame

        return matgame.matgame_json(obj)
    elif game_type == "neighbor":
        logging.info("loading game with neighbor learning format")
        from gameanalysis import learning

        return learning.neighbor_json(obj)
    elif game_type == "point":
        logging.info("loading game with point learning format")
        from gameanalysis import learning

        return learning.point_json(obj)
    elif game_type == "rbf":
        logging.info("loading game with rbf learning format")
        from gameanalysis import learning

        return learning.rbfgame_json(obj)
    elif game_type == "sample":
        logging.info("loading game with sample learning format")
        from gameanalysis import learning

        return learning.sample_json(obj)
    elif game_type == "samplegame":
        logging.info("loading game with sample payoff format")
        from gameanalysis import paygame

        return paygame.samplegame_json(obj)
    else:
        raise ValueError("unknown game type {}".format(game_type))


def dumpj(game):
    """Dump a game to json"""
    return game.to_json()


def dumps(game):
    """Dump a game to a string"""
    return json.dumps(dumpj(game))


def dump(game, file_like):
    """Dump a game to a file object"""
    return json.dump(dumpj(game), file_like)
