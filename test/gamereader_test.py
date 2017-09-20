import json
import pytest

import numpy as np

from gameanalysis import gamereader
from gameanalysis import agggen
from gameanalysis import matgame
from gameanalysis import gamegen
from gameanalysis import rsgame
from gameanalysis import serialize


GAME = gamegen.add_profiles(rsgame.emptygame([3, 4], [4, 3]), 0.5)
SERIAL = gamegen.serializer(GAME)

SGAME = gamegen.add_noise(GAME, 1, 3)
SSERIAL = serialize.samplegameserializer_copy(SERIAL)

AGG = agggen.random_aggfn([3, 4], [4, 3], 10)
ASERIAL = agggen.serializer(AGG)

MAT = matgame.matgame(np.random.random((4, 3, 2, 3)))
MSERIAL = matgame.matgameserializer_copy(gamegen.serializer(MAT))


@pytest.mark.parametrize('game,serial', [
    (GAME, SERIAL),
    (SGAME, SSERIAL),
    (AGG, ASERIAL),
    (MAT, MSERIAL),
])
def test_automatic_deserialization(game, serial):
    """Test that we can serialize and deserialize arbitrary games"""
    jgame = json.dumps(serial.to_json(game))
    copy, scopy = gamereader.read(json.loads(jgame))
    assert game == copy
    assert serial == scopy
