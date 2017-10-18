import json
import pytest

import numpy as np

from gameanalysis import gamereader
from gameanalysis import agggen
from gameanalysis import matgame
from gameanalysis import gamegen
from gameanalysis import rsgame


egame = rsgame.emptygame([3, 4], [4, 3])
game = gamegen.add_profiles(egame, 0.5)
sgame = gamegen.add_noise(game, 1, 3)
agg = agggen.random_aggfn([3, 4], [4, 3], 10)
mat = matgame.matgame(np.random.random((4, 3, 2, 3)))


@pytest.mark.parametrize('game', [egame, game, sgame, agg, mat])
def test_automatic_deserialization(game):
    """Test that we can serialize and deserialize arbitrary games"""
    jgame = json.dumps(game.to_json())
    copy = gamereader.read(json.loads(jgame))
    assert game == copy
