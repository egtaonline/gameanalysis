import itertools
import json

import numpy as np
import pytest

from gameanalysis import agggen
from gameanalysis import canongame
from gameanalysis import paygame


def test_canon():
    profs = [[2, 0, 0, 3],
             [1, 1, 0, 3]]
    pays = [[1, 0, 0, 1],
            [2, 3, 0, np.nan]]
    game = paygame.game([2, 3], [3, 1], profs, pays)
    cgame = canongame.canon(game)
    assert cgame.num_profiles == 2
    assert cgame.num_complete_profiles == 1
    pay = cgame.get_payoffs([2, 0, 0])
    assert np.allclose(pay, [1, 0, 0])

    expected = [[2, 0, 0],
                [1, 1, 0]]
    assert np.all(cgame.profiles() == expected)
    expected = [[1, 0, 0],
                [2, 3, 0]]
    assert np.allclose(cgame.payoffs(), expected)

    assert np.allclose(cgame.deviation_payoffs([1, 0, 0]), [1, 3, np.nan],
                       equal_nan=True)
    dev, jac = cgame.deviation_payoffs([0.5, 0.5, 0], jacobian=True)
    assert dev.shape == (3,)
    assert jac.shape == (3, 3)

    assert np.allclose(cgame.min_strat_payoffs(), [1, 3, np.nan],
                       equal_nan=True)
    assert np.allclose(cgame.max_strat_payoffs(), [2, 3, np.nan],
                       equal_nan=True)

    ngame = cgame.normalize()
    expected = [[0, 0, 0],
                [0.5, 1, 0]]
    assert np.allclose(ngame.payoffs(), expected)

    rgame = cgame.restrict([True, True, False])
    expected = [[1, 0],
                [2, 3]]
    assert np.allclose(rgame.payoffs(), expected)

    copy_str = json.dumps(cgame.to_json())
    copy = canongame.canon_json(json.loads(copy_str))
    assert cgame == copy
    assert hash(cgame) == hash(copy)

    assert [2, 0, 0] in cgame
    assert [0, 2, 0] not in cgame

    assert repr(cgame) == 'CanonGame([2], [3], 2 / 6)'


@pytest.mark.parametrize('strats', itertools.product(*[[1, 2]] * 3))
def test_random_canongame(strats):
    strats = np.array(strats)
    if np.all(strats == 1):
        return  # not a game
    game = agggen.normal_aggfn(2, strats, strats.sum())
    cgame = canongame.canon(game)
    paygame.game_copy(cgame)
