"""Module for computing dominated strategies"""
import math
import numpy as np

from gameanalysis import regret, subgame

# TODO: I dislike the way that the missing data handling is specified. (0, 1,
# 2) that are also used as boolean values. It also seems like some of these
# functions could be more efficient.

# TODO subgame.subgame duplicates all of the data in the game, but
# most of these functions only rely on the profile map. It'd be much faster to
# just scan the map once and filter out the invalid profiles. This game copy
# that just references the full game would be much faster, but lacks some of
# the functionality. I'm not sure what the best way to handle this is, but it
# would make this significantly more performant.


def iterated_elimination(game, criterion, *args, **kwargs):
    """Iterated elimination of dominated strategies

    input:
    criterion = function to find dominated strategies

    """
    reduced_game = _eliminate_strategies(game, criterion, *args, **kwargs)
    while len(reduced_game) < len(game):
        game = reduced_game
        reduced_game = _eliminate_strategies(game, criterion, *args, **kwargs)
    return game


def _eliminate_strategies(game, criterion, *args, **kwargs):
    eliminated = criterion(game, *args, **kwargs)
    return subgame.subgame(game,
                           {role: set(strats) - eliminated[role]
                            for role, strats in game.strategies.items()})


def _best_responses(game, prof):
    """Returns the best responses to a profile

    The return a dict mapping role to a two tuple. The first element of the two
    tuple is a set of confirmed best responses. The second is a set of
    strategies without data so they may be a better response or not.

    """
    prof = game.as_profile(prof)
    gains = regret.pure_strategy_deviation_gains(game, prof)
    responses = {}

    for role, strat_gains in gains.items():
        role_best = set()
        role_unknown = set()
        for strat, dev_gains in strat_gains.items():
            best_deviations = []
            biggest_gain = -np.inf
            unknown = set()
            for dev, gain in dev_gains.items():
                if math.isnan(gain):
                    unknown.add(dev)
                elif gain > biggest_gain:
                    best_deviations = {dev}
                    biggest_gain = gain
                elif gain == biggest_gain:
                    best_deviations.add(dev)
            role_best.update(best_deviations)
            role_unknown.update(unknown)
        responses[role] = (role_best, role_unknown)
    return responses


def never_best_response(game, conditional=1):
    """Never-a-weak-best-response criterion for IEDS

    This criterion is very strong: it can eliminate strict Nash equilibria.

    """
    non_best_responses = {role: set(strats) for role, strats
                          in game.strategies.items()}
    for prof in game:
        for role, (best, unknown) in _best_responses(game, prof).items():
            non_best_responses[role] -= set(best)
            if conditional:
                non_best_responses[role] -= unknown
    return non_best_responses


def _undominated(game, prof, conditional=1, weak=False):
    """Returns a mapping from role to strategies to strategy set

    The final set are the of the deviations that the first strategy is not
    dominated by. Another way to put it would this this maps from role to
    strategies to deviations that don't dominate strategy.

    """
    gains = regret.pure_strategy_deviation_gains(game, prof)
    return {role:
            {strat:
             {dev for dev, gain in dev_gains.items()
              if gain < 0  # There was a positive gain once
              or (gain == 0 and not weak)  # Tie counts for strict
              or (conditional and np.isnan(gain))  # Missing data
              or (conditional > 1 and prof.deviate(dev, strat) not in game)}
             for strat, dev_gains in strat_gains.items()}  # noqa
            for role, strat_gains in gains.items()}


def pure_strategy_dominance(game, conditional=1, weak=False):
    """Pure-strategy dominance criterion for IEDS

    Returns a mapping from role to dominated strategies.

    conditional:
        0: unconditional dominance
        1: conditional dominance
        2: extra-conservative conditional dominance

    """
    dominated_strategies = {role: {st: set(strats) - {st} for st in strats}
                            for role, strats in game.strategies.items()}
    for prof in game:
        undominated = _undominated(game, prof, conditional, weak)
        for role, strats in undominated.items():
            for strat, undom in strats.items():
                dominated_strategies[role][strat] -= undom
        if all(all(not dom for dom in strats.values())
               for strats in dominated_strategies.values()):
            break  # No domination

    return {role: {strat for strat, dom in strats.items() if dom}
            for role, strats in dominated_strategies.items()}
