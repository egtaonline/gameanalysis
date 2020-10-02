"""This file runs a profiler for nash equilibira finding

This script compiles the results into a restructured text file in the
documentation.
"""
import argparse
import functools
import json
import logging
import multiprocessing
import sys
import time
import warnings
from os import path

import numpy as np

from gameanalysis import collect, gamegen, gamereader, learning, nash, regret

_DIR = path.join(path.dirname(__file__), "..")


def random_small():
    """Role symmetric game small sizes"""
    roles = np.random.randint(1, 4)
    players = np.random.randint(1 + (roles == 1), 5, roles)
    strats = np.random.randint(2, 5 - (roles > 2), roles)
    return players, strats


def random_agg_small():
    """Agg game small sizes"""
    roles = np.random.randint(1, 4)
    players = np.random.randint(1 + (roles == 1), 5, roles)
    strats = np.random.randint(2, 5 - (roles > 2), roles)
    functions = np.random.randint(2, 6)
    return players, strats, functions


def random_agg_large():
    """Agg game large sizes"""
    roles = np.random.randint(1, 4)
    players = np.random.randint(10, 101, roles)
    strats = np.random.randint(2, 5 - (roles > 2), roles)
    functions = np.random.randint(2, 6)
    return players, strats, functions


def generate_games(num):  # pylint: disable=too-many-branches
    """Produce num random games per type"""
    np.random.seed(0)
    with open(path.join(_DIR, "example_games", "hard_nash.json")) as fil:
        yield "hard", gamereader.load(fil).normalize()
    with open(path.join(_DIR, "example_games", "2x2x2.nfg")) as fil:
        yield "gambit", gamereader.load(fil).normalize()
    for _ in range(num):
        yield "random", gamegen.game(*random_small()).normalize()
    for _ in range(num):
        strats = np.random.randint(2, 5, np.random.randint(2, 4))
        yield "covariant", gamegen.covariant_game(strats).normalize()
    for _ in range(num):
        strats = np.random.randint(2, 5, 2)
        yield "zero sum", gamegen.two_player_zero_sum_game(strats).normalize()
    for _ in range(num):
        yield "prisoners", gamegen.prisoners_dilemma().normalize()
    for _ in range(num):
        yield "chicken", gamegen.sym_2p2s_game(0, 3, 1, 2).normalize()
    for _ in range(num):
        prob = np.random.random()
        yield "mix", gamegen.sym_2p2s_known_eq(prob).normalize()
    for _ in range(num):
        strats = np.random.randint(2, 4)
        plays = np.random.randint(2, 4)
        yield "polymatrix", gamegen.polymatrix_game(plays, strats).normalize()
    for _ in range(num):
        wins = np.random.random(3) + 0.5
        loss = -np.random.random(3) - 0.5
        yield "roshambo", gamegen.rock_paper_scissors(wins, loss).normalize()
    yield "shapley easy", gamegen.rock_paper_scissors(win=2).normalize()
    yield "shapley normal", gamegen.rock_paper_scissors(win=1).normalize()
    yield "shapley hard", gamegen.rock_paper_scissors(win=0.5).normalize()
    for _ in range(num):
        yield "normagg small", gamegen.normal_aggfn(*random_agg_small())
    for _ in range(num):
        yield "polyagg small", gamegen.poly_aggfn(*random_agg_small())
    for _ in range(num):
        yield "sineagg small", gamegen.sine_aggfn(*random_agg_small())
    for _ in range(num):
        facs = np.random.randint(2, 6)
        req = np.random.randint(1, facs)
        players = np.random.randint(2, 11)
        yield "congestion", gamegen.congestion(players, facs, req)
    for _ in range(num):
        strats = np.random.randint(2, 6)
        players = np.random.randint(2, 11)
        yield "local effect", gamegen.local_effect(players, strats)
    for _ in range(num):
        yield "normagg large", gamegen.normal_aggfn(*random_agg_large())
    for _ in range(num):
        yield "polyagg large", gamegen.poly_aggfn(*random_agg_large())
    for _ in range(num):
        yield "sineagg large", gamegen.sine_aggfn(*random_agg_large())
    for _ in range(num):
        agg = gamegen.sine_aggfn(*random_agg_small())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield "rbf", learning.rbfgame_train(agg)


def gen_profiles(game):
    """Generate profile types and names"""
    return {
        "uniform": lambda: [game.uniform_mixture()],
        "pure": game.pure_mixtures,
        "biased": game.biased_mixtures,
        "role biased": game.role_biased_mixtures,
        "random": lambda: game.random_mixtures(game.num_role_strats.prod()),
    }


def gen_methods():
    """Generate meothds and their names"""
    timeout = 30 * 60
    yield "replicator dynamics", False, functools.partial(
        nash.replicator_dynamics, timeout=timeout
    )
    yield "regret matching", False, functools.partial(
        nash._regret_matching_mix, timeout=timeout
    )  # pylint: disable=protected-access
    yield "regret minimization", False, nash.regret_minimize
    yield "fictitious play", False, functools.partial(
        nash.fictitious_play, timeout=timeout
    )
    yield "fictitious play long", False, functools.partial(
        nash.fictitious_play, max_iters=10 ** 7, timeout=timeout
    )
    yield "multiplicative weights dist", False, functools.partial(
        nash.multiplicative_weights_dist, timeout=timeout
    )
    yield "multiplicative weights stoch", False, functools.partial(
        nash.multiplicative_weights_stoch, timeout=timeout
    )
    yield "multiplicative weights bandit", False, functools.partial(
        nash.multiplicative_weights_bandit, timeout=timeout
    )
    yield "scarf 1", False, functools.partial(nash.scarfs_algorithm, timeout=60)
    yield "scarf 5", False, functools.partial(nash.scarfs_algorithm, timeout=5 * 60)
    yield "scarf", True, functools.partial(nash.scarfs_algorithm, timeout=timeout)


def process_game(args):  # pylint: disable=too-many-locals
    """Compute information about a game"""
    i, (name, game) = args
    np.random.seed(i)  # Reproducible randomness
    profiles = gen_profiles(game)

    reg_thresh = 1e-2  # FIXME
    conv_thresh = 1e-2 * np.sqrt(2 * game.num_roles)  # FIXME

    all_eqa = collect.mcces(conv_thresh)
    meth_times = {}
    meth_eqa = {}
    for method, single, func in gen_methods():
        logging.warning("Starting {} for {} {:d}".format(method, name, i))
        prof_times = {}
        prof_eqa = {}
        for prof, mix_gen in profiles.items():
            times = []
            eqa = []
            if prof != "uniform" and single:
                continue
            for mix in mix_gen():
                start = time.time()
                eqm = func(game, mix)
                speed = time.time() - start
                reg = regret.mixture_regret(game, eqm)
                times.append(speed)
                if reg < reg_thresh:
                    all_eqa.add(eqm, reg)
                    eqa.append(eqm)
            prof_times[prof] = times
            prof_eqa[prof] = eqa
        meth_times[method] = prof_times
        meth_eqa[method] = prof_eqa
        logging.warning(
            "Finished {} for {} {:d} - took {:f} seconds".format(method, name, i, speed)
        )

    inds = {}
    for norm, _ in all_eqa:
        inds[norm] = len(inds)

    for prof_eqa in meth_eqa.values():
        for prof, eqa in prof_eqa.items():
            prof_eqa[prof] = list({inds[all_eqa.get(e)] for e in eqa})

    return name, meth_times, meth_eqa


def profile(num):
    """Compute profiling information"""
    with multiprocessing.Pool() as pool:
        for name, times, eqa in pool.imap_unordered(
            process_game, enumerate(generate_games(num))
        ):
            json.dump({"name": name, "times": times, "eqa": eqa}, sys.stdout)
            sys.stdout.write("\n")


def main(*argv):
    """Run profiling"""
    parser = argparse.ArgumentParser(
        description="""Run nash profiling and update documentation with
        results."""
    )
    parser.add_argument(
        "num",
        nargs="?",
        metavar="<num>",
        type=int,
        default=1,
        help="""Number
        of each game type to generate, when games are drawn from a
        distribution.""",
    )
    # TODO Set logging and number of processes
    args = parser.parse_args(argv)

    profile(args.num)


if __name__ == "__main__":
    main(*sys.argv[1:])
