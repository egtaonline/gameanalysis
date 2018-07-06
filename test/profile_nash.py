"""This file runs a profiler for nash equilibira finding

This script compiles the results into a restructured text file in the
documentation.
"""
import argparse
import collections
import functools
import itertools
import logging
import multiprocessing
import sys
import time
import warnings
from os import path

import numpy as np
import tabulate

from gameanalysis import collect, gamegen, gamereader, learning, nash, regret

_DIR = path.join(path.dirname(__file__), '..')


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


def generate_games(num): # pylint: disable=too-many-branches
    """Produce num random games per type"""
    np.random.seed(0)
    with open(path.join(_DIR, 'example_games', 'hard_nash.json')) as fil:
        yield 'hard', gamereader.load(fil).normalize()
    with open(path.join(_DIR, 'example_games', '2x2x2.nfg')) as fil:
        yield 'gambit', gamereader.load(fil).normalize()
    for _ in range(num):
        yield 'random', gamegen.game(*random_small()).normalize()
    for _ in range(num):
        strats = np.random.randint(2, 5, np.random.randint(2, 4))
        yield 'covariant', gamegen.covariant_game(strats).normalize()
    for _ in range(num):
        strats = np.random.randint(2, 5, 2)
        yield 'zero sum', gamegen.two_player_zero_sum_game(strats).normalize()
    for _ in range(num):
        yield 'prisoners', gamegen.prisoners_dilemma().normalize()
    for _ in range(num):
        yield 'chicken', gamegen.sym_2p2s_game(0, 3, 1, 2).normalize()
    for _ in range(num):
        prob = np.random.random()
        yield 'mix', gamegen.sym_2p2s_known_eq(prob).normalize()
    for _ in range(num):
        strats = np.random.randint(2, 4)
        plays = np.random.randint(2, 4)
        yield 'polymatrix', gamegen.polymatrix_game(plays, strats).normalize()
    for _ in range(num):
        wins = np.random.random(3) + .5
        loss = -np.random.random(3) - .5
        yield 'roshambo', gamegen.rock_paper_scissors(wins, loss).normalize()
    yield 'shapley easy', gamegen.rock_paper_scissors(win=2).normalize()
    yield 'shapley normal', gamegen.rock_paper_scissors(win=1).normalize()
    yield 'shapley hard', gamegen.rock_paper_scissors(win=0.5).normalize()
    for _ in range(num):
        yield 'normagg small', gamegen.normal_aggfn(*random_agg_small())
    for _ in range(num):
        yield 'polyagg small', gamegen.poly_aggfn(*random_agg_small())
    for _ in range(num):
        yield 'sineagg small', gamegen.sine_aggfn(*random_agg_small())
    for _ in range(num):
        facs = np.random.randint(2, 6)
        req = np.random.randint(1, facs)
        players = np.random.randint(2, 11)
        yield 'congestion', gamegen.congestion(players, facs, req)
    for _ in range(num):
        strats = np.random.randint(2, 6)
        players = np.random.randint(2, 11)
        yield 'local effect', gamegen.local_effect(players, strats)
    yield 'normagg large', gamegen.normal_aggfn(*random_agg_large())
    yield 'polyagg large', gamegen.poly_aggfn(*random_agg_large())
    yield 'sineagg large', gamegen.sine_aggfn(*random_agg_large())
    for _ in range(num):
        agg = gamegen.sine_aggfn(*random_agg_small())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            yield 'rbf', learning.rbfgame_train(agg)


def compute(thresh, sets):
    """Compute overlap information from dictionary of sets"""
    joined = collect.mcces(thresh, itertools.chain.from_iterable(
        sets.values()))
    if joined: # pylint: disable=no-else-return
        uniqs = {k: set(joined.get(v) for v, _ in vecs)
                 for k, vecs in sets.items()}
        counts = collections.Counter(itertools.chain.from_iterable(
            uniqs.values()))
        cards = {k: len(uniq) / len(joined) for k, uniq in uniqs.items()}
        weights = {k: sum(1 / counts[u] for u in uniq) / len(joined)
                   for k, uniq in uniqs.items()}
        return joined, cards, weights
    else:
        zeros = {k: 0.0 for k in sets}
        return joined, zeros, zeros


def gen_profiles(game):
    """Generate profile types and names"""
    return {
        'uniform': lambda: [game.uniform_mixture()],
        'pure': game.pure_mixtures,
        'biased': game.biased_mixtures,
        'role biased': game.role_biased_mixtures,
        'random': lambda: game.random_mixtures(game.num_role_strats.prod()),
    }


def gen_methods():
    """Generate meothds and their names"""
    yield 'replicator dynamics', False, nash.replicator_dynamics
    yield 'regret matching', False, nash._regret_matching_mix # pylint: disable=protected-access
    yield 'regret minimization', False, nash.regret_minimize
    yield 'fictitious play', False, nash.fictitious_play
    yield 'fictitious play long', False, functools.partial(
        nash.fictitious_play, max_iters=10**7, timeout=30 * 60)
    yield (
        'multiplicative weights dist', False, nash.multiplicative_weights_dist)
    yield (
        'multiplicative weights stoch', False,
        nash.multiplicative_weights_stoch)
    yield (
        'multiplicative weights bandit', False,
        nash.multiplicative_weights_bandit)
    yield 'scarf 1', True, functools.partial(nash.scarfs_algorithm, timeout=60)
    yield 'scarf 5', True, functools.partial(
        nash.scarfs_algorithm, timeout=5 * 60)
    yield 'scarf 30', True, functools.partial(
        nash.scarfs_algorithm, timeout=30 * 60)


def process_game(args): # pylint: disable=too-many-locals
    """Compute information about a game"""
    i, (name, game) = args
    np.random.seed(i)  # Reproducible randomness
    profiles = gen_profiles(game)

    reg_thresh = 1e-2 # FIXME
    conv_thresh = 1e-2 * np.sqrt(2 * game.num_roles) # FIXME

    meth_eqa = {}
    methods = {}
    for method, single, func in gen_methods():
        logging.warning('Starting {} for {} {:d}'.format(method, name, i))
        speed = 0
        count = 0
        prof_eqa = {}
        for prof, mix_gen in profiles.items():
            if prof != 'uniform' and single:
                continue
            profs = collect.mcces(conv_thresh)
            for mix in mix_gen():
                count += 1
                start = time.time()
                eqm = func(game, mix)
                speed += (time.time() - start) / count
                reg = regret.mixture_regret(game, eqm)
                if reg < reg_thresh:
                    profs.add(eqm, reg)
            prof_eqa[prof] = profs
        meth, prof_card, prof_weight = compute(conv_thresh, prof_eqa)
        meth_eqa[method] = meth
        methods[method] = {'profcard': prof_card,
                           'profweight': prof_weight,
                           'speed': speed}
        logging.warning('Finished {} for {} {:d} - took {:f} seconds'.format(
            method, name, i, speed))
    _, meth_card, meth_weight = compute(conv_thresh, meth_eqa)
    info = {
        ma: {
            'pair': {
                mb: (len(set(sb.get(e) for e, _ in sa).difference({None})) /
                     len(sb))
                    if sb else 1.0
                for mb, sb in meth_eqa.items()
                if mb != ma},
            'card': meth_card[ma],
            'weight': meth_weight[ma],
            **methods[ma]}
        for ma, sa in meth_eqa.items()}
    return name, info


def update(means, new, count, num=1):
    """Recursively update mean dictionary"""
    for key, val in new.items():
        if isinstance(val, dict):
            update(means.setdefault(key, {}), val, count)
        else:
            value = means.get(key, 0)
            means[key] = value + (val - value) * num / count


def profile(num):
    """Compute profiling information"""
    methods = {}
    with multiprocessing.Pool() as pool:
        for name, res in pool.imap_unordered(
                process_game, enumerate(generate_games(num))):
            for method, info in res.items():
                meth = methods.setdefault(method, {})
                game = meth.setdefault(name, {'count': 0})
                game['count'] += 1
                update(game, info, game['count'])
    return methods


def write_file(results):
    """Write file with results"""
    # Compute normalized speeds
    for game in next(iter(results.values())):
        min_speed = min(g[game]['speed'] for g in results.values())
        for games in results.values():
            games[game]['norm_speed'] = games[game]['speed'] / min_speed

    # Aggregate info over all games
    agg_results = {}
    for method, game_info in results.items():
        agg_info = {}
        game_count = 0
        for info in game_info.values():
            count = info.pop('count')
            game_count += count
            update(agg_info, info, game_count, count)
        agg_results[method] = agg_info

    with open(path.join(_DIR, 'sphinx', 'profile_nash.rst'), 'w') as fil:
        fil.write(""".. _profile_nash:

Nash Equilibrium Methods Comparison
===================================

For each method available for Nash equilibrium finding, this lists various
information about the performance across different game types and starting
locations. "Fraction of Eqa" is the mean fraction of all equilibria found via
that method or starting location. "Weigted Fraction (of Eqa)" is the same,
except each equilibrium is down weighted by the number of methods that found
it, thus a larger weighted fraction indicates that this method found more
unique equilibria. "Time" is the average time in seconds it took to run this
method for every starting location. "Normalized Time" sets the minimum time for
each game type and sets it to one, thus somewhat mitigating the fact that
certain games may be more difficult than others. It also provides an easy
comparison metric to for baseline timing.

""")
        fil.write(
            'Comparisons Between Methods\n'
            '----------------------------------\n\n')
        fil.write(tabulate.tabulate(
            sorted(([m.title(), v['card'], v['weight'], v['speed'],
                     v['norm_speed']]
                    for m, v in agg_results.items()),
                   key=lambda x: x[1], reverse=True),
            headers=['Method', 'Fraction of Eqa', 'Weighted Fraction',
                     'Time (sec)', 'Normalized Time'],
            tablefmt='rst'))
        fil.write('\n\n')

        for method, game_info in results.items():
            title = method.title()
            fil.write(title)
            fil.write('\n')
            fil.writelines(itertools.repeat('-', len(title)))
            fil.write('\n\n')

            agg_info = agg_results[method]
            fil.write(
                'Initial Profile Rates\n'
                '^^^^^^^^^^^^^^^^^^^^^\n\n')
            fil.write(tabulate.tabulate(
                sorted(([k.capitalize(), v, agg_info['profweight'][k]]
                        for k, v in agg_info['profcard'].items()),
                       key=lambda x: x[1], reverse=True),
                headers=['Starting Type', 'Fraction of Eqa',
                         'Weighted Fraction'], tablefmt='rst'))
            fil.write('\n\n')

            fil.write(
                'Compared to Other Methods\n'
                '^^^^^^^^^^^^^^^^^^^^^^^^^\n\n')
            fil.write(tabulate.tabulate(
                sorted(([m.title(), v,
                         agg_info['norm_speed'] / agg_results[m]['norm_speed']]
                        for m, v in agg_info['pair'].items()),
                       key=lambda x: x[1], reverse=True),
                headers=['Method', 'Fraction of Eqa', 'Time Ratio'],
                tablefmt='rst'))
            fil.write('\n\n')

            fil.write(
                'By Game Type\n'
                '^^^^^^^^^^^^\n\n')
            for game, info in game_info.items():
                fil.write(game.capitalize())
                fil.write('\n')
                fil.writelines(itertools.repeat('"', len(game)))
                fil.write('\n\n')
                fil.write(tabulate.tabulate([
                    ['Fraction of Eqa', info['card']],
                    ['Weighted Fraction of Eqa', info['weight']],
                    ['Time (sec)', info['speed']],
                    ['Normalized Time', info['norm_speed']],
                ], headers=['Metric', 'Value'], tablefmt='rst'))
                fil.write('\n\n')


def main(*argv):
    """Run profiling"""
    parser = argparse.ArgumentParser(
        description="""Run nash profiling and update documentation with
        results.""")
    parser.add_argument(
        'num', nargs='?', metavar='<num>', type=int, default=1, help="""Number
        of each game type to generate, when games are drawn from a
        distribution.""")
    # TODO Set logging and number of processes
    args = parser.parse_args(argv)

    write_file(profile(args.num))


if __name__ == '__main__':
    main(*sys.argv[1:])
