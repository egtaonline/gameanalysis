"""This file runs a profiler for nash equilibira finding

This script compiles the results into a restructured text file in the
documentation.
"""
import collections
import itertools
import logging
import multiprocessing
import sys
import time
import warnings
from os import path

import numpy as np
from numpy import linalg
import tabulate

from gameanalysis import gamegen
from gameanalysis import gamereader
from gameanalysis import learning
from gameanalysis import nash
from gameanalysis import regret


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


def generate_games(num):
    """Produce num random games per type"""
    np.random.seed(0)
    with open(path.join(_DIR, 'example_games', 'hard_nash.json')) as f:
        yield 'hard', gamereader.load(f).normalize()
    with open(path.join(_DIR, 'example_games', '2x2x2.nfg')) as f:
        yield 'gambit', gamereader.load(f).normalize()
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
        params = np.random.random(3) + .5
        yield 'shapley', gamegen.shapley(*params).normalize()
    for _ in range(num):
        wins = np.random.random(3) + .5
        loss = -np.random.random(3) - .5
        yield 'roshambo', gamegen.rock_paper_scissors(wins, loss).normalize()
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
    for _ in range(num):
        yield 'normagg large', gamegen.normal_aggfn(*random_agg_large())
    for _ in range(num):
        yield 'polyagg large', gamegen.poly_aggfn(*random_agg_large())
    for _ in range(num):
        yield 'sineagg large', gamegen.sine_aggfn(*random_agg_large())
    for _ in range(num):
        agg = gamegen.sine_aggfn(*random_agg_small())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            yield 'rbf', learning.rbfgame_train(agg)


def compute(sets):
    """Compute overlap information from dictionary of sets"""
    counts = collections.Counter(itertools.chain.from_iterable(sets.values()))
    if counts:
        cards = {k: len(s) / len(counts) for k, s in sets.items()}
        weights = {k: sum(1 / counts[e] for e in s) / len(counts)
                   for k, s in sets.items()}
        return set(counts.keys()), cards, weights
    else:
        zeros = {k: 0.0 for k in sets}
        return set(), zeros, zeros


def process_game(args):
    """Compute information about a game"""
    i, (name, game) = args
    np.random.seed(i)  # Reproducible randomness
    profiles = {
        'uniform': [game.uniform_mixture()],
        'pure': game.grid_mixtures(2),
        'biased': game.biased_mixtures(),
        'role biased': game.role_biased_mixtures(),
        'random': game.random_mixtures(game.num_role_strats.prod())
    }

    equilibria = []
    meth_eqa = {}
    methods = {}
    for method, func in nash._AVAILABLE_METHODS.items():
        logging.warning('Starting {} for {} {:d}'.format(
            method, name, i))
        speed = 0
        prof_eqa = {}
        for prof, mixes in profiles.items():
            profs = set()
            for mix in mixes:
                start = time.time()
                try:
                    eqm = func(game, mix)
                    speed += time.time() - start
                    reg = regret.mixture_regret(game, eqm)
                    if reg < 1e-3:
                        ind = next((i for i, eq in enumerate(equilibria)
                                    if linalg.norm(eqm - eq) < 1e-2),
                                   len(equilibria))
                        profs.add(ind)
                        if ind == len(equilibria):
                            equilibria.append(eqm)
                except Exception as ex:
                    speed += time.time() - start
                    logging.error(ex)
            prof_eqa[prof] = profs
        meth, prof_card, prof_weight = compute(prof_eqa)
        meth_eqa[method] = meth
        methods[method] = {'profcard': prof_card,
                           'profweight': prof_weight,
                           'speed': speed}
        logging.warning('Finished {} for {} {:d} - took {:f} seconds'.format(
            method, name, i, speed))
    _, meth_card, meth_weight = compute(meth_eqa)
    info = {
        ma: {
            'pair': {
                mb: len(set.intersection(sa, sb)) / len(sb)
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
    for k, v in new.items():
        if isinstance(v, dict):
            update(means.setdefault(k, {}), v, count)
        else:
            val = means.get(k, 0)
            means[k] = val + (v - val) * num / count


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

    names = {
        'replicator': 'Replicator Dynamics',
        'fictitious': 'Fictitious Play',
        'matching': 'Regret Matching',
        'optimize': 'Regret Minimization',
        'scarf': 'Simplical Subdivision',
        'regret': 'EXP3',
        'regret_dev': 'EXP3 Pure Deviations',
        'regret_pay': 'EXP3 Single Payoff',
    }
    titles = {
        m: '{} ({})'.format(' '.join(
            names.get(m, w.capitalize()) for w in m.split('_')), m)
        for m in results}

    with open(path.join(_DIR, 'sphinx', 'profile_nash.rst'), 'w') as f:
        f.write(""".. _profile_nash:

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
        f.write('Comparisons Between Methods\n'
                '----------------------------------\n\n')
        f.write(tabulate.tabulate(
            sorted(([titles[m], v['card'], v['weight'], v['speed'],
                     v['norm_speed']]
                    for m, v in agg_results.items()),
                   key=lambda x: x[1], reverse=True),
            headers=['Method', 'Fraction of Eqa', 'Weighted Fraction',
                     'Time (sec)', 'Normalized Time'],
            tablefmt='rst'))
        f.write('\n\n')

        for method, game_info in results.items():
            title = titles[method]
            f.write(title)
            f.write('\n')
            f.writelines(itertools.repeat('-', len(title)))
            f.write('\n\n')

            agg_info = agg_results[method]
            f.write('Initial Profile Rates\n'
                    '^^^^^^^^^^^^^^^^^^^^^\n\n')
            f.write(tabulate.tabulate(
                sorted(([k.capitalize(), v, agg_info['profweight'][k]]
                        for k, v in agg_info['profcard'].items()),
                       key=lambda x: x[1], reverse=True),
                headers=['Starting Type', 'Fraction of Eqa',
                         'Weighted Fraction'], tablefmt='rst'))
            f.write('\n\n')

            f.write('Compared to Other Methods\n'
                    '^^^^^^^^^^^^^^^^^^^^^^^^^\n\n')
            f.write(tabulate.tabulate(
                sorted(([titles[m], v,
                         agg_info['norm_speed'] / agg_results[m]['norm_speed']]
                        for m, v in agg_info['pair'].items()),
                       key=lambda x: x[1], reverse=True),
                headers=['Method', 'Fraction of Eqa', 'Time Ratio'],
                tablefmt='rst'))
            f.write('\n\n')

            f.write('By Game Type\n'
                    '^^^^^^^^^^^^\n\n')
            for game, info in game_info.items():
                f.write(game.capitalize())
                f.write('\n')
                f.writelines(itertools.repeat('"', len(game)))
                f.write('\n\n')
                f.write(tabulate.tabulate([
                    ['Fraction of Eqa', info['card']],
                    ['Weighted Fraction of Eqa', info['weight']],
                    ['Time (sec)', info['speed']],
                    ['Normalized Time', info['norm_speed']],
                ], headers=['Metric', 'Value'], tablefmt='rst'))
                f.write('\n\n')


# TODO Use argparse to set number of samples, logging, and number of processes
def main():
    """Run profiling"""
    write_file(profile(int(sys.argv[1]) if len(sys.argv) > 1 else 1))


if __name__ == '__main__':
    main()
