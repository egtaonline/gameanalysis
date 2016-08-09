"""Analyze a game using gp learn"""
import argparse
import json
import sys
from os import path

from gameanalysis import gameio
from gameanalysis import nash
from gameanalysis import regret
from gameanalysis import gpgame


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Perform
                                 game analysis""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Input file for script.
                    (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""Output file for script. (default: stdout)""")
PARSER.add_argument('--dist-thresh', metavar='<distance-threshold>',
                    type=float, default=1e-3, help="""L2 norm threshold, inside
                    of which, equilibria are considered identical. (default:
                    %(default)f)""")
PARSER.add_argument('--regret-thresh', '-r', metavar='<regret-threshold>',
                    type=float, default=1e-3, help="""Maximum regret to
                    consider an equilibrium confirmed. (default:
                    %(default)f)""")
PARSER.add_argument('--supp-thresh', '-t', metavar='<support-threshold>',
                    type=float, default=1e-3, help="""Maximum probability to
                    consider a strategy in support. (default: %(default)f)""")
PARSER.add_argument('--rand-restarts', metavar='<random-restarts>', type=int,
                    default=0, help="""The number of random points to add to
                    nash equilibrium finding. (default: %(default)d)""")
PARSER.add_argument('--max-iters', '-m', metavar='<maximum-iterations>',
                    type=int, default=10000, help="""The maximum number of
                    iterations to run through replicator dynamics. (default:
                    %(default)d)""")
PARSER.add_argument('--converge-thresh', '-c',
                    metavar='<convergence-threshold>', type=float,
                    default=1e-8, help="""The convergence threshold for
                    replicator dynamics. (default: %(default)f)""")
PARSER.add_argument('--processes', '-p', metavar='<num-procs>', type=int,
                    help="""Number of processes to use to run nash finding.
                    (default: number of cores)""")


def main():
    args = PARSER.parse_args()
    game, serial = gameio.read_game(json.load(args.input))

    # create gpgame
    lgame = gpgame.PointGPGame(game)

    # mixed strategy nash equilibria search
    methods = {
        'replicator': {
            'max_iters': args.max_iters,
            'converge_thresh': args.converge_thresh}}

    mixed_equilibria = game.trim_mixture_support(
        nash.mixed_nash(lgame, regret_thresh=args.regret_thresh,
                        dist_thresh=args.dist_thresh, processes=args.processes,
                        at_least_one=True, **methods),
        args.supp_thresh)

    equilibria = [(eqm, regret.mixture_regret(lgame, eqm))
                  for eqm in mixed_equilibria]

    # Output game
    args.output.write('Game Learning\n')
    args.output.write('=============\n')
    args.output.write(serial.to_str(game))
    args.output.write('\n\n')

    # Output social welfare
    args.output.write('Social Welfare\n')
    args.output.write('--------------\n')
    welfare, profile = game.get_max_social_welfare()
    if profile is None:
        args.output.write('There was no profile with complete payoff data\n\n')
    else:
        args.output.write('\nMaximum social welfare profile:\n')
        args.output.write(serial.to_prof_printstring(profile))
        args.output.write('Welfare: {:.4f}\n\n'.format(welfare))

        if game.num_roles > 1:
            for role, welfare, profile in zip(
                    serial.role_names, *game.get_max_social_welfare(True)):
                args.output.write('Maximum "{}" welfare profile:\n'.format(
                    role))
                args.output.write(serial.to_prof_printstring(profile))
                args.output.write('Welfare: {:.4f}\n\n'.format(welfare))

    args.output.write('\n')

    # Output Equilibria
    args.output.write('Equilibria\n')
    args.output.write('----------\n')

    if equilibria:
        args.output.write('Found {:d} equilibri{}\n\n'.format(
            len(equilibria), 'um' if len(equilibria) == 1 else 'a'))
        for i, (eqm, reg) in enumerate(equilibria, 1):
            args.output.write('Equilibrium {:d}:\n'.format(i))
            args.output.write(serial.to_prof_printstring(eqm))
            args.output.write('Regret: {:.4f}\n\n'.format(reg))
    else:
        args.output.write('Found no equilibria\n\n')
    args.output.write('\n')

    # Output json data
    args.output.write('Json Data\n')
    args.output.write('=========\n')
    json_data = {
        'equilibria': [serial.to_prof_json(eqm) for eqm, _ in equilibria]}
    json.dump(json_data, args.output)
    args.output.write('\n')


if __name__ == '__main__':
    main()
