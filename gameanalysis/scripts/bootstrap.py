"""Script for calculating regrets, deviations gains, and social welfare"""
import argparse
import collections
import json
import sys
from collections import abc

from gameanalysis import bootstrap
from gameanalysis import regret
from gameanalysis import rsgame


class SampleGameBootstrap(object):

    choices = {
        'regret': bootstrap.mixture_regret,
        'surplus': bootstrap.mixture_welfare,
    }

    @staticmethod
    def update_parser(parser):
        parser.description = """Compute bootstrap statistics using a sample
        game with data for every profile in the support of the subgame and
        potentially deviations."""
        parser.add_argument('profiles', metavar='<profile-file>',
                            type=argparse.FileType('r'), help="""File with
                            profiles from input game for which regrets should
                            be calculated. This file needs to be a json list of
                            profiles""")
        parser.add_argument('-t', '--type', default='regret',
                            choices=SampleGameBootstrap.choices, help="""What
to return. regret: returns the the regret of the profile; gains: returns a json
object of the deviators gains for every deviation; ne: return the "nash
                            equilibrium regrets", these are identical to gains.
                            (default: %(default)s)""")
        parser.add_argument('--processes', metavar='num-processes', type=int,
                            help="""The number of processes when constructing
                            bootstrap samples. Default will use all the cores
                            available.""")

    @staticmethod
    def run(args):
        game = rsgame.SampleGame.from_json(json.load(args.input))
        profiles = json.load(args.profiles)
        func = SampleGameBootstrap.choices[args.type]
        results = func(game, profiles, args.num_bootstraps, args.percentiles,
                       args.processes)
        json.dump(results.tolist(), args.output)
        args.output.write('\n')


bootstrap_types = collections.OrderedDict([
    ('sample', SampleGameBootstrap),
])


def update_parser(parser, base):
    sub_base = argparse.ArgumentParser(add_help=False, parents=[base])
    base_group = sub_base.add_argument_group('generic bootstrap arguments')
    base_group.add_argument('--percentiles', '-p', metavar='percentile',
                            type=float, nargs='+', help="""Percentiles to
                            return in [0, 100]. By default all bootstrap
                            samples will be returned sorted.""")
    base_group.add_argument('--num-bootstraps', '-n', metavar='num-bootstraps',
                            default=101, type=int, help="""The number of
bootstrap samples to acquire. More samples takes longer, but in general the
percentiles requested should be a multiple of this number minus 1, otherwise
there will be some error due to linear interpolation between points. (default:
                            %(default)s)""")

    parser.description = """Compute bootstrap statistics"""
    subcommands = parser.add_subparsers(title='Bootstrap types', dest='boot',
                                        help="""The bootstrap style analysis to
                                        run. sample - use a sample game with
                                        data for the entire subgame and
                                        deviations that are to be used.""")
    subcommands.required = True

    class Help(object):

        @staticmethod
        def update_parser(parser):
            parser.add_argument('boottype', metavar='bootstrap-type', nargs='?',
                                help="""Bootstrap type to get help on""")

        @staticmethod
        def run(args):
            if args.boottype is None:
                game = parser
            else:
                game = subcommands.choices[args.boottype]
            game.print_help()
            sys.exit(0)

    bootstrap_types['help'] = Help

    for name, cls in bootstrap_types.items():
        sub_parser = subcommands.add_parser(name, parents=[sub_base])
        cls.update_parser(sub_parser)


def main(args):
    bootstrap_types[args.boot].run(args)
