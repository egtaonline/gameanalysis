import argparse
import collections
import sys

from gameanalysis.scripts import bootstrap
from gameanalysis.scripts import convert
from gameanalysis.scripts import dominance
from gameanalysis.scripts import gamegen
from gameanalysis.scripts import nash
from gameanalysis.scripts import payoff
from gameanalysis.scripts import reduction
from gameanalysis.scripts import regret
from gameanalysis.scripts import subgames


_BASE = argparse.ArgumentParser(add_help=False)
_BASE_GROUP = _BASE.add_argument_group('game analysis arguments')
_BASE_GROUP.add_argument('--input', '-i', metavar='<input-file>',
                         default=sys.stdin, type=argparse.FileType('r'),
                         help="""Input file for script.  (default: stdin)""")
_BASE_GROUP.add_argument('--output', '-o', metavar='<output-file>',
                         default=sys.stdout, type=argparse.FileType('w'),
                         help="""Output file for script. (default: stdout)""")

_PARSER = argparse.ArgumentParser(prog='ga', parents=[_BASE], description="""Command line access
                                  to the game analysis toolkit""")
_SUBPARSERS = _PARSER.add_subparsers(title='Subcommands', dest='command',
                                     help="""The specific aspect of the toolkit
to interact with. See each possible command for details. boot - bootstrap, conv
- game conversion, dom - strategy dominance, gen - generate games, nash -
                                     compute nash equilibria, pay - compute
                                     profile payoffs, red - reduce games, reg -
                                     compute profile regret, sub - compute
                                     subgames, help - get help on commands.""")
_SUBPARSERS.required = True


class help(object):

    @staticmethod
    def update_parser(parser, base):
        parser.add_argument('subcommand', metavar='command', nargs='?',
                            help="""Command to get help on""")

    @staticmethod
    def main(args):
        if args.subcommand is None:
            parser = _PARSER
        else:
            parser = _SUBPARSERS.choices[args.subcommand]
        parser.print_help()


_SUBCOMMANDS = collections.OrderedDict([
    ('boot', bootstrap),
    ('conv', convert),
    ('dom', dominance),
    ('gen', gamegen),
    ('nash', nash),
    ('pay', payoff),
    ('red', reduction),
    ('reg', regret),
    ('sub', subgames),
    ('help', help),
])


def main():
    for name, module in _SUBCOMMANDS.items():
        parser = _SUBPARSERS.add_parser(name, parents=[_BASE])
        module.update_parser(parser, _BASE)

    args = _PARSER.parse_args()
    _SUBCOMMANDS[args.command].main(args)


if __name__ == '__main__':
    main()
