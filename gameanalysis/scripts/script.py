import sys
import argparse

from gameanalysis.scripts import nash
from gameanalysis.scripts import reduction
from gameanalysis.scripts import subgames
from gameanalysis.scripts import gamegen
from gameanalysis.scripts import regret
from gameanalysis.scripts import dominance
from gameanalysis.scripts import convert
from gameanalysis.scripts import payoff


_PARSER = argparse.ArgumentParser(prog='ga', description="""Command line access
                                  to the game analysis toolkit""")
_PARSER.add_argument('--input', '-i', metavar='<input-file>',
                     default=sys.stdin, type=argparse.FileType('r'),
                     help="""Input file for script.  (default: stdin)""")
_PARSER.add_argument('--output', '-o', metavar='<output-file>',
                     default=sys.stdout, type=argparse.FileType('w'),
                     help="""Output file for script. (default: stdout)""")
_SUBPARSERS = _PARSER.add_subparsers(title='Subcommands', dest='command',
                                     help="""The specific aspect of the toolkit
                                     to interact with. See each possible
                                     command for help.""")
_SUBPARSERS.required = True


class help(object):

    @staticmethod
    def update_parser(parser):
        parser.add_argument('subcommand', metavar='command', nargs='?',
                            help="""Command to get help on""")

    @staticmethod
    def main(args):
        if args.subcommand is None:
            parser = _PARSER
        else:
            parser = _SUBPARSERS.choices[args.subcommand]
        parser.print_help()


_SUBCOMMANDS = {
    'nash': nash,
    'red': reduction,
    'sub': subgames,
    'reg': regret,
    'dom': dominance,
    'gen': gamegen,
    'conv': convert,
    'pay': payoff,
    'help': help,
}


def main():
    for name, module in _SUBCOMMANDS.items():
        parser = _SUBPARSERS.add_parser(name)
        module.update_parser(parser)

    args = _PARSER.parse_args()
    _SUBCOMMANDS[args.command].main(args)


if __name__ == '__main__':
    main()
