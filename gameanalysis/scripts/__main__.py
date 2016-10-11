import argparse
import runpy
import pkgutil
import sys

from gameanalysis import scripts


PACKAGES = [(name, imp.find_module(name).load_module(name).__doc__)
            for imp, name, _
            in pkgutil.iter_modules(scripts.__path__)
            if name != '__main__']
PACKAGE_HELP = ' '.join('`{}` - {}.'.format(n, 'unknown' if d is None else d)
                        for n, d in PACKAGES)

PARSER = argparse.ArgumentParser(prog='ga', description="""Command line access
                                 to the game analysis toolkit""")
SUBCOMMANDS = PARSER.add_subparsers(title='commands', dest='command',
                                    metavar='<command>', help="""The commands
                                    to execute. Available commands are: """ +
                                    PACKAGE_HELP)
SUBCOMMANDS.required = True
for name, _ in PACKAGES:
    SUB_PARSER = SUBCOMMANDS.add_parser(name, add_help=False)


def main():
    args, _ = PARSER.parse_known_args()
    sys.argv.pop(1)
    runpy.run_module('gameanalysis.scripts.' + args.command,
                     run_name='__main__', alter_sys=True)
