"""get help about a specific game analysis command"""
import argparse
import pkgutil
import runpy
import sys
from os import path

from gameanalysis import scripts


PACKAGES = [name for _, name, _
            in pkgutil.iter_modules(scripts.__path__)
            if name != '__main__']

PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Get help
                                 for a command.""")
PARSER.add_argument('command', metavar='<command>', nargs='?',
                    choices=PACKAGES, help="""The command to execute. Available
                    commands are: """ + ', '.join(PACKAGES))


def main():
    args = PARSER.parse_args()
    command = args.command if args.command is not None else '__main__'
    sys.argv = [None, '--help']
    runpy.run_module('gameanalysis.scripts.' + command, run_name='__main__',
                     alter_sys=True)


if __name__ == '__main__':
    main()
