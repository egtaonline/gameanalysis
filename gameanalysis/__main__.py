"""Command line module"""
import argparse
import pkgutil
import sys

import gameanalysis
from gameanalysis import script


def create_parser():
    """Create the default parser"""
    modules = [imp.find_module(name).load_module(name) for imp, name, _
               in pkgutil.iter_modules(script.__path__)]
    parser = argparse.ArgumentParser(
        description="""Command line access to the game analysis toolkit.""")
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s {}'.format(gameanalysis.__version__))
    subparsers = parser.add_subparsers(
        title='commands', dest='command', metavar='<command>', help="""The
        commands to execute. Available commands are:""")
    for module in modules:
        subparser = module.add_parser(subparsers)
        subparser.main = module.main
    return parser, subparsers.choices


def amain(*argv):
    """Entry point for game analysis"""
    parser, commands = create_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    else:
        commands[args.command].main(args)


def main():
    """Entry point for game analysis"""
    amain(*sys.argv[1:])


if __name__ == '__main__':
    main()
