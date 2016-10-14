import argparse
import pkgutil

from gameanalysis import scripts


def create_parser():
    modules = [imp.find_module(name).load_module(name) for imp, name, _
               in pkgutil.iter_modules(scripts.__path__)]
    parser = argparse.ArgumentParser(prog='ga', description="""Command line
                                     access to the game analysis toolkit.""")
    subparsers = parser.add_subparsers(title='commands', dest='command',
                                       metavar='<command>', help="""The
                                       commands to execute. Available commands
                                       are:""")
    for module in modules:
        subparser = module.add_parser(subparsers)
        subparser.main = module.main
    return parser, subparsers.choices


def main():
    parser, commands = create_parser()
    args = parser.parse_args()
    commands[args.command].main(args)
