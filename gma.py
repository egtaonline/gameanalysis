#! /usr/bin/env python3
import argparse

from gameanalysis import randgames

# TODO add a way to run tests from here


class HelpPrinter(object):
    '''Handles printing help with the help command'''
    @staticmethod
    def command(args, _):
        parser = argparse.ArgumentParser()
        parser.add_argument('command', nargs='?', choices=COMMANDS.keys(),
                            help='''The command to get help on''')

        args = parser.parse_args(args)
        if args.command is None:
            PARSER.parse_args(['-h'])
        else:
            COMMANDS[args.command].command(['-h'], args.command)


COMMANDS = {
    'rand': randgames,
    'help': HelpPrinter
}

# TODO: Help should probably have the main parser. Instead, this should look
# for the first argument to parse, and if failing that, call help by default
# with the appropriate arguments. OR something...

PARSER = argparse.ArgumentParser(description='''This script is way you call all
game analysis functions''')
PARSER.add_argument('command', choices=COMMANDS.keys(),
                    help='''The game analysis function to run''')


def main():
    args, extra = PARSER.parse_known_args()
    COMMANDS[args.command].command(extra, args.command)

if __name__ == '__main__':
    main()
