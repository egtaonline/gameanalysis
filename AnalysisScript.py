#!/usr/local/bin/python2.7

from GameIO import *
from GameAnalysis import *

from sys import argv

g = readGame(argv[1])
g.popitem()
print len(cliques(g))
