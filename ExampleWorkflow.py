#! /usr/bin/env python2.7

import sys
from tempfile import NamedTemporaryFile
from json import loads

from BasicFunctions import call
from GameIO import read, to_JSON_str


if __name__ == "__main__":
	STDIN = sys.stdin.read()
	reduced_game = call("./Reductions.py HR 4", STDIN)
	subgames = call("./Subgames.py", reduced_game)
	equilibria = call("./Nash.py -r 0.1", subgames)
	tmp = NamedTemporaryFile()
	tmp.write(equilibria)
	tmp.flush()
	import Regret as R
	print map(type, read(open(tmp.name).read()))
	regrets = call("./Regret.py " + tmp.name, reduced_game)
	tmp.close()

#print to_JSON_str(equilibria)
#print read(to_JSON_str(equilibria))

#print len(read(subgames))
#print map(len, read(equilibria))
#print loads(regrets)
#
#print ",\n".join(map(str, zip(map(lambda g: g.strategies, read(subgames)), \
#		read(equilibria), loads(regrets))))
