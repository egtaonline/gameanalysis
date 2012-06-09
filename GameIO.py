#! /usr/bin/env python2.7

from urllib import urlopen
from json import loads, dumps
from xml.dom.minidom import parseString, Document
from xml.parsers.expat import ExpatError
from os.path import exists, splitext
from functools import partial

from RoleSymmetricGame import *


def read(source):
	if isinstance(source, basestring) and exists(source):
		f = open(source)
		data = f.read()
		f.close()
	elif isinstance(source, basestring) and source.startswith("http://"):
		#source is a url
		u = urlopen(source)
		data = u.read()
		u.close()
	elif hasattr(source, 'read'):
		#source is file-like
		data = source.read()
		source.close()
	else:
		#assume source is already a data string or object
		data = source
	json_data = loadJSON(data)
	if json_data != None:
		return readJSON(json_data)
	xml_data = loadXML(data)
	if xml_data != None:
		return readXML(xml_data)
	raise IOError("unrecognized input format")


def loadJSON(data):
	if isinstance(data, list) or isinstance(data, dict):
		return data
	try:
		return loads(data)
	except:
		return None


def loadXML(data):
	if isinstance(data, Document):
		return data
	try:
		return parseString(data)
	except:
		return None


def readJSON(data):
	"""
	Convert loaded json data (list or dict) into GameAnalysis classes.
	"""
	if isinstance(data, list):
		return map(readJSON, data)
	if "profiles" in data:
		return readGameJSON(data)
	if "sample_count" in data:
		return readTestbedProfile(data)
	return readProfile(data)


def readGameJSON(gameJSON):
	if "strategy_array" in gameJSON["roles"][0]:
		return readGameJSON_old(gameJSON)
	elif "strategies" in gameJSON["roles"][0]:
		return readGameJSON_v2(gameJSON)
	else:
		raise IOError("invalid game JSON: " + str(data))


def readGameJSON_v2(gameJSON):
	players = {r["name"] : int(r["count"]) for r in gameJSON["roles"]}
	strategies = {r["name"] : r["strategies"] for r in gameJSON["roles"]}
	roles = list(players.keys())
	profiles = []
	if "profiles" in gameJSON:
		for profileJSON in gameJSON["profiles"]:
			profiles.append(readTestbedProfile(profileJSON))
	return Game(roles, players, strategies, profiles)


def readTestbedProfile(profileJSON):
	profile = {r["name"]:[] for r in profileJSON["roles"]}
	for roleDict in profileJSON["roles"]:
		role = roleDict["name"]
		for strategyDict in roleDict["strategies"]:
			profile[role].append(payoff_data(str(strategyDict[ \
					"name"]), int(strategyDict["count"]), \
					float(strategyDict["payoff"])))
	return profile


def readProfile(profileJSON, game=None):
	if all([isinstance(p, int) for p in r.values() for r in \
			profileJSON.values()]):
		return Profile(profileJSON)
	return game.mixtureArray(profileJSON)


def readGameJSON_old(json_data):
	players = {r["name"] : int(r["count"]) for r in json_data["roles"]}
	strategies = {r["name"] : r["strategy_array"] for r in json_data["roles"]}
	roles = list(players.keys())
	profiles = []
	for profileDict in json_data["profiles"]:
		profile = {r:[] for r in roles}
		prof_strat = {}
		for role_str in profileDict["proto_string"].split("; "):
			role, strategy_str = role_str.split(": ")
			prof_strat[role] = strategy_str.split(", ")
		for roleDict in profileDict["roles"]:
			role = roleDict["name"]
			for strategyDict in roleDict["strategies"]:
				s = strategyDict["name"]
				profile[role].append(payoff_data(str(s), int(prof_strat[role] \
						.count(s)), float(strategyDict["payoff"])))
		profiles.append(profile)
	return Game(roles, players, strategies, profiles)


def readXML(data):
	"""
	Convert loaded xml data (Document) into GameAnalysis classes.
	"""
	gameNode = data.getElementsByTagName("nfg")[0]
	if len(gameNode.getElementsByTagName("player")[0]. \
			getElementsByTagName("action")) > 0:
		return parseStrategicXML(gameNode)
	return parseSymmetricXML(gameNode)


def parseStrategicXML(gameNode):
	strategies = {p.getAttribute('id') : map(lambda s: s.getAttribute('id'), \
			p.getElementsByTagName('action')) for p in \
			gameNode.getElementsByTagName('player')}
	roles = list(strategies.keys())
	counts = {r:1 for r in roles}
	payoffs = []
	for payoffNode in gameNode.getElementsByTagName('payoff'):
		data = {r:[] for r in roles}
		for outcomeNode in payoffNode.getElementsByTagName('outcome'):
			role = outcomeNode.getAttribute('player')
			strategy = outcomeNode.getAttribute('action')
			value = float(outcomeNode.getAttribute('value'))
			data[role].append(payoff_data(strategy, 1, value))
		payoffs.append(data)
	return Game(roles, counts, strategies, payoffs)


def parseSymmetricXML(gameNode):
	roles = ["All"]
	counts= {"All" : len(gameNode.getElementsByTagName("player"))}
	strategies = {"All" : [e.getAttribute("id") for e in \
			gameNode.getElementsByTagName("action")]}
	payoffs = []
	for payoffNode in gameNode.getElementsByTagName("payoff"):
		data = []
		for outcomeNode in payoffNode.getElementsByTagName("outcome"):
			strategy = outcomeNode.getAttribute("action")
			count = int(outcomeNode.getAttribute("count"))
			value = float(outcomeNode.getAttribute("value"))
			data.append(payoff_data(strategy, count, value))
		payoffs.append({"All":data})
	return Game(roles, counts, strategies, payoffs)


def toJSONstr(*objects):
	return dumps(toJSON(*objects), sort_keys=True, indent=2)


def toJSON(*objects):
	if len(objects) > 1:
		return map(toJSON, objects)
	o = objects[0]
	if hasattr(o, 'toJSON'):
		return o.toJSON()
	return o


def toXML(game, filename):
	if len(game.roles) == 1:
		toSymmetricXML(game, filename)
	elif all(map(lambda c: c==1, game.counts.values())):
		toStrategicXML(game, filename)
	else:
		raise NotImplementedError("no EGAT XML spec for role-symmetric games")


def toSymmetricXML(game, filename):
	"""
	Writes game to XML according to the EGAT symmetric game spec.
	Assumes (but doesn't check) that game is symmetric.
	"""
	raise NotImplementedError("TODO")


def toStrategicXML(game, filename):
	"""
	Writes game to XML according to the EGAT strategic game spec.
	Assumes (but doesn't check) that game is not role-symmetric.
	"""
	raise NotImplementedError("TODO")


class io_parser(ArgumentParser):
	def __init__(self, *args, **kwargs):
		ArgumentParser.__init__(self, *args, **kwargs)
		self.add_argument("-input", type=str, default="", help= \
				"Input file. Defaults to stdin.")
		self.add_argument("-output", type=str, default="", help= \
				"Output file. Defaults to stdout.")

	def parse_args(self, *args, **kwargs):
		args = ArgumentParser.parse_args(self, *args, **kwargs)
		if args.input == "":
			args.input = read(sys.stdin.read())
		else:
			i = open(args.input)
			args.input = read(i.read())
			i.close()
		if args.output != "":
			sys.stdout = open(args.output, "w")
		return args


def parse_args():
	parser = io_parser()
	parser.add_argument("-format", choices=["json", "xml"], default="json", \
			help="Output format.")
	return parser.parse_args()


def main():
	args = parse_args()
	if args.format == "json":
		print toJSONstr(args.input)
	elif args.format == "xml":
		print toXML(args.input)


if __name__ == "__main__":
	main()
