#! /usr/bin/env python2.7

from urllib import urlopen
from json import loads, dumps
from xml.dom.minidom import parseString, Document

from BasicFunctions import flatten, one_line
from RoleSymmetricGame import *


def read(source):
	if isinstance(source, basestring):
		try: #assume source is a filename or url
			u = urlopen(source)
			data = u.read()
			u.close()
		except: #assume source is already a data string
			data = source
	elif hasattr(source, 'read'): #source is file-like
		data = source.read()
		source.close()
	else:
		raise IOError(one_line("could not read source: " + str(source), 71))
	return detect_and_load(data)


def detect_and_load(data):
	json_data = loadJSON(data)
	if json_data != None:
		return readJSON(json_data)
	xml_data = loadXML(data)
	if xml_data != None:
		return readXML(xml_data)
	raise IOError(one_line("could not detect format of data: " + str(data), 71))


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
	type(data)
	if isinstance(data, list):
		return map(readJSON, data)
	if "profiles" in data:
		return readGameJSON(data)
	if "sample_count" in data:
		return readTestbedProfile(data)
	if "type" in data and data["type"] == "GA_Profile":
		return readProfile(data)
	raise IOError(one_line("no GameAnalysis class found in JSON: " + \
			str(data), 71))


def readGameJSON(gameJSON):
	if "strategies" in gameJSON["roles"][0]:
		return readGameJSON_v2(gameJSON)
	elif "strategy_array" in gameJSON["roles"][0]:
		return readGameJSON_old(gameJSON)
	else:
		raise IOError(one_line("invalid game JSON: " + str(data), 71))


def readGameJSON_v2(gameJSON):
	players = {r["name"] : int(r["count"]) for r in gameJSON["roles"]}
	strategies = {r["name"] : r["strategies"] for r in gameJSON["roles"]}
	roles = list(players.keys())
	profiles = []
	if "profiles" in gameJSON:
		for profileJSON in gameJSON["profiles"]:
			profiles.append(readTestbedProfile(profileJSON))
	return Game(roles, players, strategies, profiles)


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


def readTestbedProfile(profileJSON):
	profile = {r["name"]:[] for r in profileJSON["roles"]}
	for roleDict in profileJSON["roles"]:
		role = roleDict["name"]
		for strategyDict in roleDict["strategies"]:
			profile[role].append(payoff_data(str(strategyDict[ \
					"name"]), int(strategyDict["count"]), \
					float(strategyDict["payoff"])))
	return profile


def readProfile(profileJSON):
	try:
		return Profile(profileJSON["data"])
	except KeyError:
		return Profile(profileJSON)


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
	return dumps(toJSONobj(objects), sort_keys=True, indent=2)


def toJSONobj(obj):
	if isinstance(obj, (list, tuple)):
		return map(toJSONobj, obj)
	if hasattr(obj, 'toJSON'):
		return obj.toJSON()
	if isinstance(obj, dict):
		return {k:toJSONobj(v) for k,v in obj.items()}
	return loads(dumps(obj))


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
		a = ArgumentParser.parse_args(self, *args, **kwargs)
		if a.input == "":
			a.input = read(sys.stdin.read())
		else:
			i = open(a.input)
			a.input = read(i.read())
			i.close()
		if a.output != "":
			sys.stdout = open(a.output, "w")
		return a


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
