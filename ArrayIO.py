from json import load
from xml.dom.minidom import parse, Document
from os.path import exists, splitext
from collections import namedtuple

from GameArray import *

payoff = namedtuple("payoff", "strategy count value")


def readGame(filename):
	assert exists(filename)
	ext = splitext(filename)[-1]
	if ext == '.xml':
		return readXML(filename)
	elif ext == '.json':
		return readJSON(filename)
	else:
		raise IOError("unsupported file type: " + ext)


def readJSON(filename):
	f = open(filename)
	data = load(f)
	f.close()
	counts = {r["name"] : int(r["count"]) for r in data["roles"]}
	strategies = {r["name"] : r["strategy_array"] for r in data["roles"]}
	roles = list(counts.keys())

	payoffs = []
	for profileDict in data["profiles"]:
		profile = {r:[] for r in roles}
		prof_strat = {}
		for role_str in profileDict["proto_string"].split("; "):
			role, strategy_str = role_str.split(": ")
			prof_strat[role] = strategy_str.split(", ")
		for roleDict in profileDict["roles"]:
			role = roleDict["name"]
			for strategyDict in roleDict["strategies"]:
				s = strategyDict["name"]
				profile[role].append(payoff(str(s), int(prof_strat[role] \
						.count(s)), float(strategyDict["payoff"])))
		payoffs.append(profile)
	return Game(roles, counts, strategies, payoffs)


def readXML(filename):
	gameNode = parse(filename).getElementsByTagName("nfg")[0]
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
			data[role].append(payoff(strategy, 1, value))
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
			data.append(payoff(strategy, count, value))
		payoffs.append({"All":data})
	return Game(roles, counts, strategies, payoffs)


def writeJSON(game, filename):
	"""
	Writes game to JSON according to the Testbed role-symmetric game spec.
	"""
	raise NotImplementedError("waiting for final Testbed JSON spec")


def writeXML(game, filename):
	if len(game.roles) == 1:
		writeSymmetricXML(game, filename)
	elif all(map(lambda c: c==1, game.counts.values())):
		writeStrategicXML(game, filename)
	else:
		raise NotImplementedError("no EGAT XML spec for role-symmetric games")


def writeSymmetricXML(game, filename):
	"""
	Writes game to XML according to the EGAT symmetric game spec.
	Assumes (but doesn't check) that game is symmetric.
	"""
	raise NotImplementedError("TODO")

def writeStrategicXML(game, filename):
	"""
	Writes game to XML according to the EGAT strategic game spec.
	Assumes (but doesn't check) that game is not role-symmetric.
	"""
	raise NotImplementedError("TODO")


def readHeader(filename):
	assert exists(filename)
	ext = splitext(filename)[-1]
	if ext == '.xml':
		gameNode = parse(filename).getElementsByTagName("nfg")[0]
		if len(gameNode.getElementsByTagName("player")[0]. \
				getElementsByTagName("action")) > 0:
			strategies = {p.getAttribute('id') : map(lambda s: \
					s.getAttribute('id'), p.getElementsByTagName('action')) \
					for p in gameNode.getElementsByTagName('player')}
			roles = list(strategies.keys())
			counts = {r:1 for r in roles}
		else:
			roles = ["All"]
			counts= {"All" : len(gameNode.getElementsByTagName("player"))}
			strategies = {"All" : [e.getAttribute("id") for e in \
				gameNode.getElementsByTagName("action")]}
	elif ext == '.json':
		f = open(filename)
		data = load(f)
		f.close()
		counts = {r["name"] : int(r["count"]) for r in data["roles"]}
		strategies = {r["name"] : r["strategy_array"] for r in data["roles"]}
		roles = list(counts.keys())
	else:
		raise IOError("unsupported file type: " + ext)
	return Game(roles, counts, strategies)


