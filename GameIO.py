from json import load
from xml.dom.minidom import parse, Document
from os.path import exists, splitext

from RoleSymmetricGame import *

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

	payoffs = {}
	for profileDict in data["profiles"]:
		profile = {}
		for role_str in profileDict["proto_string"].split("; "):
			role, strategy_str = role_str.split(": ")
			profile[role] = SymmetricProfile(strategy_str.split(", "))
		payoffs[Profile(profile)] = {r["name"]: {s["name"]:s["payoff"] \
				for s in r["strategies"]} for r in profileDict["roles"]}
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
	payoffs = {}
	for payoffNode in gameNode.getElementsByTagName('payoff'):
		str_prof = {}
		payoff = {r:{} for r in roles}
		for outcomeNode in payoffNode.getElementsByTagName('outcome'):
			role = outcomeNode.getAttribute('player')
			strategy = outcomeNode.getAttribute('action')
			value = float(outcomeNode.getAttribute('value'))
			payoff[role][strategy] = value
			str_prof[role] = SymmetricProfile([strategy])
		payoffs[Profile(str_prof)] = payoff
	return Game(roles, counts, strategies, payoffs)


def parseSymmetricXML(gameNode):
	roles = ["All"]
	counts= {"All" : len(gameNode.getElementsByTagName("player"))}
	strategies = {"All" : [e.getAttribute("id") for e in \
			gameNode.getElementsByTagName("action")]}
	payoffs = {}
	for payoffNode in gameNode.getElementsByTagName("payoff"):
		sym_prof = []
		payoff = {"All":{}}
		for outcomeNode in payoffNode.getElementsByTagName("outcome"):
			action = outcomeNode.getAttribute("action")
			count = int(outcomeNode.getAttribute("count"))
			value = float(outcomeNode.getAttribute("value"))
			sym_prof.extend([action] * count)
			payoff["All"][action] = value
		payoffs[Profile({"All":SymmetricProfile(sym_prof)})] = payoff
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
	role = game.roles[0]
	doc = Document()
	gameNode = doc.createElement("nfg")
	doc.appendChild(gameNode)
	playerSetNode = doc.createElement("players")
	gameNode.appendChild(playerSetNode)
	for i in range(game.counts[role]):
		playerNode = doc.createElement("player")
		playerNode.setAttribute("id", "p" + str(i))
		playerSetNode.appendChild(playerNode)
	actionSetNode = doc.createElement("actions")
	gameNode.appendChild(actionSetNode)
	for strategy in game.strategies[role]:
		actionNode = doc.createElement("action")
		actionNode.setAttribute("id", str(strategy))
		actionSetNode.appendChild(actionNode)
	payoffSetNode = doc.createElement("payoffs")
	gameNode.appendChild(payoffSetNode)
	for profile in game:
		payoffNode = doc.createElement("payoff")
		payoffSetNode.appendChild(payoffNode)
		for strategy in profile[role].getStrategies():
			outcomeNode = doc.createElement("outcome")
			outcomeNode.setAttribute("action", str(strategy))
			outcomeNode.setAttribute("count", \
					str(profile[role].count(strategy)))
			outcomeNode.setAttribute("value", str(game.getPayoff( \
					profile, role, strategy)))
			payoffNode.appendChild(outcomeNode)
	file = open(filename, 'w')
	file.write(doc.toprettyxml())
	file.close()


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


