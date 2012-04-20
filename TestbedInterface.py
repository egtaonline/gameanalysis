from json import loads
from string import join

import requests

from GameIO import readGame


class TestbedObject:
	def __init__(self, name, obj_type, name_field, options={}):
		self.auth = {'auth_token':"g2LHz1mEtbysFngwLMCz"}
		self.url = "http://d-108-249.eecs.umich.edu/api/v2/" + obj_type

		data = dict(options)
		data.update(self.auth)

		ID = name #try treating name as an id
		r = requests.get(self.url +"/"+ ID + ".json", data=data)

		if r.status_code != 200: #treat name as a name_field
			r = requests.get(self.url + ".json", data=data)
			ID = filter(lambda s: s[name_field] == name, loads(r.text))[0]["id"]
			r = requests.get(self.url +"/"+ ID + ".json", data=self.auth)

		self.url += "/" + ID
		self.json = r.text
		self.__dict__.update(loads(r.text))

	def update(self):
		r = requests.get(self.url + ".json", data=self.auth)
		sel.json = r.text
		self.__dict__.update(r.text)
		return r.status_code

	def __repr__(self):
		return join(map(lambda p: str(p[0]) + ": " + str(p[1]), \
				self.__dict__.items()), "\n")


class TestbedScheduler(TestbedObject):
	def __init__(self, name):
		TestbedObject.__init__(self, name, "generic_schedulers", "name")

	def addProfile(self, profile, samples):
		data = {'sample_count':samples, 'profile_name':str(profile)}
		data.update(self.auth)
		r = requests.post(self.url + "/add_profile.json", data=data)
		return r.status_code

	def update(self, **kwds):
		assert all((k in modifiable_attributes for k in kwds)), \
				"invalid parameter for update"
		self.__dict__.update(kwds)
		data = {"scheduler":kwds}
		data.update(self.auth)
		r = requests.put(self.url + ".json", data=data)
		return r.status_code


class TestbedSimulator(TestbedObject):
	def __init__(self, name):
		TestbedObject.__init__(self, name, "simulators", "simulator_fullname")

	def updateRoles(self):
		r = requests.get(self.url + ".json", data=self.auth)
		self.roles = loads(r.text)["roles"]
		return r.status_code

	def hasRole(self, role):
		return any([r["name"] == role for r in self.roles])

	def hasStrategy(self, role, strategy):
		return self.hasRole(role) and strategy in filter(lambda r: \
				r["name"] == role, self.roles)[0]["strategies"]

	def addRole(self, role):
		data = {"role":role}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "add_role.json", data=data)
		self.updateRoles()
		return r.status_code

	def addStrategy(self, role, strategy):
		assert self.hasRole(role), "no such role: " + role
		data = {"role":role, "strategy":strategy}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "add_strategy.json", data=data)
		self.updateRoles()
		return r.status_code

	def removeRole(self, role):
		data = {"role":role}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "remove_role.json", data=data)
		self.updateRoles()
		return r.status_code

	def removeStrategy(self, role, strategy):
		assert self.hasRole(role), "no such role: " + role
		data = {"role":role, "strategy":strategy}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "remove_strategy.json", data=data)
		self.updateRoles()
		return r.status_code


class TestbedGame(TestbedObject):
	def __init__(self, name, full=False):
		TestbedObject.__init__(self, name, "games", "name", {"full":full})
		self.game = readGame(self.json)

	def update(self):
		TestbedObject.update(self)
		self.game = readJSON(self.json)


modifiable_attributes = ["process_memory", "name", "parameter_hash", "active", \
						"time_per_sample", "nodes", "samples_per_simulation"]


def create_scheduler(name, simulator_id="4f7f469c4a98064ef3000001", \
		time_per_sample=15, samples_per_simulation=10, parameter_hash={ \
		"sims_per_sample":10.0, "events":10000.0, "def_alpha":1.0, "def_beta":\
		1.0, "rate_alpha":2.0, "min_value":1.0, "max_value":2.0, "min_cost":\
		1.0, "max_cost":1.0, "price":"cost", "social_network":"EmptyGraph", \
		"def_samples":"inf", "num_banks":1.0, "bank_policy":"agents2_banks10"},\
		active=True, nodes=1):
	data = {"scheduler":{"name":name, "simulator_id":simulator_id, \
			"time_per_sample":time_per_sample, "samples_per_simulation":\
			samples_per_simulation, "parameter_hash":parameter_hash, "active":\
			active, "nodes":nodes}, "auth_token":"g2LHz1mEtbysFngwLMCz"}
	r = requests.post("http://d-108-249.eecs.umich.edu/api/v2/generic_" + \
			"schedulers.json", data=data)
	return r.status_code
