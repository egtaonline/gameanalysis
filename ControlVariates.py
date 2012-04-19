import numpy as np

def adjustedPayoffs(source):
	x_by_id = extractXData(source)
	y_by_id = extractYData(source)
	adjusted_payoff = {}
	for pid, role_hash in y_by_id.items():
		x = x_by_id[pid]
		adjusted_role_hash = {}
		for role, strategy_hash in role_hash.items():
			adjusted_role_hash[role] = {strategy: adjustPayoff(x, payoff_array) \
				for strategy, payoff_array in strategy_hash.items()}
		adjusted_payoff[pid] = adjusted_role_hash
	return adjusted_payoff
			

def adjustPayoff(x, y):
	 return np.linalg.lstsq(np.array(x), np.array(y))[0][0]

def extractYData(source):
	y_by_id = {}
	for profile in source["profiles"]:
		role_dict = {}
		for role in profile["roles"]:
			strategy_dict = {strategy["name"]: \
				payoffSamples(profile, role, strategy) \
				for strategy in role["strategies"]}
			role_dict[role["name"]] = strategy_dict
		y_by_id[profile["id"]] = role_dict
	return y_by_id


def extractXData(source):
	x_by_id = {}
	for profile in source["profiles"]:
		x_by_id[profile["id"]] = [featureSamples(source, sample) \
			for sample in profile["sample_records"]]
	return x_by_id
	
	
def payoffSamples(profile, role, strategy):
	return [sample["payoffs"][role["name"]][strategy["name"]] \
		for sample in profile["sample_records"]]
		

def featureSamples(source, sample):
	x_row = [1]
	for feature in source["features"]:
		if feature["name"] in sample["features"]:
			x_row.append(sample["features"][feature["name"]]-feature["expected_value"])
	return x_row