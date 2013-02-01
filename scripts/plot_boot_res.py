import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from matplotlib.backends.backend_pdf import PdfPages

from argparse import ArgumentParser
import json

from bisect import bisect


def parse_input():
	"""
	Sets stdout and parses json files from input argument.

	If input is a directory, this function applies a filter to find files in 
	that directory containing a specified substring, then parses all of them.

	This function always returns a list of json-objects (dicts or lists), 
	followed by the args Namespaces, which contains bucket width, output file,
	and mode information.
	"""
	parser = ArgumentParser()
	parser.add_argument("in_file", type=str, help="Input file or directory.")
	parser.add_argument("out_file", type=str, help="Output file.")
	parser.add_argument("-filter", type=str, default="", help= "If input is "+\
						"a directory, then it gets filtered for files with"+\
						"names contain this substring.")
	parser.add_argument("-bucket", type=float, default=.01, help="Fraction "+\
						"of the data to include in each bucket.")
	parser.add_argument("-mode", type=str, choices=["pct","dst","qq"], \
						default="pct", help="Set mode=dst to plot true and "+\
						"bootstrap regret distributions. The default mode="+\
						"pct generates bootstrap percentile vs. true regret "+\
						"cumulative fraction plots. Set mode=qq Q-Q plots of "+\
						"bootstrap mean/median vs true regret")
	parser.add_argument("-axes", type=float, default=[0.,0.], nargs=2, help= \
						"Optional maximum values for the chart's x and y-axes.")
	args = parser.parse_args()

	if os.path.isfile(args.in_file):
		with open(args.in_file) as f:
			return json.load(f), args
	elif os.path.isdir(args.in_file):
		names = filter(lambda fn: args.filter in fn, os.listdir(args.in_file))
		files = map(open, [os.path.join(args.in_file, n) for n in names])
		data = map(lambda f: f.read(), files)
		map(lambda f: f.close(), files)
		return map(json.loads, data), args
	else:
		raise IOError("Input must be a file or directory.")


def get_keys(data):
	variances = sorted(map(float, data.keys()))
	subsamples = sorted(map(int, data[str(variances[0])].keys()))
	resample_count = len(data[str(variances[0])][str(subsamples[0])][0][ \
			"bootstrap"])
	return variances, subsamples, resample_count


def generate_percentiles(data, bucket_size):
	variances, subsamples, resample_count = get_keys(data[0])
	num_buckets = 1. / bucket_size
	bucket_width = resample_count / num_buckets
	percentiles = {v:{s:np.zeros(num_buckets + 2) for s in subsamples} for v \
					in variances}
	for d in data:
		for var in variances:
			for sam in subsamples:
				for eq_data in d[str(var)][str(sam)]:
					eq = np.array(eq_data['profile'])
					regr = eq_data['statistic']
					regr_dstr = sorted(eq_data["bootstrap"])
					regr_pct = bisect(list(regr_dstr[::int(bucket_width)]) + \
										[regr_dstr[-1]], regr)
					percentiles[var][sam][regr_pct:] += 1.
	return percentiles


def plot_percentiles(percentiles, bucket_size, out_file):
	pp = PdfPages(out_file)
	variances = sorted(percentiles)
	subsamples = sorted(percentiles[variances[0]])
	x_axis_points = np.arange(0, 101, 100*bucket_size)
	for i,v in enumerate(variances):
		plt.figure(i)
		plt.xlabel("bootstrap regret distribution percentile")
		plt.ylabel("cumulative fraction of true game regrets")
		plt.title("$\sigma \\sim$" +str(v))
		plt.axis([0, 100, 0, 1])
		plt.plot(x_axis_points, x_axis_points/100., "k--", \
				label="perfect calibration")
		for s in subsamples:
			plt.plot(x_axis_points, percentiles[v][s][:-1] / float( \
					percentiles[v][s][-1]), label=str(s) + " samples")
		plt.legend(loc="lower right", prop={'size':6})
		pp.savefig()
	pp.close()


def generate_distributions(data):
	variances, subsamples, resample_count = get_keys(data[0])
	distributions = {v:{s:[] for s in subsamples} for v in variances}
	distributions[0] = []
	for d in data:
		for var in variances:
			for sam in subsamples:
				for eq_data in d[str(var)][str(sam)]:
					eq = np.array(eq_data['profile'])
					regr = eq_data['regret']
					distributions[0].append(eq_data["statistic"])
					distributions[var][sam].extend(eq_data["bootstrap"])

	for v in variances:
		for s in subsamples:
			distributions[v][s].sort()
	distributions[0].sort()
	return distributions


def plot_distributions(distributions, bucket_pct, axes, out_file):
	x_max, y_max = axes
	pp = PdfPages(out_file)
	variances = sorted(distributions)
	subsamples = sorted(distributions[variances[-1]])
	if x_max == 0.:
		x_max = distributions[0][-1]
	bucket_width = x_max * bucket_pct
	bucket_boundaries = np.arange(0, x_max + bucket_width / 2., bucket_width)
	x_axis_points = np.arange(bucket_width / 2., x_max, bucket_width)
	for i,v in enumerate(variances):
		plt.figure(i)
		plt.xlabel("regret distribution")
		if v == 0:
			plt.title("true game")
			cum_dist = np.array([bisect(distributions[0], b) for b in \
								bucket_boundaries])
			plt.plot(x_axis_points, (cum_dist[1:] - cum_dist[:-1]) / \
					float(cum_dist[-1]), label="true game")
		else:
			plt.title("$\sigma \\approx$" +str(v))
			for s in subsamples:
				cum_dist = np.array([bisect(distributions[v][s], b) for b in \
									bucket_boundaries])
				plt.plot(x_axis_points, (cum_dist[1:] - cum_dist[:-1]) / \
						float(cum_dist[-1]), label=str(s)+" samples")
		plt.legend(loc="upper right", prop={'size':6})
		if y_max != 0.:
			plt.axis([0, x_max, 0, y_max])
		pp.savefig()
	pp.close()


def generate_quantiles(data, bucket_size):
	variances, subsamples, resample_count = get_keys(data[0])
	sam_dsts = {v:{s:[] for s in subsamples} for v in variances}	
	true_dst = []
	for d in data:
		for var in variances:
			for sam in subsamples:
				for eq_data in d[str(var)][str(sam)]:
					eq = np.array(eq_data['profile'])
					regr = eq_data['stastic']
					true_dst.append(eq_data["statistic"])
					sam_dsts[var][sam].extend(eq_data["bootstrap"])
	
	quantiles = {v:{s:[] for s in subsamples} for v in variances}
	true_dst.sort()
	quantiles[0] = [true_dst[int(i*len(true_dst))] for i in \
					np.arange(bucket_size,1,bucket_size)]
	for var in variances:
		for sam in subsamples:
			sam_dsts[var][sam].sort()
			for i in np.arange(bucket_size,1,bucket_size):
				quantiles[var][sam].append(sam_dsts[var][sam][int(i * \
												len(sam_dsts[var][sam]))])
	return quantiles


def plot_quantiles(quantiles, bucket_size, out_file):
	pp = PdfPages(out_file)
	variances = sorted(quantiles)
	subsamples = sorted(quantiles[variances[-1]])
	for i,v in enumerate(variances):
		if v == 0:
			continue
		plt.figure(i)
		plt.xlabel("true regret distribution")
		plt.ylabel("bootstrap regret distribution")
		plt.title("$\sigma \\sim$" +str(v))
		for s in subsamples:
			plt.plot(quantiles[0], quantiles[v][s], label=str(s) + " samples")
		plt.legend(loc="lower right", prop={'size':6})
		pp.savefig()
	pp.close()


if __name__ == "__main__":
	data, args = parse_input()
	if args.mode == "pct":
		percentiles = generate_percentiles(data, args.bucket)
		plot_percentiles(percentiles, args.bucket, args.out_file)
	elif args.mode == "dst":
		distributions = generate_distributions(data)
		plot_distributions(distributions, args.bucket, args.axes, args.out_file)
	elif args.mode == "qq":
		quantiles = generate_quantiles(data, args.bucket)
		plot_quantiles(quantiles, args.bucket, args.out_file)
