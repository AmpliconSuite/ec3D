"""
Identify significant interactions in an ecDNA Hi-C matrix
"""
import os
import sys
import time
import argparse
import logging
import numpy as np
import networkx as nx
import community

from sklearn.metrics import euclidean_distances
from scipy.stats import poisson, nbinom
from statsmodels.stats.multitest import multipletests


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Visualize the 3D structure of ecDNA.")
	parser.add_argument("--matrix", help = "Input expanded Hi-C matrix, in *.txt or *.npy format", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--pval_cutoff", help = "P-value cutoff as significant interactions.", type = float, default = 0.05)
	parser.add_argument("--model", help = "Statistical model used to computet the P-values", default = "negative_binomial")
	parser.add_argument("--filter_interactions_by_percentile", 
				help = "Only keep distant significant interactions greater than the specified percentile wrt genomic distance/spatial distance.", 
				type = int)
	parser.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format.")
	parser.add_argument("--log_fn", help = "Name of log file.")
	start_time = time.time()
	args = parser.parse_args()
	if args.model != "poisson" and args.model != "negative_binomial":
		print ("Please speficy either poisson or negative_binomial in --model.")
		os.abort()

	"""
	Set up logging
	"""
	log_fn = ""
	if not args.log_fn:
		log_fn = args.output_prefix + "_significant_interaction.log"
	else:
		log_fn = args.log_fn
	logging.basicConfig(filename = log_fn, filemode = 'w', level = logging.DEBUG, 
						format = '[%(name)s:%(levelname)s]\t%(message)s')
	logging.info("Python version " + sys.version + "\n")
	commandstring = 'Command line: '
	for arg in sys.argv:
		if ' ' in arg:
			commandstring += '"{}" '.format(arg)
		else:
			commandstring += "{} ".format(arg)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + commandstring)

	"""
	Load Hi-C matrix
	"""
	data = np.array([])
	if args.matrix.endswith(".txt"):
		data = np.loadtxt(args.matrix)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix without duplication, in txt format.")
	elif args.matrix.endswith(".npy"):
		data = np.load(args.matrix)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix without duplication, in npy format.")
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")
	
	"""
	Estimating mu and alpha at each genomic distance
	"""
	N = data.shape[0]
	params_est = dict()
	for i in range(N):
		for j in range(i + 1, N):
			d = min(abs(i - j), N - abs(i - j))
			try:
				params_est[d].append(data[i][j])
			except:
				params_est[d] = [data[i][j]]
	for d in params_est.keys():
		q25, q75 = np.percentile(params_est[d], [25 ,75])
		params_est[d] = [c for c in params_est[d] if 2.5 * q25 - 1.5 * q75 <= c <= 2.5 * q75 - 1.5 * q25]
		params_est[d] = [np.mean(params_est[d]), np.var(params_est[d])]
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Estimated the mean and variance of interactions at each genomic distance.")
	
	"""
	Compute p-values
	"""
	pvals = []
	if args.model == "poisson":
		for i in range(N):
			for j in range(i + 1, N):
				d = min(abs(i - j), N - abs(i - j))
				mu = params_est[d][0]
				pval = 1 - poisson.cdf(data[i][j], mu)
				pvals.append(pval)
	else:
		for i in range(N):
			for j in range(i + 1, N):
				d = min(abs(i - j), N - abs(i - j))
				mu = params_est[d][0]
				sigma2 = params_est[d][1]
				n = (mu ** 2) / (sigma2 - mu)
				p = mu / sigma2
				pval = 1 - nbinom.cdf(data[i][j], n, p)
				if np.isnan(pval):
					pval = 1.0
				pvals.append(pval)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Computed the P-values for all interactions.")
	
	qvals = multipletests(pvals, alpha = 0.05, method = 'fdr_bh')[1]
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Corrected the P-values with Benjamini-Hochberg procedure.")

	"""
	Output the significant interactions to tsv file
	"""
	si = []
	qi = 0
	for i in range(N):
		for j in range(i + 1, N):
			if qvals[qi] <= args.pval_cutoff:
				si.append([i, j, qi])
			qi += 1

	if args.filter_interactions_by_percentile:
		if not args.structure:
			print("3D structure is required to compute spatial distances.")
			os.abort()
		X = np.array([])
		if args.structure.endswith(".txt"):
			X = np.loadtxt(args.structure)
		elif args.structure.endswith(".npy"):
			X = np.load(args.structure)
		else:
			raise OSError("Input matrix must be in *.txt or *.npy format.")
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA 3D structure.")
		
		dis = euclidean_distances(X)
		ratio = []
		for i in range(N):
			for j in range(i + 1, N):
				ratio.append(min(abs(i - j), N - abs(i - j)) / dis[i][j])
		si_thredhold = np.percentile(ratio, args.filter_interactions_by_percentile)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Threshold for distant significant interactions: %f." %si_thredhold)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "There are %d significant interactions before filtering." %len(si))
		si = [si_item for si_item in si if min(abs(si_item[0] - si_item[1]), N - abs(si_item[0] - si_item[1])) / dis[si_item[0]][si_item[1]] >= si_thredhold]
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "There are %d significant interactions after filtering." %len(si))
	
	G = nx.Graph()
	tsv_fn = args.output_prefix + "_significant_interactions.tsv"
	fp = open(tsv_fn, 'w')
	fp.write('bin1\tbin2\tinteraction\tp_value\tq_value\n')
	for si_item in si:
		if si_item[0] not in G:
			G.add_node(si_item[0])
		if si_item[1] not in G:
			G.add_node(si_item[1])
		G.add_edge(si_item[0], si_item[1])
		fp.write("%d\t%d\t%f\t%f\t%f\n" %(si_item[0], si_item[1], data[si_item[0]][si_item[1]], pvals[si_item[2]], qvals[si_item[2]]))
	fp.close()
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Wrote significant interactions to %s." %tsv_fn)

	"""
	Cluster significant interactions
	"""	
	partition = community.best_partition(G)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Clustered bins involved in significant interactions.")

	modularity_score = community.modularity(partition, G)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Modularity score of the partition: %f." %modularity_score)
    
	cluster_fn = args.output_prefix + "_clustered_bins.tsv"
	fp = open(cluster_fn, 'w')
	fp.write('bin\tcluster\n')
	for node in sorted(partition.keys()):
		fp.write("%d\t%d\n" %(node, partition[node]))
	fp.close()
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "%d Clusters were detected with Louvain Clustering." %len(set(partition.values())))
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Wrote Clustered bins to %s." %cluster_fn)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")

  