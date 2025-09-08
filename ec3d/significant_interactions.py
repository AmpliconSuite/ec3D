"""
Identify significant interactions in an ecDNA Hi-C matrix
"""
import os
import sys
import time
import argparse
import copy
import numpy as np
import networkx as nx
import community

from sklearn.metrics import euclidean_distances
from scipy.stats import poisson, nbinom
from statsmodels.stats.multitest import multipletests

try:
	from ec3d.util import create_logger
except:
	from util import create_logger

def identify_significant_interactions(output_prefix, matrix=None, pval_cutoff=0.05, model='global_nb', 
									  padding='average', genomic_distance_model='circular', significant_interactions=None, 
									  structure=None, annotation=None, max_pooling=False, exclude=None, log_fn=None):
	if model not in ['distance_ratio', 'local', 'global_poisson', 'global_nb']:
		raise ValueError(f'significant_interactions.py: The model {model} is not one of the choices: [\'distance_ratio\', \'local\', \'global_poisson\', \'global_nb\']')
	if padding not in ['zero', 'average', 'cyclic']:
		raise ValueError(f'significant_interactions.py: The padding {padding} is not one of the choices: [\'zero\', \'average\', \'cyclic\']')
	if genomic_distance_model not in ['circular', 'linear', 'reference']:
		raise ValueError(f'significant_interactions.py: The genomic_distance_model {genomic_distance_model} is not one of the choices: [\'circular\', \'linear\', \'reference\']')
	if not matrix and not significant_interactions:
		raise ValueError("significant_interactions.py: Please input either Hi-C matrix or significant interactions.")

	"""
	Set up logging
	"""
	print("Identifying significant interactions ...")
	start_time = time.time()
	if not log_fn:
		log_fn = output_prefix + "_significant_interaction.log"
	logger = create_logger('significant_interactions.py', log_fn)
	logger.info("Python version " + sys.version + "\n")
	function_param = f'identify_significant_interactions(output_prefix=\'{output_prefix}\', matrix=\'{matrix}\', pval_cutoff={pval_cutoff}, model=\'{model}\', padding=\'{padding}\', genomic_distance_model=\'{genomic_distance_model}\', significant_interactions=\'{significant_interactions}\', structure=\'{structure}\', annotation=\'{annotation}\', max_pooling={max_pooling}, exclude={exclude}, log_fn=\'{log_fn}\')'
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + function_param)
	"""
	Load Hi-C matrix
	"""
	data = np.array([])
	if matrix:
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Will identify significant interactions in expanded matrix.")
		if matrix.endswith(".txt"):
			data = np.loadtxt(matrix)
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix without duplication, in txt format.")
		elif matrix.endswith(".npy"):
			data = np.load(matrix)
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix without duplication, in npy format.")
		else:
			raise OSError("Input matrix must be in *.txt or *.npy format.")

	"""
	Load annotation file
	"""
	bins = dict()
	res = -1
	if annotation:
		fp = open(annotation, 'r')
		for line in fp:
			s = line.strip().split('\t')
			if res < 0:
				res = int(s[2]) - int(s[1])
			for i in range(3, len(s)):
				bins[int(s[i])] = [s[0], int(s[1]) // res]
		fp.close()
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix annotations.")
	
	"""
	Estimating mu and alpha at each genomic distance
	"""
	N = data.shape[0]
	params_est = dict()
	if matrix and (model == 'global_poisson' or model == 'global_nb'):
		if genomic_distance_model == 'circular':
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
		elif genomic_distance_model == 'linear':
			ld, d = 1, 1
			interaction_freqs = []
			for i in range(1, N): # Equal occupancy binning
				for j in range(N - i):
					interaction_freqs.append(data[j][i + j])
				if len(interaction_freqs) > 0.5 * N:
					q25, q75 = np.percentile(interaction_freqs, [25 ,75])
					interaction_freqs = [c for c in interaction_freqs if 2.5 * q25 - 1.5 * q75 <= c <= 2.5 * q75 - 1.5 * q25]
					for d_ in range(d, i + 1):
						params_est[d_] = [np.mean(interaction_freqs), np.var(interaction_freqs)]
					interaction_freqs = []
					ld = d
					d = i + 1
			if len(interaction_freqs) > 0:
				interaction_freqs = []
				for i in range(ld, N):
					for j in range(N - i):
						interaction_freqs.append(data[j][i + j])
				q25, q75 = np.percentile(interaction_freqs, [25 ,75])
				interaction_freqs = [c for c in interaction_freqs if 2.5 * q25 - 1.5 * q75 <= c <= 2.5 * q75 - 1.5 * q25]
				for d_ in range(ld, N):
					params_est[d_] = [np.mean(interaction_freqs), np.var(interaction_freqs)]
		else:
			if not annotation:
				raise ValueError("Annotation file is required.")
			for i in range(N):
				for j in range(i + 1, N):
					d = N
					if bins[i][0] == bins[j][0]:
						d = min(d, abs(bins[i][1] - bins[j][1]))
					try:
						if d > 0:
							params_est[d].append(data[i][j])
					except:
						if d > 0:
							params_est[d] = [data[i][j]]
			dist_partitions = []
			partition, nbins = [], 0
			for d_ in sorted(params_est.keys())[:-1]: # Equal occupancy binning
				partition.append(d_)
				nbins += len(params_est[d_])
				if nbins > 0.5 * N:
					dist_partitions.append(partition)
					partition = []
					nbins = 0
			ldist = sorted(params_est.keys())[-1]
			nbins += len(params_est[ldist])
			if nbins > 0.5 * N:
				dist_partitions.append(partition + [ldist])
			else:
				dist_partitions[-1] += (partition + [ldist])
			for partition in dist_partitions:
				interaction_freqs = []
				for d in partition:
					interaction_freqs += params_est[d]
				q25, q75 = np.percentile(interaction_freqs, [25 ,75])
				interaction_freqs = [c for c in interaction_freqs if 2.5 * q25 - 1.5 * q75 <= c <= 2.5 * q75 - 1.5 * q25]
				for d in partition:
					params_est[d] = [np.mean(interaction_freqs), np.var(interaction_freqs)]
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Estimated the mean and variance of interactions at each genomic distance.")
	elif matrix and model == 'distance_ratio':
		if not (structure and annotation):
			raise ValueError("3D structure and annotation files are required to compute the distance ratio.")
		X = np.array([])
		if structure.endswith(".txt"):
			X = np.loadtxt(structure)
		elif structure.endswith(".npy"):
			X = np.load(structure)
		else:
			raise OSError("Input matrix must be in *.txt or *.npy format.")
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA 3D structure.")

		dis = euclidean_distances(X)
		if genomic_distance_model == 'circular':
			for i in range(N):
				for j in range(i + 1, N):
					d = min(abs(i - j), N - abs(i - j))
					try:
						params_est[d].append(d / dis[i][j])
					except:
						params_est[d] = [d / dis[i][j]]
			for d in params_est.keys():
				q25, q75 = np.percentile(params_est[d], [25 ,75])
				params_est[d] = [c for c in params_est[d] if 2.5 * q25 - 1.5 * q75 <= c <= 2.5 * q75 - 1.5 * q25]
				params_est[d] = [np.mean(params_est[d]), np.var(params_est[d])]
		else: # Reference genomic distance
			for i in range(N):
				for j in range(i + 1, N):
					d = N
					if bins[i][0] == bins[j][0]:
						d = min(d, abs(bins[i][1] - bins[j][1]))
					try:
						if d > 0:
							params_est[d].append(d / dis[i][j])
					except:
						if d > 0:
							params_est[d] = [d / dis[i][j]]
			dist_partitions = []
			partition, nbins = [], 0
			for d_ in sorted(params_est.keys())[:-1]: # Equal occupancy binning
				partition.append(d_)
				nbins += len(params_est[d_])
				if nbins > 0.5 * N:
					dist_partitions.append(partition)
					partition = []
					nbins = 0
			nbins += len(params_est[N])
			if nbins > 0.5 * N:
				dist_partitions.append(partition + [N])
			else:
				dist_partitions[-1] += (partition + [N])
			for partition in dist_partitions:
				interaction_freqs = []
				for d in partition:
					interaction_freqs += params_est[d]
				q25, q75 = np.percentile(interaction_freqs, [25 ,75])
				interaction_freqs = [c for c in interaction_freqs if 2.5 * q25 - 1.5 * q75 <= c <= 2.5 * q75 - 1.5 * q25]
				for d in partition:
					params_est[d] = [np.mean(interaction_freqs), np.var(interaction_freqs)]
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Estimated the mean and variance of interactions at each genomic distance.")
	
	"""
	Compute p-values
	"""
	si = dict()
	if matrix and model == 'local':
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Padding ecDNA Hi-C matrix of size %d * %d." %(N, N))
		pl = 10 # pad matrices with 10 pixels on each side
		if padding == 'zero' or padding == 'cyclic':
			data = np.pad(data, ((pl, pl), (pl, pl)))
		else:
			data = np.pad(data, ((pl, pl), (pl, pl)), constant_values = np.average(data))
		if padding == 'cyclic':
			for i in range(pl):
				for j in range(pl, pl + N):
					data[j][i] = data[j][N + i]
					data[j][N + pl + i] = data[j][pl + i]
					data[i][j] = data[N + i][j]
					data[N + pl + i][j] = data[pl + i][j]
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Matrix size after padding: %d * %d." %(data.shape[0], data.shape[1]))
		
		for (p, w) in [(1, 3), (2, 5), (4, 7)]:
			pvals_h, pvals_v, pvals_ll, pvals_donut = [], [], [], []
			for i in range(pl, N + pl):
				for j in range(i + w + 1, N + pl):
					hneighbor, vneighbor, llneighbor, donutneighbor = [], [], [], []
					for ii in range(i - w, i + w):
						for jj in range(j - 1, j + 1):
							if ii < i - p or ii > i + p:
								vneighbor.append(data[ii][jj])
					for ii in range(i - 1, i + 1):
						for jj in range(j - w, j + w):
							if jj < j - p or jj > j + p:
								hneighbor.append(data[ii][jj])
					for ii in range(i - w, i - 1):
						for jj in range(j + 1, j + w):
							if (ii < i - p) or (jj > j + p):
								llneighbor.append(data[ii][jj])
					for ii in range(i - w, i + w):
						for jj in range(j - w, j + w):
							if (ii != i) and (jj != j) and ((ii > i + p) or (ii < i - p) or (jj > j + p) or (jj < j - p)):
								donutneighbor.append(data[ii][jj])
					mu_v = np.average(vneighbor)
					mu_h = np.average(hneighbor)
					mu_ll = np.average(llneighbor)
					mu_donut = np.average(donutneighbor)
					pval_v = 1 - poisson.cdf(data[i][j], mu_v)
					pval_h = 1 - poisson.cdf(data[i][j], mu_h)
					pval_ll = 1 - poisson.cdf(data[i][j], mu_ll)
					pval_donut = 1 - poisson.cdf(data[i][j], mu_donut)
					if np.isnan(pval_v):
						pval_v = 1.0
					if np.isnan(pval_h):
						pval_h = 1.0
					if np.isnan(pval_ll):
						pval_ll = 1.0
					if np.isnan(pval_donut):
						pval_donut = 1.0
					pvals_v.append(pval_v)
					pvals_h.append(pval_h)
					pvals_ll.append(pval_ll)
					pvals_donut.append(pval_donut)
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Computed the P-values for all interactions with p = %d and w = %d." %(p, w))
			
			qvals_v = multipletests(pvals_v, alpha = 0.05, method = 'fdr_bh')[1]
			qvals_h = multipletests(pvals_h, alpha = 0.05, method = 'fdr_bh')[1]
			qvals_ll = multipletests(pvals_ll, alpha = 0.05, method = 'fdr_bh')[1]
			qvals_donut = multipletests(pvals_donut, alpha = 0.05, method = 'fdr_bh')[1]
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Corrected the P-values with Benjamini-Hochberg procedure.")

			qi = 0
			for i in range(pl, N + pl):
				for j in range(i + w + 1, N + pl):
					pval = max(pvals_v[qi], pvals_h[qi], pvals_ll[qi], pvals_donut[qi])
					qval = max(qvals_v[qi], qvals_h[qi], qvals_ll[qi], qvals_donut[qi])
					if qval <= pval_cutoff:
						try:
							si[(i - pl, j - pl)][0] = min(pval, si[(i - pl, j - pl)][0])
							si[(i - pl, j - pl)][1] = min(qval, si[(i - pl, j - pl)][1])
						except:
							si[(i - pl, j - pl)] = [data[i][j], pval, qval]
					qi += 1
	elif matrix and (model == 'global_poisson' or model == 'global_nb'):
		"""
		if not matrix:
			print("Please input the Hi-C matrix, in *.txt or *.npy format.")
			exit(1)
		"""
		pvals = []
		if model == "global_poisson":
			if genomic_distance_model == 'circular':
				for i in range(N):
					for j in range(i + 1, N):
						d = min(j - i, N - j + i)
						mu = params_est[d][0]
						pval = 1 - poisson.cdf(data[i][j], mu)
						if np.isnan(pval):
							pval = 1.0
						pvals.append(pval)
			elif genomic_distance_model == 'linear':
				for i in range(N):
					for j in range(i + 1, N):
						d = j - i
						mu = params_est[d][0]
						pval = 1 - poisson.cdf(data[i][j], mu)
						if np.isnan(pval):
							pval = 1.0
						pvals.append(pval)
			else:
				for i in range(N):
					for j in range(i + 1, N):
						d = N
						if bins[i][0] == bins[j][0]:
							d = min(d, abs(bins[i][1] - bins[j][1]))
						pval = 1.0
						if d > 0:
							mu = params_est[d][0]
							pval = 1 - poisson.cdf(data[i][j], mu)
						if np.isnan(pval):
							pval = 1.0
						pvals.append(pval)
		else:
			if genomic_distance_model == 'circular':
				for i in range(N):
					for j in range(i + 1, N):
						d = min(j - i, N - j + i)
						mu = params_est[d][0]
						sigma2 = params_est[d][1]
						n = (mu ** 2) / (sigma2 - mu)
						p = mu / sigma2
						pval = 1 - nbinom.cdf(data[i][j], n, p)
						if np.isnan(pval):
							pval = 1.0
						pvals.append(pval)
			elif genomic_distance_model == 'linear':
				for i in range(N):
					for j in range(i + 1, N):
						d = j - i
						mu = params_est[d][0]
						sigma2 = params_est[d][1]
						n = (mu ** 2) / (sigma2 - mu)
						p = mu / sigma2
						pval = 1 - nbinom.cdf(data[i][j], n, p)
						if np.isnan(pval):
							pval = 1.0
						pvals.append(pval)
			else:
				for i in range(N):
					for j in range(i + 1, N):
						d = N
						if bins[i][0] == bins[j][0]:
							d = min(d, abs(bins[i][1] - bins[j][1]))
						pval = 1.0
						if d > 0:
							mu = params_est[d][0]
							sigma2 = params_est[d][1]
							n = (mu ** 2) / (sigma2 - mu)
							p = mu / sigma2
							pval = 1 - nbinom.cdf(data[i][j], n, p)
						if np.isnan(pval):
							pval = 1.0
						pvals.append(pval)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Computed the P-values for all interactions.")
		qvals = multipletests(pvals, alpha = 0.05, method = 'fdr_bh')[1]
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Corrected the P-values with Benjamini-Hochberg procedure.")
		qi = 0
		for i in range(N):
			for j in range(i + 1, N):
				if qvals[qi] <= pval_cutoff:
					si[(i, j)] = [data[i][j], pvals[qi], qvals[qi]]
				qi += 1
	elif matrix:
		assert (model == 'distance_ratio')
		pvals = []
		if genomic_distance_model == 'circular':
			for i in range(N):
				for j in range(i + 1, N):
					d = min(abs(i - j), N - abs(i - j))
					pval = 1.0
					if d > 0:
						mu = params_est[d][0]
						sigma2 = params_est[d][1]
						n = (mu ** 2) / (sigma2 - mu)
						p = mu / sigma2
						pval = 1 - nbinom.cdf(d / dis[i][j], n, p)
					if np.isnan(pval):
						pval = 1.0
					pvals.append(pval)
		else:
			for i in range(N):
				for j in range(i + 1, N):
					d = N
					if bins[i][0] == bins[j][0]:
						d = min(d, abs(bins[i][1] - bins[j][1]))
					pval = 1.0
					if d > 0:
						mu = params_est[d][0]
						sigma2 = params_est[d][1]
						n = (mu ** 2) / (sigma2 - mu)
						p = mu / sigma2
						pval = 1 - nbinom.cdf(d / dis[i][j], n, p)
					if np.isnan(pval):
						pval = 1.0
					pvals.append(pval)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Computed the P-values for all interactions.")
		qvals = multipletests(pvals, alpha = 0.05, method = 'fdr_bh')[1]
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Corrected the P-values with Benjamini-Hochberg procedure.")
		qi = 0
		for i in range(N):
			for j in range(i + 1, N):
				if qvals[qi] <= pval_cutoff:
					si[(i, j)] = [data[i][j], pvals[qi], qvals[qi]]
				qi += 1	

	"""
	Filtering out interactions; compute connected components
	"""
	if matrix and significant_interactions:
		logger.warning("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"Ignoring input significant interactions, use those called in the input matrix.")
	elif significant_interactions:
		for si_fn in significant_interactions:
			fp = open(si_fn, 'r')
			for line in fp:
				s = line.strip().split('\t')
				if s[0] != "bin1":
					i = int(s[0])
					j = int(s[1])
					if (i, j) in si:
						print (i, j)
					si[(min(i, j), max(i, j))] = [float(s[2]), float(s[3]), float(s[4])]
			fp.close()
			print (len(si))
	if exclude:
		del_list = []
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"There are %d significant interactions before filtering." %len(si))
		for i in exclude:
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"Filtering out significant interactions involving bin %d." %i)
			for (i_, j_) in si.keys():
				if i_ == i or j_ == i:
					 del_list.append((i_, j_))
		for (i, j) in del_list:
			if (i, j) in si:
				del si[(i, j)]
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"%d significant interactions remain after filtering." %len(si))
	"""
	si_cc = {(i, j): -1 for (i, j) in si.keys()}
	ccid = 0
	for (i, j) in si_cc.keys():
		if si_cc[(i, j)] == -1:
			L = [(i, j)]
			while len(L) > 0:
				(i_, j_) = L.pop(0)
				if si_cc[(i_, j_)] == -1:
					si_cc[(i_, j_)] = ccid
				if (i_ - 1, j_) in si and si_cc[(i_ - 1, j_)] == -1 and (i_ - 1, j_) not in L:
					L.append((i_ - 1, j_))
				if (i_ + 1, j_) in si and si_cc[(i_ + 1, j_)] == -1 and (i_ + 1, j_) not in L:
					L.append((i_ + 1, j_))
				if (i_, j_ - 1) in si and si_cc[(i_, j_ - 1)] == -1 and (i_, j_ - 1) not in L:
					L.append((i_, j_ - 1))
				if (i_, j_ + 1) in si and si_cc[(i_, j_ + 1)] == -1 and (i_, j_ + 1) not in L:
					L.append((i_, j_ + 1))
			ccid += 1
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"The %d significant interactions form %d connected components." %(len(si), ccid))
	"""
	del_list = set([])
	if max_pooling:
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"There are %d significant interactions before max pooling." %len(si))
		for (i, j) in si.keys():
			if (i + 1, j) in si:
				if si[(i, j)][0] > si[(i + 1, j)][0]:
					del_list.add((i + 1, j))
				if si[(i, j)][0] < si[(i + 1, j)][0]:
					del_list.add((i, j))
			if (i, j + 1) in si:
				if si[(i, j)][0] > si[(i, j + 1)][0]:
					del_list.add((i, j + 1))
				if si[(i, j)][0] < si[(i, j + 1)][0]:
					del_list.add((i, j))
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"%d significant interactions remain after max pooling." %(len(si) - len(del_list)))
	
	"""
	Output the significant interactions to tsv file
	"""
	G = nx.Graph()
	tsv_fn = output_prefix + "_significant_interactions.tsv"
	fp = open(tsv_fn, 'w')
	fp.write('bin1\tbin2\tinteraction\tp_value\tq_value\n')
	for (i, j) in si.keys():
		if i not in G:
			G.add_node(i)
		if j not in G:
			G.add_node(j)
		G.add_edge(i, j)
		if max_pooling:
			if (i, j) not in del_list:
				fp.write("%d\t%d\t%f\t%f\t%f\n" %(i, j, si[(i, j)][0], si[(i, j)][1], si[(i, j)][2]))
		else:
			fp.write("%d\t%d\t%f\t%f\t%f\n" %(i, j, si[(i, j)][0], si[(i, j)][1], si[(i, j)][2]))
	fp.close()
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Wrote significant interactions to %s." %tsv_fn)

	"""
	Cluster significant interactions
	"""
	num_clusters = dict()
	for i in range(100):
		partition = community.best_partition(G)
		nc = len(set(partition.values()))
		try:
			num_clusters[nc][0] += 1
			if community.modularity(partition, G) > community.modularity(num_clusters[nc][1], G):
				num_clusters[nc][1] = partition
		except:
			num_clusters[nc] = [1, partition]
			
	best_nc = list(num_clusters.keys())[0]
	for nc in num_clusters.keys():
		if num_clusters[nc][0] > num_clusters[best_nc][0]:
			best_nc = nc
	best_partition = num_clusters[best_nc][1]
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Clustered bins involved in significant interactions.")

	modularity_score = community.modularity(best_partition, G)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Modularity score of the partition: %f." %modularity_score)
    
	cluster_fn = output_prefix + "_clustered_bins.tsv"
	fp = open(cluster_fn, 'w')
	fp.write('bin\tcluster\n')
	remaining_nodes = set([])
	if len(del_list) > 0:
		for (i, j) in si.keys():
			if (i, j) not in del_list:
				remaining_nodes.add(i)
				remaining_nodes.add(j)
	for node in sorted(best_partition.keys()):
		if max_pooling:
			if node in remaining_nodes:
				fp.write("%d\t%d\n" %(node, best_partition[node]))
		else:
			fp.write("%d\t%d\n" %(node, best_partition[node]))
	fp.close()
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "%d Clusters were detected with Louvain Clustering." %len(set(best_partition.values())))
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Wrote Clustered bins to %s." %cluster_fn)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")
	print('Significant interactions identification is done. Significant interactions are written to %s.' %tsv_fn)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Identify significant interactions from ecDNA.")
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--matrix", help = "Input expanded Hi-C matrix, in *.txt or *.npy format.")
	parser.add_argument("--pval_cutoff", help = "P-value cutoff as significant interactions.", type = float, default = 0.05)
	parser.add_argument("--model", help = "Statistical model used to computet the P-values.", default = "global_nb",
				choices = ['distance_ratio', 'local', 'global_poisson', 'global_nb'])
	# Local: HiCCUPS
	# Global Poisson/Negative Binomial: Significant interactions on each diagonal
	# Distance ratio: Significant interactions wrt spatial distance/genomic distance ratio
	parser.add_argument("--padding", help = "Pad expanded Hi-C matrix with certain values, for HiCCUPS interaction calling.", 
				default = "average", choices = ['zero', 'average', 'cyclic'])
	parser.add_argument("--genomic_distance_model", help = "Model of genomic distance between two bins.", default = "circular",
				choices = ['circular', 'linear', 'reference'])
	parser.add_argument("--significant_interactions", 
				help = "Take significant interactions, in *.tsv format and perform Louvain Clustering.", nargs = '+')
	parser.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format.")
	parser.add_argument("--annotation", help = "Annotation of bins in the input matrix.")
	parser.add_argument("--max_pooling", help = "Only keep significant interactions larger than their neighbors.", action = 'store_true')
	parser.add_argument("--exclude", help = "Exclude significant interactions at given indices.", type = int, nargs = '+')
	parser.add_argument("--log_fn", help = "Name of log file.")
	
	args = parser.parse_args()
	identify_significant_interactions(**vars(args))