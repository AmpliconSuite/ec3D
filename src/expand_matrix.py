import sys
import argparse
import warnings
import numpy as np
import logging
import time
import autograd.numpy as auto_np

from sklearn.metrics import euclidean_distances
from scipy import optimize
from scipy.special import loggamma, digamma
from autograd import grad
from iced import normalization


def c_obj(x, X, N, S = None, C_dup = None, alpha = -3.0, beta = 1.0):
	C1 = x.reshape(N, -1).copy()
	N_nodup = N - C1.shape[1]
	dis = euclidean_distances(X)[:, N_nodup:]
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		lfdis = np.log(beta * (dis ** alpha))
	lfdis[(dis == 0.0) | np.isinf(lfdis) | np.isnan(lfdis)] = 0.0
	obj = loggamma(C1 + 1.0).sum() - (C1 * lfdis).sum()
	C1 = np.concatenate((C1[: N_nodup, :], np.dot(S, C1[N_nodup:, :])))
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		obj2 = (np.dot(C1, S.T) - C_dup) ** 2
	obj2 = obj2[np.invert(np.isnan(obj2) | np.isinf(obj2))].sum()
	if np.isnan(obj + obj2):
		raise ValueError("Function evaluation returns nan.")
	print (obj, obj2)
	return obj + obj2


def c_gradient(x, X, N, S = None, C_dup = None, alpha = -3.0, beta = 1.0):
	C1 = x.reshape(N, -1).copy()
	N_nodup = N - C1.shape[1]
	dis = euclidean_distances(X)[:, N_nodup:]
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		lfdis = np.log(beta * (dis ** alpha))
	lfdis[(dis == 0.0) | np.isinf(lfdis) | np.isnan(lfdis)] = 0.0
	grad = digamma(C1 + 1.0) - lfdis
	grad = grad.flatten()
	C1 = np.concatenate((C1[: N_nodup, :], np.dot(S, C1[N_nodup:, :])))
	ssum1 = S.sum(axis = 1).astype(int)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		grad_counts = (2.0 * (np.dot(C1, S.T) - C_dup)).repeat(ssum1, axis = 1). \
				repeat(np.concatenate((np.ones([N_nodup]).astype(int), ssum1)), axis = 0).flatten()
	grad_counts[np.isnan(grad_counts)] = 0.0
	return grad + grad_counts


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Compute the expanded Hi-C from collapsed Hi-C.")
	parser.add_argument("--raw_matrix", help = "Raw, collapsed Hi-C matrix, in *.txt or *.npy format.", required = True)
	parser.add_argument("--annotation", help = "Annotation of bins in the input matrix.", required = True)
	parser.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format.", required = True)
	parser.add_argument("--alpha", help = "The value of alpha from 3D structure computation.", required = True, type = float)
	parser.add_argument("--beta", help = "The value of beta from 3D structure computation.", required = True, type = float)
	parser.add_argument("--output_prefix", help = "Prefix of the output files.", required = True)
	parser.add_argument("--strategy", help = "One option from redist/poisson/hybrid.", default = "redist")
	parser.add_argument("--log_fn", help = "Name of log file.")
	parser.add_argument("--save_npy", help = "Save matrices to *.npy format", action = "store_true")
	start_time = time.time()
	args = parser.parse_args()

	"""
	Set up logging
	"""
	log_fn = ""
	if not args.log_fn:
		log_fn = args.output_prefix + "_matrix_expansion.log"
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
	Read collapsed matrix
	"""
	C = np.array([])
	if args.raw_matrix.endswith(".txt"):
		C = np.loadtxt(args.raw_matrix)
	elif args.raw_matrix.endswith(".npy"):
		C = np.load(args.raw_matrix)
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded raw collapsed ecDNA matrix.")

	"""
	Read in annotation file
	"""
	N = -1 # Num bins in expanded Hi-C
	bins = []
	row_labels = dict() # Map an interval of size RES to the index of all its copies
	fp = open(args.annotation, 'r')
	for line in fp:
		s = line.strip().split()
		bin = (s[0], int(s[1]))
		bins.append(bin)
		row_labels[bin] = [int(s[3])]
		N = max(N, int(s[3]))
		if len(s) > 4:
			for i in range(4, len(s)):
				row_labels[(s[0], int(s[1]))].append(int(s[i]))
				N = max(N, int(s[i]))
	N += 1
	fp.close()
	assert (len(row_labels) == C.shape[0])
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix annotations.")
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Expanded ecDNA matrix size: %d." %N)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Collapsed ecDNA matrix size: %d." %len(row_labels))

	bins = sorted(bins, key = lambda bin: row_labels[bin][0])
	idx_nodup = [bi for bi in range(len(bins)) if len(row_labels[bins[bi]]) == 1]
	idx_dup = [bi for bi in range(len(bins)) if len(row_labels[bins[bi]]) > 1]
	dup_times = np.array([len(row_labels[bins[bi]]) for bi in idx_dup])
	N_nodup = len(idx_nodup)
	C_nodup = C[np.ix_(idx_nodup, idx_nodup)]
	C_dup = np.concatenate((C[np.ix_(idx_nodup, idx_dup)], C[np.ix_(idx_dup, idx_dup)]))

	i_nodup, i_dup = 0, 0
	idx_map = dict()
	idx_map_rev = dict()
	for bi in range(len(bins)):
		if bi in idx_nodup:
			idx_map[row_labels[bins[bi]][0]] = i_nodup
			idx_map_rev[i_nodup] = row_labels[bins[bi]][0]
			i_nodup += 1
		else:
			for i_ in range(len(row_labels[bins[bi]])):
				idx_map[row_labels[bins[bi]][i_]] = N_nodup + i_dup
				idx_map_rev[N_nodup + i_dup] = row_labels[bins[bi]][i_]
				i_dup += 1
	idx_map = np.array([idx_map[i] for i in range(len(idx_map))])
	idx_map_rev = np.array([idx_map_rev[i] for i in range(len(idx_map_rev))])

	X = np.array([])
	if args.structure.endswith(".txt"):
		X = np.loadtxt(args.structure)
	elif args.structure.endswith(".npy"):
		X = np.load(args.structure)
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA 3D structure.")
	
	X_ = X[idx_map_rev]
	dis = euclidean_distances(X_)
	C1_ = np.zeros([N, N - N_nodup])

	if args.strategy == 'redist' or args.strategy == 'hybrid':
		for ci in range(N_nodup):
			cj = 0
			for dupi in range(len(dup_times)):
				ddi = np.array([dis[ci][N_nodup + cj + di] for di in range(dup_times[dupi])])
				sum_d = sum(ddi ** args.alpha)
				for j_ in range(len(ddi)):
					C1_[ci][cj] = C_dup[ci][dupi] / sum_d * (ddi[j_] ** args.alpha)
					cj += 1
		avg_dis_adj = np.average([dis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)])
		ci = N_nodup
		for bi1 in range(len(dup_times)):
			ni1 = dup_times[bi1]
			cj = 0
			for bi2 in range(len(dup_times)):
				ni2 = dup_times[bi2]
				if bi1 == bi2:
					C_total = C_dup[N_nodup + bi1][bi2]
					for i_ in range(ni1):
						for j_ in range(ni1):
							if i_ != j_:
								C_ij_ = args.beta * (dis[ci + i_][N_nodup + cj + j_] ** args.alpha)
								C1_[ci + i_][cj + j_] = C_ij_
								C_total -= C_ij_
					for i_ in range(ni1):
						for j_ in range(ni1):
							if i_ == j_:
								C1_[ci + i_][cj + j_] = max(C_total, 0.5 * C_dup[N_nodup + bi1][bi2]) / ni1
				else:
					ddij = np.ones([ni1, ni2])
					for i_ in range(ni1):
						for j_ in range(ni2):
							assert ci + i_ != N_nodup + cj + j_
							ddij[i_][j_] = dis[ci + i_][N_nodup + cj + j_]
					sum_d = (ddij ** args.alpha).sum()
					for i_ in range(ni1):
						for j_ in range(ni2):
							C1_[ci + i_][cj + j_] = C_dup[N_nodup + bi1][bi2] / sum_d * (ddij[i_][j_] ** args.alpha)
				cj += ni2 
			ci += ni1
	if args.strategy == 'poisson':
		for ci in range(N_nodup):
			cj = 0
			for dupi in range(len(dup_times)):
				for di in range(dup_times[dupi]):
					C1_[ci][cj + di] = args.beta * (dis[ci][N_nodup + cj + di] ** args.alpha)
				cj += dup_times[dupi]
		ci = N_nodup
		for bi1 in range(len(dup_times)):
			ni1 = dup_times[bi1]
			cj = 0
			for bi2 in range(len(dup_times)):
				ni2 = dup_times[bi2]
				for i_ in range(ni1):
					for j_ in range(ni2):
						if ci + i_ == N_nodup + cj + j_:
							assert bi1 == bi2
							C1_[ci + i_][cj + j_] = C_dup[N_nodup + bi1][bi2] / ni1
						else:
							C1_[ci + i_][cj + j_] = args.beta * (dis[ci + i_][N_nodup + cj + j_] ** args.alpha)
				cj += ni2 
			ci += ni1
	if args.strategy == 'hybrid':
		S = np.zeros([C.shape[0] - N_nodup, N - N_nodup])
		s = 0
		for i in range(len(dup_times)):
			for j in range(s, s + dup_times[i]):
				S[i][j] = 1
			s += dup_times[i]
		C1 = C[np.ix_(idx_nodup, idx_dup)].repeat(dup_times, axis = 1)
		C2 = C[np.ix_(idx_dup, idx_dup)].repeat(dup_times, axis = 1).repeat(dup_times, axis = 0)
		bounds_c = np.concatenate((C1, C2)).flatten()
		bounds_c = [(0.0, bounds_c[i]) for i in range(len(bounds_c))]
		results = optimize.fmin_l_bfgs_b(c_obj, C1_.flatten(), fprime = c_gradient, args = (X_, N, S, C_dup, args.alpha, args.beta, ), 
							bounds = bounds_c, maxiter = 10000)
		C1_ = results[0].reshape(N, -1)
		ci = N_nodup
		cj = 0
		for bi1 in range(len(dup_times)):
			ni1 = dup_times[bi1]
			C_total = C_dup[N_nodup + bi1][bi1]
			for i_ in range(ni1):
				for j_ in range(ni1):
					if i_ != j_:
						C_ij_ = args.beta * (dis[ci + i_][N_nodup + cj + j_] ** args.alpha)
						C1_[ci + i_][cj + j_] = C_ij_
						C_total -= C_ij_
			for i_ in range(ni1):
				for j_ in range(ni1):
					if i_ == j_:
						C1_[ci + i_][cj + j_] = max(C_total, 0.5 * C_dup[N_nodup + bi1][bi1]) / ni1
			ci += ni1
			cj += ni1

	D = np.block([[C_nodup, C1_[: N_nodup, :]], [C1_.T]])
	D = D[np.ix_(idx_map, idx_map)]
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Reconstructed the expanded matrix.")
	D = normalization.ICE_normalization(D)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Normalized the expanded matrix.")
	if args.save_npy:
		np.save(args.output_prefix + "_expanded_matrix.npy", D)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the expanded matrix to %s." %(args.output_prefix + "_expanded_matrix.npy"))
	else:
		np.savetxt(args.output_prefix + "_expanded_matrix.txt", D)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the expanded matrix to %s." %(args.output_prefix + "_expanded_matrix.txt"))
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")


	