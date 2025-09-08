"""
Infer ecDNA 3D structure with MDS and Poisson models
"""
import sys
import time
import warnings
import numpy as np
import argparse
import logging
import autograd.numpy as auto_np
from scipy import optimize
from scipy.special import loggamma, digamma
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances
from autograd import grad

try:
	from ec3d.util import getTransformation, create_logger
except:
	from util import getTransformation, create_logger


def compute_wish_distances(C_, alpha = -3.0, beta = 1.0):
	"""
	Computes wish distances from a counts matrix

	C_: np array, Interaction counts matrix

	alpha: float, optional, default: -3
	       Coefficient of the power law used in converting interaction counts to wish distances

	beta: float, optional, default: 1
	      Scaling factor of the structure

	Returns: np array, Wish distances
	"""
	wish_distances = C_.copy() / beta
	wish_distances[wish_distances != 0] **= (1.0 / alpha)
	return wish_distances


def mds_obj(x, N, wd_nodup, S, C_dup, alpha = -3.0, beta = 1.0):
	"""
	Objective function for MDS
	
	x: 1d variable array of length 3 * N + N * (N - N_nodup)
	   where N is the size of donor ecDNA matrix (possibly with duplicated bins)
	   N_nodup is the number of bins that only occur one time on ecDNA
	   The first 3 * N entries correspond to the coordinates
	   The following N * (N - N_nodup) entries correspond to interaction counts in duplicated bins

	N: size of ecDNA matrix including duplicated bins

	wd_nodup: numpy array of size N_nodup * N_nodup
	          Wish distances between bins that only occur one time on ecDNA

	S: numpy array of size (N_dedup - N_nodup) * (N - N_nodup)
	   where N_dedup us the total number of distinct bins in ecDNA on the reference genome
	   matrix for summing up rows or cols representing duplicated bins

	C_dup: numpy array of size N_dedup * (N_dedup - N_nodup)
	       Observed sum counts in duplicated bins

	alpha: float, optional, default: -3
	       Coefficient of the power law used in converting interaction counts to wish distances

	beta: float, optional, default: 1
	      Scaling factor of the structure

	Returns: float, value of objective function
	"""
	N_nodup = wd_nodup.shape[0]
	assert (N > 0 and N_nodup > 0 and N_nodup <= N and len(x) == 3 * N + N * (N - N_nodup))
	X = x[:3 * N].reshape(-1, 3)
	dis = euclidean_distances(X)
	distances = wd_nodup
	if N_nodup < N:
		C1 = x[3 * N:].reshape(N, -1)
		wd_dup = compute_wish_distances(C1, alpha, beta)
		distances = np.concatenate((distances, wd_dup[:N_nodup, :]), axis = 1)
		distances = np.concatenate((distances, wd_dup.T))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			obj1 = (1.0 / distances ** 2) * (dis - distances) ** 2
		C1 = np.concatenate((C1[: N_nodup, :], np.dot(S, C1[N_nodup:, :])))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			obj2 = (1.0 / C_dup) * (np.dot(C1, S.T) - C_dup) ** 2
		return obj1[np.invert(np.isnan(obj1) | np.isinf(obj1))].sum() + obj2[np.invert(np.isnan(obj2) | np.isinf(obj2))].sum()
	else:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			obj = (1.0 / distances ** 2) * (dis - distances) ** 2
		return obj[np.invert(np.isnan(obj) | np.isinf(obj))].sum()


def mds_gradient(x, N, wd_nodup, S, C_dup, alpha = -3.0, beta = 1.0):
	"""
	Gradient vector for MDS
	
	x: 1d variable array of length 3 * N + N * (N - N_nodup)
	   where N is the size of donor ecDNA matrix (possibly with duplicated bins)
	   N_nodup is the number of bins that only occur one time on ecDNA
	   The first 3 * N entries correspond to the coordinates
	   The following N * (N - N_nodup) entries correspond to interaction counts in duplicated bins

	N: size of ecDNA matrix including duplicated bins

	wd_nodup: numpy array of size N_nodup * N_nodup
	          Wish distances between bins that only occur one time on ecDNA

	S: numpy array of size (N_dedup - N_nodup) * (N - N_nodup)
	   where N_dedup us the total number of distinct bins in ecDNA on the reference genome
	   matrix for summing up rows or cols representing duplicated bins

	C_dup: numpy array of size N_dedup * (N_dedup - N_nodup)
	       Observed sum counts in duplicated bins

	alpha: float, optional, default: -3
	       Coefficient of the power law used in converting interaction counts to wish distances

	beta: float, optional, default: 1
	      Scaling factor of the structure

	Returns: 1d gradient array of length 3 * N + N * (N - N_nodup)
	"""
	N_nodup = wd_nodup.shape[0]
	assert (N > 0 and N_nodup > 0 and N_nodup <= N and len(x) == 3 * N + N * (N - N_nodup))
	X = x[:3 * N].reshape(-1, 3)
	tmp = X.repeat(N, axis = 0).reshape(N, N, 3)
	dif = tmp - tmp.transpose(1, 0, 2)
	dis = euclidean_distances(X).repeat(3, axis = 1).flatten()
	distances = wd_nodup
	dis_dup = np.array([])
	C1 = np.array([])
	if N_nodup < N:
		C1 = x[3 * N:].reshape(N, -1)
		dis_dup = euclidean_distances(X)[:, N_nodup:]
		wd_dup = compute_wish_distances(C1, alpha, beta)
		distances = np.concatenate((distances, wd_dup[:N_nodup, :]), axis = 1)
		distances = np.concatenate((distances, wd_dup.T))
	distances = distances.repeat(3, axis = 1).flatten()
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		grad1 = (2 * dif.flatten() * (dis - distances) / dis) / (distances ** 2)
	grad1[(distances == 0) | np.isnan(grad1)] = 0.0
	if N_nodup < N:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			grad2 = (2 * dis_dup * ((C1 / beta) ** (1.0 / alpha) - dis_dup)) / \
				(alpha * C1 * ((C1 / beta) ** (2.0 / alpha)))
		C1 = np.concatenate((C1[: N_nodup, :], np.dot(S, C1[N_nodup:, :])))
		ssum1 = S.sum(axis = 1).astype(int)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			grad_counts = ((1.0 / C_dup) * (np.dot(C1, S.T) - C_dup)).repeat(ssum1, axis = 1). \
					repeat(np.concatenate((np.ones([N_nodup]).astype(int), ssum1)), axis = 0).flatten()
		grad2[np.isinf(grad2) | np.isnan(grad2)] = 0.0
		grad2 = grad2.flatten()
		grad_counts[np.isnan(grad_counts)] = 0.0
		grad2 += 2 * grad_counts
		return np.concatenate((grad1.reshape(N, N, 3).sum(axis = 1).flatten(), grad2))
	else:
		return grad1.reshape(N, N, 3).sum(axis = 1).flatten()
 

def mds(C, N, idx_nodup, idx_dup, dup_times, ini_c = None, alpha = -3.0, beta = 1.0, maxiter = 10000):
	"""
	MDS caller
	"""
	N_nodup = len(idx_nodup)
	C_nodup = C[np.ix_(idx_nodup, idx_nodup)]
	random_state = check_random_state(None)
	ini_x = 1 - 2 * random_state.rand(N * 3)
	bounds_x = [(-1.0, 1.0) for i in range(N * 3)]
	wish_distances_nodup = compute_wish_distances(C_nodup, alpha = alpha, beta = beta)
	results = []
	if N_nodup < N:
		C1 = C[np.ix_(idx_nodup, idx_dup)].repeat(dup_times, axis = 1)
		C2 = C[np.ix_(idx_dup, idx_dup)].repeat(dup_times, axis = 1).repeat(dup_times, axis = 0)
		bounds_c = np.concatenate((C1, C2)).flatten()
		bounds_c = [(0.0, bounds_c[i]) for i in range(len(bounds_c))]
		dup_times_r = dup_times.reshape(1, -1).repeat(dup_times, axis = 1)
		if ini_c is None:
			ini_c = np.concatenate((C1 / dup_times_r.repeat(N_nodup, axis = 0), \
						C2 / np.outer(dup_times_r, dup_times_r)))
		S = np.zeros([C.shape[0] - N_nodup, N - N_nodup])
		s = 0
		for i in range(len(dup_times)):
			for j in range(s, s + dup_times[i]):
				S[i][j] = 1
			s += dup_times[i]
		C_dup = np.concatenate((C[np.ix_(idx_nodup, idx_dup)], C[np.ix_(idx_dup, idx_dup)]))
		ini = np.concatenate((ini_x.flatten(), ini_c.flatten()))
		results = optimize.fmin_l_bfgs_b(mds_obj, ini, fprime = mds_gradient,
						args = (N, wish_distances_nodup, S, C_dup, alpha, beta, ),
						bounds = bounds_x + bounds_c, maxiter = maxiter)
	else:
		results = optimize.fmin_l_bfgs_b(mds_obj, ini_x.flatten(), fprime = mds_gradient, args = 
						(N, wish_distances_nodup, None, None, alpha, beta, ), bounds = bounds_x, 
						maxiter = maxiter)
	return results[0][:3 * N].reshape(-1, 3), results[0][3 * N:].reshape(N, -1)


def exponent_obj(x, X, C_nodup, S = None, C_dup = None, beta = 1.0):
	"""
	Objective function for optimizing alpha
	
	x: 1d variable array of length 1

	X: numpy array of size N * 3

	Todo: other parameters

	beta: float, Estimated scaling factor of the structures

	Returns: float, value of log likelihood (involving alpha) evaludated with given alpha
	"""
	alpha = x[0]
	N = X.shape[0]
	N_nodup = C_nodup.shape[0]
	dis = euclidean_distances(X)

	if N_nodup < N:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		fdis[(dis == 0.0) | np.isnan(fdis)] = 0.0
		fdis_dup = fdis[:, N_nodup:]
		fdis_dup = np.concatenate((fdis_dup[: N_nodup, :], np.dot(S, fdis_dup[N_nodup:, :])))
		fdis_dup = np.dot(fdis_dup, S.T)
		fdis = np.block([[fdis[: N_nodup, : N_nodup], fdis_dup[:N_nodup, :]], [fdis_dup.T]])
		D = np.block([[C_nodup, C_dup[: N_nodup, :]], [C_dup.T]])
		N_dedup = C_dup.shape[0]
		mask = np.invert(np.tri(N_dedup, dtype = bool)) & (D != 0) & (fdis != 0)
		fdis = fdis[mask]
		ll_alpha = fdis.sum()
		ll_alpha -= (D[mask] * np.log(fdis)).sum()
		if np.isnan(ll_alpha):
			raise ValueError("Function evaluation returns nan.")
		return ll_alpha
	else:
		mask = np.invert(np.tri(N, dtype = bool)) & (C_nodup != 0) & (dis != 0)
		ll_alpha = (beta * (dis[mask] ** alpha)).sum() - (alpha * C_nodup[mask] * np.log(dis[mask])).sum()
		if np.isnan(ll_alpha):
			raise ValueError("Function evaluation returns nan.")
		return ll_alpha


def exponent_obj_alpha_auto(x, X, C_nodup, S = None, C_dup = None, beta = 1.0, reg = True, gamma = 0.01):
	"""
	Objective function for optimizing alpha
	
	x: 1d variable array of length 1

	X: numpy array of size N * 3

	Todo: other parameters

	beta: float, Estimated scaling factor of the structures

	Returns: float, value of log likelihood (involving alpha) evaludated with given alpha
	"""
	alpha = x[0]
	N = X.shape[0]
	N_nodup = C_nodup.shape[0]
	dis = euclidean_distances(X)
	mask = (dis == 0.0)
	dis = auto_np.where(mask, 0.01, dis)

	if N_nodup < N:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		mask1 = (dis == 0.0)
		fdis = auto_np.where(mask1, 0.0, fdis)
		mask1 = (auto_np.isnan(fdis))
		fdis = auto_np.where(mask1, 0.0, fdis)
		ll_alpha = 0.0
		if reg and idx_map is not None:
			diagonal_off_1 = [fdis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)]
			ll_alpha += auto_np.var(auto_np.array(diagonal_off_1))
		ll_alpha *= (gamma * N)
		fdis_dup = fdis[:, N_nodup:]
		fdis_dup = auto_np.concatenate((fdis_dup[: N_nodup, :], auto_np.dot(S, fdis_dup[N_nodup:, :])))
		fdis_dup = auto_np.dot(fdis_dup, S.T)
		fdis = auto_np.hstack([fdis[: N_nodup, : N_nodup], fdis_dup[:N_nodup, :]])
		fdis = auto_np.vstack([fdis, fdis_dup.T])
		D = auto_np.block([[C_nodup, C_dup[: N_nodup, :]], [C_dup.T]])
		N_dedup = C_dup.shape[0]
		mask = auto_np.invert(auto_np.tri(N_dedup, dtype = bool)) & (D != 0)
		fdis = fdis[mask]
		ll_alpha += fdis.sum()
		mask1 = (fdis == 0.0)
		fdis = auto_np.where(mask1, 1.0, fdis)
		ll_alpha -= (D[mask] * auto_np.log(fdis)).sum()
		if auto_np.isnan(ll_alpha):
			raise ValueError("Function evaluation returns nan.")
		return ll_alpha
	else:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		mask1 = (dis == 0.0)
		fdis = auto_np.where(mask1, 0.0, fdis)
		mask1 = (auto_np.isnan(fdis))
		fdis = auto_np.where(mask1, 0.0, fdis)
		ll_alpha = 0.0
		if reg and idx_map is not None:
			diagonal_off_1 = [fdis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)]
			ll_alpha += auto_np.var(auto_np.array(diagonal_off_1))
		ll_alpha *= (gamma * N)

		mask = auto_np.invert(auto_np.tri(N, dtype = bool)) & (C_nodup != 0)
		fdis = fdis[mask]
		ll_alpha += fdis.sum()
		mask1 = (fdis == 0.0)
		fdis = auto_np.where(mask1, 1.0, fdis)
		ll_alpha -= (C_nodup[mask] * auto_np.log(fdis)).sum()
		if auto_np.isnan(ll_alpha):
			raise ValueError("Function evaluation returns nan.")
		return ll_alpha


def exponent_obj_beta_auto(x, X, C_nodup, S = None, C_dup = None, alpha = -3, reg = True, gamma = 0.01):
	"""
	Objective function for optimizing alpha
	
	x: 1d variable array of length 1

	X: numpy array of size N * 3

	Todo: other parameters

	beta: float, Estimated scaling factor of the structures

	Returns: float, value of log likelihood (involving alpha) evaludated with given alpha
	"""
	beta = x[0]
	N = X.shape[0]
	N_nodup = C_nodup.shape[0]
	dis = euclidean_distances(X)
	mask = (dis == 0.0)
	dis = auto_np.where(mask, 0.01, dis)

	if N_nodup < N:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		mask1 = (dis == 0.0)
		fdis = auto_np.where(mask1, 0.0, fdis)
		mask1 = (auto_np.isnan(fdis))
		fdis = auto_np.where(mask1, 0.0, fdis)
		ll_beta = 0.0
		if reg and idx_map is not None:
			diagonal_off_1 = [fdis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)]
			ll_beta += auto_np.var(auto_np.array(diagonal_off_1))
		ll_beta *= (gamma * N)
		fdis_dup = fdis[:, N_nodup:]
		fdis_dup = auto_np.concatenate((fdis_dup[: N_nodup, :], auto_np.dot(S, fdis_dup[N_nodup:, :])))
		fdis_dup = auto_np.dot(fdis_dup, S.T)
		fdis = auto_np.hstack([fdis[: N_nodup, : N_nodup], fdis_dup[:N_nodup, :]])
		fdis = auto_np.vstack([fdis, fdis_dup.T])
		D = auto_np.block([[C_nodup, C_dup[: N_nodup, :]], [C_dup.T]])
		N_dedup = C_dup.shape[0]
		mask = auto_np.invert(auto_np.tri(N_dedup, dtype = bool)) & (D != 0)
		fdis = fdis[mask]
		ll_beta += fdis.sum()
		mask1 = (fdis == 0.0)
		fdis = auto_np.where(mask1, 1.0, fdis)
		ll_beta -= (D[mask] * auto_np.log(fdis)).sum()
		if auto_np.isnan(ll_beta):
			raise ValueError("Function evaluation returns nan.")
		return ll_beta
	else:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		mask1 = (dis == 0.0)
		fdis = auto_np.where(mask1, 0.0, fdis)
		mask1 = (auto_np.isnan(fdis))
		fdis = auto_np.where(mask1, 0.0, fdis)
		ll_beta = 0.0
		if reg and idx_map is not None:
			diagonal_off_1 = [fdis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)]
			ll_beta += auto_np.var(auto_np.array(diagonal_off_1))
		ll_beta *= (gamma * N)

		mask = auto_np.invert(auto_np.tri(N, dtype = bool)) & (C_nodup != 0)
		fdis = fdis[mask]
		ll_beta += fdis.sum()
		mask1 = (fdis == 0.0)
		fdis = auto_np.where(mask1, 1.0, fdis)
		ll_beta -= (C_nodup[mask] * auto_np.log(fdis)).sum()
		if auto_np.isnan(ll_beta):
			raise ValueError("Function evaluation returns nan.")
		return ll_beta


def exponent_gradient(x, X, C_nodup, S = None, C_dup = None, beta = 1.0):
	"""
	Gradient function for optimizing alpha
	
	x: 1d variable array of length 1

	X: numpy array of size N * 3

	D: numpy array of size N * N

	beta: float, Estimated scaling factor of the structures

	Returns: gradient array of length 1
	"""
	alpha = x[0]
	N = X.shape[0]
	N_nodup = C_nodup.shape[0]
	dis = euclidean_distances(X)

	if N_nodup < N:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = dis ** alpha
			fdis1 = (dis ** alpha) * np.log(dis)
		fdis[(dis == 0.0) | np.isnan(fdis)] = 0.0
		fdis1[(dis == 0.0) | np.isnan(fdis1)] = 0.0
		fdis_dup = fdis[:, N_nodup:]
		fdis_dup = np.concatenate((fdis_dup[: N_nodup, :], np.dot(S, fdis_dup[N_nodup:, :])))
		fdis_dup = np.dot(fdis_dup, S.T)
		fdis_dup1 = fdis1[:, N_nodup:]
		fdis_dup1 = np.concatenate((fdis_dup1[: N_nodup, :], np.dot(S, fdis_dup1[N_nodup:, :])))
		fdis_dup1 = np.dot(fdis_dup1, S.T)
		fdis1 = np.block([[fdis1[: N_nodup, : N_nodup], fdis_dup1[:N_nodup, :]], [fdis_dup1.T]])
		fdis_dup = fdis_dup1 / fdis_dup
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			ldis = np.log(dis)
			ldis = np.block([[ldis[: N_nodup, : N_nodup], fdis_dup[:N_nodup, :]], [fdis_dup.T]])
		ldis[np.isnan(ldis) | np.isinf(ldis)] = 0.0
		D = np.block([[C_nodup, C_dup[: N_nodup, :]], [C_dup.T]])
		N_dedup = C_dup.shape[0]
		mask = np.invert(np.tri(N_dedup, dtype = bool)) & (D != 0) & (ldis != 0)
		grad_alpha = beta * fdis1[mask].sum() - (D[mask] * ldis[mask]).sum()
		if np.isnan(grad_alpha):
			raise ValueError("Function evaluation returns nan.")
		return np.array([grad_alpha])
	else:
		mask = np.invert(np.tri(N, dtype = bool)) & (C_nodup != 0) & (dis != 0)
		grad_alpha = (beta * (dis[mask] ** alpha) * np.log(dis[mask])).sum() - \
				(C_nodup[mask] * np.log(dis[mask])).sum()
		if np.isnan(grad_alpha):
			raise ValueError("Function evaluation returns nan.")
		return np.array([grad_alpha])


def estimate_alpha_beta(X, C_nodup, S = None, C_dup = None, alpha = -3.0, beta = 1.0):
	"""
	Estimate alpha and beta in Poisson model

	X: numpy array of size N * 3

	D: numpy array of size N * N

	Returns: (Estimated) alpha, beta
	"""
	N = X.shape[0]
	N_nodup = C_nodup.shape[0]
	bounds_ = np.array([[-100, 1e-2]])
	random_state = check_random_state(None)
	#if alpha == -3.0:
	#ini = -random_state.randint(1, 100) + random_state.rand(1)
	results = optimize.fmin_l_bfgs_b(exponent_obj, np.array([alpha]), fprime = exponent_gradient,
					args = (X, C_nodup, S, C_dup, beta, ), bounds = bounds_, maxiter = 1000)
	# results = optimize.fmin_l_bfgs_b(exponent_obj_, np.array([alpha]), fprime = exponent_gradient_,
	# 				args = (X, C_nodup, S, C_dup, beta, ), bounds = bounds_, maxiter = 1000)
	alpha = results[0]
	dis = euclidean_distances(X)
	if N_nodup < N:
		mask = np.invert(np.tri(N_nodup, dtype = bool)) & (C_nodup != 0)
		C_sum = C_nodup[mask].sum()
		C_sum_1 = C_sum
		C_sum += C_dup[: N_nodup, :].sum()
		mask = np.invert(np.tri(C_dup.shape[1], dtype = bool)) & (C_dup[N_nodup:, :] != 0)
		C_sum += C_dup[N_nodup:, :][mask].sum()
		mask = np.invert(np.tri(N, dtype = bool)) & (dis != 0)
		d_sum = (dis[mask] ** alpha).sum()
		beta = C_sum / d_sum
		mask = np.invert(np.tri(N_nodup, dtype = bool)) & (dis[:N_nodup, :N_nodup] != 0)
		return results[0][0], beta, results[1]
	else:
		mask = np.invert(np.tri(N, dtype = bool)) & (C_nodup != 0) & (dis != 0)
		beta = C_nodup[mask].sum() / (dis[mask] ** alpha).sum()                                            
		return results[0][0], beta, results[1]


def poisson_obj(x, N, C_nodup, S = None, C_dup = None, idx_map = None, alpha = -3.0, beta = 1.0, reg = True, gamma = 0.01):
	"""
	Objective function for Poisson model
	
	x: 1d variable array of length 3 * N + N * (N - N_nodup)
	   where N is the size of donor ecDNA matrix (possibly with duplicated bins)
	   N_nodup is the number of bins that only occur one time on ecDNA
	   The first 3 * N entries correspond to the coordinates
	   The following N * (N - N_nodup) entries correspond to interaction counts in duplicated bins

	N: size of ecDNA matrix including duplicated bins

	C_nodup: numpy array of size N_nodup * N_nodup
	         Count matrix for bins that only occur one time on ecDNA

	S: numpy array of size (N_dedup - N_nodup) * (N - N_nodup)
	   where N_dedup us the total number of distinct bins in ecDNA on the reference genome
	   matrix for summing up rows or cols representing duplicated bins

	C_dup: numpy array of size N_dedup * (N_dedup - N_nodup)
		   Observed sum counts in duplicated bins

	alpha: float, optional, default: -3
	       Coefficient of the power law used in converting interaction counts to wish distances

	beta: float, optional, default: 1
	      Scaling factor of the structure

	Returns: float, value of objective function
	"""
	N_nodup = C_nodup.shape[0]
	assert (N > 0 and N_nodup > 0 and N_nodup <= N and len(x) == 3 * N)
	X = x.reshape(-1, 3)
	dis = euclidean_distances(X)

	if N_nodup < N:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		fdis[(dis == 0.0) | np.isnan(fdis)] = 0.0
		obj = 0.0
		if reg and idx_map is not None:
			obj = sum([fdis[idx_map[i]][idx_map[i + 1]] ** 2 for i in range(N - 1)]) / (N - 1)
			obj -= ((sum([fdis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)]) / (N - 1)) ** 2)
		obj *= (gamma * N)
		fdis_dup = fdis[:, N_nodup:]
		fdis_dup = np.concatenate((fdis_dup[: N_nodup, :], np.dot(S, fdis_dup[N_nodup:, :])))
		fdis_dup = np.dot(fdis_dup, S.T)
		fdis = np.block([[fdis[: N_nodup, : N_nodup], fdis_dup[:N_nodup, :]], [fdis_dup.T]])
		D = np.block([[C_nodup, C_dup[: N_nodup, :]], [C_dup.T]])
		N_dedup = C_dup.shape[0]
		mask = np.invert(np.tri(N_dedup, dtype = bool)) & (D != 0)
		fdis = fdis[mask]
		obj += fdis.sum()
		fdis[(fdis == 0.0)] = 1.0
		obj -= (D[mask] * np.log(fdis)).sum()
		if np.isnan(obj):
			raise ValueError("Function evaluation returns nan.")
		return obj
	else:
		mask = np.invert(np.tri(N, dtype = bool)) & (C_nodup != 0) & (dis != 0)
		fdis = beta * (dis[mask] ** alpha)
		obj = fdis.sum() - (C_nodup[mask] * np.log(fdis)).sum()    
		if np.isnan(obj):
			raise ValueError("Function evaluation returns nan.")
		return obj


def poisson_obj_reg_auto(x, N, C_nodup, S = None, C_dup = None, idx_map = None, alpha = -3.0, beta = 1.0, reg = True, gamma = 0.01):
	"""
	Objective function for Poisson model
	
	x: 1d variable array of length 3 * N + N * (N - N_nodup)
	   where N is the size of donor ecDNA matrix (possibly with duplicated bins)
	   N_nodup is the number of bins that only occur one time on ecDNA
	   The first 3 * N entries correspond to the coordinates
	   The following N * (N - N_nodup) entries correspond to interaction counts in duplicated bins

	N: size of ecDNA matrix including duplicated bins

	C_nodup: numpy array of size N_nodup * N_nodup
	         Count matrix for bins that only occur one time on ecDNA

	S: numpy array of size (N_dedup - N_nodup) * (N - N_nodup)
	   where N_dedup us the total number of distinct bins in ecDNA on the reference genome
	   matrix for summing up rows or cols representing duplicated bins

	C_dup: numpy array of size N_dedup * (N_dedup - N_nodup)
		   Observed sum counts in duplicated bins

	alpha: float, optional, default: -3
	       Coefficient of the power law used in converting interaction counts to wish distances

	beta: float, optional, default: 1
	      Scaling factor of the structure

	Returns: float, value of objective function
	"""
	N_nodup = C_nodup.shape[0]
	# assert (N > 0 and N_nodup > 0 and N_nodup <= N and len(x) == 3 * N)
	x = x.reshape(-1, 3)
	x1 = x[:, auto_np.newaxis, :]
	dis_sq = auto_np.sum((x1 - x) ** 2, axis=2)
	mask = (dis_sq == 0.0)
	dis_sq = auto_np.where(mask, 0.0001, dis_sq)
	dis = auto_np.sqrt(dis_sq)
	if N_nodup < N:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		mask1 = (dis == 0.0)
		fdis = auto_np.where(mask1, 0.0, fdis)
		mask1 = (auto_np.isnan(fdis))
		fdis = auto_np.where(mask1, 0.0, fdis)
		obj = 0.0
		if reg and idx_map is not None:
			diagonal_off_1 = [fdis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)]
			obj += auto_np.var(auto_np.array(diagonal_off_1))
		obj *= (gamma * N)
		fdis_dup = fdis[:, N_nodup:]
		fdis_dup = auto_np.concatenate((fdis_dup[: N_nodup, :], auto_np.dot(S, fdis_dup[N_nodup:, :])))
		fdis_dup = auto_np.dot(fdis_dup, S.T)
		fdis = auto_np.hstack([fdis[: N_nodup, : N_nodup], fdis_dup[:N_nodup, :]])
		fdis = auto_np.vstack([fdis, fdis_dup.T])
		D = auto_np.block([[C_nodup, C_dup[: N_nodup, :]], [C_dup.T]])
		N_dedup = C_dup.shape[0]
		mask = auto_np.invert(auto_np.tri(N_dedup, dtype = bool)) & (D != 0)
		fdis = fdis[mask]
		obj += fdis.sum()
		mask1 = (fdis == 0.0)
		fdis = auto_np.where(mask1, 1.0, fdis)
		obj -= (D[mask] * auto_np.log(fdis)).sum()
		if auto_np.isnan(obj):
		 	raise ValueError("Function evaluation returns nan.")
		return obj
	else:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fdis = beta * (dis ** alpha)
		mask1 = (dis == 0.0)
		fdis = auto_np.where(mask1, 0.0, fdis)
		mask1 = (auto_np.isnan(fdis))
		fdis = auto_np.where(mask1, 0.0, fdis)
		obj = 0.0
		if reg and idx_map is not None:
			diagonal_off_1 = [fdis[idx_map[i]][idx_map[i + 1]] for i in range(N - 1)]
			obj += auto_np.var(auto_np.array(diagonal_off_1))
		obj *= (gamma * N)

		mask = auto_np.invert(auto_np.tri(N, dtype = bool)) & (C_nodup != 0)
		fdis = fdis[mask]
		obj += fdis.sum()
		mask1 = (fdis == 0.0)
		fdis = auto_np.where(mask1, 1.0, fdis)
		obj -= (C_nodup[mask] * auto_np.log(fdis)).sum()
		if auto_np.isnan(obj):
			raise ValueError("Function evaluation returns nan.")
		return obj


def convergence_criteria(f_k_list, f_k_len = 10, factr = 1e9):
	"""
	Convergence criteria for joint inference of alpha, beta and structure.
	"""
	if len(f_k_list) < f_k_len:
		return False
	else:
		f_k = f_k_list[-1]
		f_k1 = f_k_list[0]
		dif = np.abs(f_k - f_k1)
		dif = dif / max(np.abs(f_k), np.abs(f_k1), 1)
		return (dif <= factr * np.finfo(float).eps)


def max_poisson_likelihood(C, N, ini_x, ini_C1, idx_nodup, idx_dup, dup_times, idx_map, logger, round = 5000, alpha = -3.0, beta = 1.0, reg_weight = 0.05, maxiter = 10000, start_time_ = None, gt_structure = None):
	"""
	Poisson model caller
	"""
	if start_time_ is None:
		start_time_ = time.time()
	X_original = None
	if gt_structure != None:
		if args.matrix.endswith(".txt"):
			X_original = np.loadtxt(gt_structure)
		elif args.matrix.endswith(".npy"):
			X_original = np.load(gt_structure)
		else:
			raise OSError("Input structure must be in *.txt or *.npy format.")
	N_nodup = len(idx_nodup)
	C_nodup = C[np.ix_(idx_nodup, idx_nodup)]
	bounds_x = [(-1.0, 1.0) for i in range(N * 3)]
	alpha_bounds_ = np.array([[-100, 1e-2]])
	beta_bounds_ = np.array([[1e-2, 100]])
	X_ = ini_x
	C1_ = ini_C1
	obj_list = []

	S = []
	C_dup = []
	if N_nodup < N:
		S = np.zeros([C.shape[0] - N_nodup, N - N_nodup])
		s = 0
		for i in range(len(dup_times)):
			for j in range(s, s + dup_times[i]):
				S[i][j] = 1
			s += dup_times[i]
		C_dup = np.concatenate((C[np.ix_(idx_nodup, idx_dup)], C[np.ix_(idx_dup, idx_dup)]))

	alpha_gradient = grad(exponent_obj_alpha_auto)
	beta_gradient = grad(exponent_obj_beta_auto)
	poisson_gradient = grad(poisson_obj_reg_auto)
	gamma = reg_weight
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time_) + "Regularizer weight %f * N_e." %(gamma))
	for it in range(round):
		#print(f'Poisson iteration {it+1}')
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time_) + "Begin iteration %d." %(it + 1))
		
		# estimate alpha and beta by autograd
		"""
		results1 = optimize.fmin_l_bfgs_b(exponent_obj_alpha_auto, np.array([alpha]), fprime = alpha_gradient,
						args = (X_, C_nodup, S, C_dup, beta, True, gamma), bounds = alpha_bounds_, maxiter = 1000)
		alpha = results1[0][0]
		results2 = optimize.fmin_l_bfgs_b(exponent_obj_beta_auto, np.array([beta]), fprime = beta_gradient,
						args = (X_, C_nodup, S, C_dup, alpha, True, gamma), bounds = beta_bounds_, maxiter = 1000)
		beta = results2[0][0]
		f_alpha = results2[1]
		"""
		alpha, beta, f_alpha = estimate_alpha_beta(X_, C_nodup, S, C_dup, alpha, beta)
		if beta > 400.0 or beta < 0.01:
			return X_, results[1], alpha, beta
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time_) + "\tEstimated alpha = %f; beta = %f." %(alpha, beta))
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time_) + 
				"\tLog likelihood evaluated with alpha and beta at iteration %d = %f." %(it + 1, f_alpha))
		results = optimize.fmin_l_bfgs_b(poisson_obj_reg_auto, X_.flatten(), fprime = poisson_gradient, 
						args = (N, C_nodup, S, C_dup, idx_map, alpha, beta, True, gamma, ), 
						bounds = bounds_x, maxiter = maxiter)
		X_ = results[0].reshape(-1, 3)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time_) + "\tLog likelihood at iteration %d = %f." %(it + 1, results[1]))
		if gt_structure != None:
			structure1, structure2 = X_[idx_map], X_original
			rmsd, X1, X2, pcc = getTransformation(structure1, structure2) # structure1 is transformed
			# np.savetxt(f'simple_structure_{it}.txt', X1)
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time_) + "\tRMSD = %f\tPCC = %f" %(rmsd, pcc))
		if convergence_criteria(obj_list):
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time_) + "\tPoisson model optimization converges at iteration %d." %(it + 1))
			break
		elif len(obj_list) < 10:
			obj_list.append(results[1])
		else:
			obj_list.pop(0)
			obj_list.append(results[1])
	return X_, results[1], alpha, beta


def reconstruct_3D_structure(matrix, annotation, output_prefix, log_fn=None, reg=0.05, init_alpha=-3.0, 
							 num_repeats=5, save_repeats=False, max_rounds=1000, gt_structure=None, save_npy=False):
	"""
	Set up logging
	"""
	print(f"Reconstructing 3D structure with {num_repeats} random seeds ...")
	start_time = time.time()
	if not log_fn:
		log_fn = output_prefix + "_optimization.log"
	logger = create_logger('spatial_structure.py', log_fn)
	logger.info("Python version " + sys.version + "\n")
	function_param = f'reconstruct_3D_structure(matrix=\'{matrix}\', annotation=\'{annotation}\', output_prefix=\'{output_prefix}\', log_fn=\'{log_fn}\', reg={reg}, init_alpha={init_alpha}, num_repeats={num_repeats}, save_repeats={save_repeats}, max_rounds={max_rounds}, gt_structure=\'{gt_structure}\', save_npy={save_npy})'
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + function_param)

	"""
	Parameter check
	"""
	if num_repeats < 1 or num_repeats > 100:
		raise OSError("Number of repetitions must stay in range [1, 100].")
	if init_alpha >= 0:
		raise OSError("Initial value of alpha must be less than 0.")

	"""
	Load ecDNA matrix
	"""
	C = np.array([])
	if matrix.endswith(".txt"):
		C = np.loadtxt(matrix)
	elif matrix.endswith(".npy"):
		C = np.load(matrix)
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded normalized collapsed ecDNA matrix.")
	
	"""
	Construct Hi-C matrix with duplication
	"""
	N = -1 # Num bins in donor Hi-C
	bins = []
	row_labels = dict() # Map an interval of size RES to the index of all its copies
	fp = open(annotation, 'r')
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
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix annotations.")
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Expanded ecDNA matrix size: %d." %N)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Collapsed ecDNA matrix size: %d." %len(row_labels))				

	"""
	Run MDS with alpha = init_alpha to compute an initial X  
	"""
	bins = sorted(bins, key = lambda bin: row_labels[bin][0])
	idx_nodup = [bi for bi in range(len(bins)) if len(row_labels[bins[bi]]) == 1]
	idx_dup = [bi for bi in range(len(bins)) if len(row_labels[bins[bi]]) > 1]
	dup_times = np.array([len(row_labels[bins[bi]]) for bi in idx_dup])
	N_nodup = len(idx_nodup)
	C_nodup = C[np.ix_(idx_nodup, idx_nodup)]

	"""
	Initialize matrix with duplications
	"""
	ini_c = None
	if N_nodup < N:
		C_avg = dict()
		for bi1 in idx_nodup:
			for bi2 in idx_nodup:
				di1 = row_labels[bins[bi1]][0]
				di2 = row_labels[bins[bi2]][0]
				if di1 != di2:
					d_i12 = min(abs(di1 - di2), abs(min(di1, di2) - max(di1, di2) + N))
					if d_i12 not in C_avg:
						C_avg[d_i12] = [C[bi1][bi2]]
					else:
						C_avg[d_i12].append(C[bi1][bi2])
		min_c_avg = np.inf
		for d_ij in C_avg.keys():
			C_avg[d_ij] = np.percentile(C_avg[d_ij], 40)
			if C_avg[d_ij] < min_c_avg:
				min_c_avg = C_avg[d_ij]

		ci = 0
		ini_c = np.zeros([N, N - N_nodup])
		for bi_nodup in idx_nodup:
			cj = 0
			for bi_dup in idx_dup:
				total_int_ij = C[bi_nodup][bi_dup]
				di = row_labels[bins[bi_nodup]][0]
				d_ij = np.array([min(abs(di - dj), abs(min(di, dj) - max(di, dj) + N)) for dj in row_labels[bins[bi_dup]]], dtype = float)
				sum_d = sum(d_ij ** (init_alpha))
				for j_ in range(len(row_labels[bins[bi_dup]])):
					try:
						ini_c[ci][cj] = max(min(total_int_ij * 0.9, C_avg[int(d_ij[j_])]), total_int_ij / sum_d * (d_ij[j_] ** (init_alpha)))
					except:
						ini_c[ci][cj] = max(min(total_int_ij * 0.9, min_c_avg), total_int_ij / sum_d * (d_ij[j_] ** (init_alpha)))
					cj += 1 
			ci += 1
		for bi1 in idx_dup:
			ni1 = len(row_labels[bins[bi1]])
			cj = 0
			for bi2 in idx_dup:
				ni2 = len(row_labels[bins[bi2]])
				total_int_ij = C[bi1][bi2]
				d_ij = np.ones([ni1, ni2], dtype = float)
				for i_ in range(ni1):
					for j_ in range(ni2):
						di = row_labels[bins[bi1]][i_]
						dj = row_labels[bins[bi2]][j_]
						d_ij[i_][j_] = max(d_ij[i_][j_], min(abs(di - dj), abs(min(di, dj) - max(di, dj) + N)))
				sum_d = (d_ij ** (init_alpha)).sum()
				for i_ in range(ni1):
					for j_ in range(ni2):
						if i_ != j_:
							try:
								ini_c[ci + i_][cj + j_] = max(min(total_int_ij * 0.9, C_avg[int(d_ij[i_][j_])]), total_int_ij / sum_d * (d_ij[i_][j_] ** (init_alpha)))
							except:
								ini_c[ci + i_][cj + j_] = max(min(total_int_ij * 0.9, min_c_avg), total_int_ij / sum_d * (d_ij[i_][j_] ** (init_alpha)))
						else:
							ini_c[ci + i_][cj + j_] = total_int_ij / sum_d * (d_ij[i_][j_] ** (init_alpha))
				cj += ni2 
			ci += ni1
		#ini_c[ini_c < 1.0] = 1.0
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Initialized expanded ecDNA matrix for MDS.")
	
	"""
	Map bin indices for optimization: place the bins without duplication in front
	"""
	i_nodup, i_dup = 0, 0
	idx_map = dict()
	for bi in range(len(bins)):
		if bi in idx_nodup:
			idx_map[row_labels[bins[bi]][0]] = i_nodup
			i_nodup += 1
		else:
			for i_ in range(len(row_labels[bins[bi]])):
				idx_map[row_labels[bins[bi]][i_]] = N_nodup + i_dup
				i_dup += 1
	idx_map = np.array([idx_map[i] for i in range(len(idx_map))])
	
	repeat = 1
	repeat_ = 1
	PM_X_min = np.zeros(N * 3)
	PM_obj_min = np.inf
	best_alpha = -3.0
	best_beta = 1.0
	while repeat < num_repeats + 1:
		print(f'Repeat {repeat}')
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Repeat %d:" %repeat)

		"""
		Run MDS
		"""
		MDS_X1, MDS_X2 = mds(C, N, idx_nodup, idx_dup, dup_times, ini_c, alpha = init_alpha)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "MDS optimization completed.")
		""" Disabled saving of MDS output
		if gt_structure == None:
			np.savetxt(f'{output_prefix}_mds_{repeat}_3d.txt', MDS_X1[idx_map])
		else:
			original, reconstructed = np.loadtxt(gt_structure), MDS_X1[idx_map]
			scale_factor = calculate_average_distance(original)
			fr_pos_array = original / scale_factor
			mds_pos_array = reconstructed / scale_factor
			fr_pos_array = remove_nan_col(fr_pos_array)
			mds_pos_array = remove_nan_col(mds_pos_array)
			rmsd, X1, X2, pcc = getTransformation(mds_pos_array,fr_pos_array)
			np.savetxt(f'{output_prefix}_mds_{repeat}_3d.txt', X1)
		"""
		"""
		Run Poisson model with initial X and matrix returned from MDS
		"""
		try:
			PM_X, PM_obj, alpha, beta = max_poisson_likelihood(C, N, MDS_X1, MDS_X2, idx_nodup, idx_dup, dup_times, idx_map, logger, max_rounds, start_time_ = start_time, gt_structure = gt_structure, alpha = init_alpha, reg_weight = reg)
			if save_repeats:
				np.savetxt(output_prefix + "_repeat" + str(repeat) + '_coordinates.txt', PM_X)
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Poisson model optimization completed.")
			if PM_obj < PM_obj_min:
				if np.isinf(PM_obj_min):
					logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Reset the current best repetition to %d." %(repeat))
				else:
					logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Repetition %d showed a better objective than current best repetition %d." %(repeat, repeat_)) 
				PM_obj_min = PM_obj
				PM_X_min = PM_X
				repeat_ = repeat
				best_alpha = alpha
				best_beta = beta
			repeat += 1
		except:
			continue
	
	"""
	Write output to file
	"""
	if N_nodup < N:
		"""
		Reorder the result matrix
		"""
		PM_X_min = PM_X_min[idx_map]
	if save_npy:
		output_coordinates_fn = output_prefix + "_coordinates.npy"
		np.save(output_coordinates_fn, PM_X_min)
	else:
		output_coordinates_fn = output_prefix + "_coordinates.txt"
		np.savetxt(output_coordinates_fn, PM_X_min)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the resolved 3D structure to %s." %output_coordinates_fn)
	output_params_fn = output_prefix + "_hyperparameters.txt"
	fp_w = open(output_params_fn, 'w')
	fp_w.write("alpha\t%f\n" %best_alpha)
	fp_w.write("beta\t%f\n" %best_beta)
	fp_w.close()
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the hyperparameters to %s." %output_params_fn)
	if gt_structure != None:
		structure1, structure2 = PM_X_min, np.loadtxt(gt_structure)
		rmsd, X1, X2, pcc = getTransformation(structure1, structure2) # structure1 is transformed
		if save_npy:
			output_coordinates_fn = output_prefix + "_aligned_coordinates.npy"
			np.save(output_coordinates_fn, X1)
		else:
			output_coordinates_fn = output_prefix + "_aligned_coordinates.txt"
			np.savetxt(output_coordinates_fn, X1)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the aligned 3D structure to %s." %output_coordinates_fn)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")
	print("3D structure reconstruction is done. The 3D structure is saved to %s." %output_coordinates_fn)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Compute the 3D coordinates from Hi-C.")
	parser.add_argument("--matrix", help = "Input collapsed Hi-C matrix, in *.txt or *.npy format.", required = True)
	parser.add_argument("--annotation", help = "Annotation of bins in the input matrix.", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--log_fn", help = "Name of log file.")
	parser.add_argument("--reg", help = "Regularizer weight.", type = float, default = 0.05)
	parser.add_argument("--init_alpha", help = "An initial guess of alpha, for initialization and MDS.", type = float, default = -3.0)
	parser.add_argument("--num_repeats", help = "Number of repetitions with random initial structures.", type = int, default = 5)
	parser.add_argument("--save_repeats", help = "Save the reconstructed structure in each repeat.", action = "store_true")
	parser.add_argument("--max_rounds", help = "Maximum number of rounds for Poisson model.", type = int, default = 1000)
	parser.add_argument("--gt_structure", help = "Input the ground truth structure, in *.txt or *.npy format, for calculating RMSD and PCC.")
	parser.add_argument("--save_npy", help = "Save matrices to *.npy format", action = "store_true")
	
	args = parser.parse_args()
	reconstruct_3D_structure(**vars(args))