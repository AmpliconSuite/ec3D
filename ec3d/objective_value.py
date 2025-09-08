import sys
import numpy as np
from scipy import optimize
import argparse

try:
	from ec3d.spatial_structure import poisson_obj_reg_auto
except:
	from spatial_structure import poisson_obj_reg_auto

def get_objective_value(X_fn, C_fn, annotation_fn, alpha, beta, reg, gamma):
	X = np.loadtxt(X_fn)
	C = np.load(C_fn)

	N = -1 # Num bins in donor Hi-C
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
	
	bins = sorted(bins, key = lambda bin: row_labels[bin][0])
	idx_nodup = [bi for bi in range(len(bins)) if len(row_labels[bins[bi]]) == 1]
	idx_dup = [bi for bi in range(len(bins)) if len(row_labels[bins[bi]]) > 1]
	dup_times = np.array([len(row_labels[bins[bi]]) for bi in idx_dup])
	N_nodup = len(idx_nodup)
	C_nodup = C[np.ix_(idx_nodup, idx_nodup)]

	i_nodup, i_dup = 0, 0
	idx_map = dict()
	idx_map_inverse = dict()
	for bi in range(len(bins)):
		if bi in idx_nodup:
			idx_map[row_labels[bins[bi]][0]] = i_nodup
			idx_map_inverse[i_nodup] = row_labels[bins[bi]][0]
			i_nodup += 1
		else:
			for i_ in range(len(row_labels[bins[bi]])):
				idx_map[row_labels[bins[bi]][i_]] = N_nodup + i_dup
				idx_map_inverse[N_nodup + i_dup] = row_labels[bins[bi]][i_]
				i_dup += 1
	idx_map = np.array([idx_map[i] for i in range(len(idx_map))])
	idx_map_inverse = np.array([idx_map_inverse[i] for i in range(len(idx_map_inverse))])

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
	# optimal_obj = poisson_obj_(X.flatten(), N, C_nodup, S, C_dup, idx_map, alpha, beta)
	optimal_obj = poisson_obj_reg_auto(X[idx_map_inverse].flatten(), N, C_nodup, 0, S = S, C_dup = C_dup, idx_map=idx_map, alpha = alpha, beta = beta, reg = reg, gamma = gamma)
	return optimal_obj

if __name__ == "__main__":
	# alpha, beta = [], []
	# for line in open('/home/chaohuili/ecDNA_structure/data/duplication/sim_para.csv', 'r').readlines()[1:]:
	# 	info = line.strip().split(',')
	# 	alpha.append(-float(info[-2]))
	# 	beta.append(float(info[-1]))
	# f = open('duplication_init_obj.csv', 'w')
	# for i in range(1, 16):
	# 	print(i, alpha[i-1], beta[i-1])
	# 	X_fn = f'/home/chaohuili/ecDNA_structure/results/test/duplication{i}_init_3d.txt'
	# 	C_fn = f'/home/chaohuili/ecDNA_structure/data/duplication/duplication{i}_HiC.txt'
	# 	# annotation_fn = f'/home/chaohuili/ecDNA_structure/results/comparison/ecDNA_annotations.bed'
	# 	annotation_fn = f'/home/chaohuili/ecDNA_structure/data/duplication/duplication{i}_annotations.bed'
	# 	obj_value = get_objective_value(X_fn, C_fn, annotation_fn, -3, 1)
	# 	f.write(f'{i},{obj_value}\n')
	# f.close()
	# # X_fn = f'/home/chaohuili/ecDNA_structure/results/test/simple_simulation.txt'
	# # C_fn = f'/home/chaohuili/ecDNA_structure/results/test/simple_HiC.txt'
	# # annotation_fn = f'/home/chaohuili/ecDNA_structure/results/test/simple_annotations.bed'
	# # obj_value = get_objective_value(X_fn, C_fn, annotation_fn, -2, 10)
	# # print(obj_value)
	parser = argparse.ArgumentParser(description = "Compute RMSD and PCC.")
	parser.add_argument("--structure", help = "Input the structure.", required = True)
	parser.add_argument("--matrix", help = "Input the Hi-C matrix.", required = True)
	parser.add_argument("--annotation", help = "Input the annotations file.", required = True)
	parser.add_argument("--a", help = "Input the alpha value.", type=float, required = True)
	parser.add_argument("--b", help = "Input the beta value.", type=float, required = True)
	# parser.add_argument("--reg", help = "Including the regularizer or not.", type=bool, required = True)
	# parser.add_argument("--g", help = "Input the gamma value.", type=float, required = True)
	args = parser.parse_args()

	X_fn, C_fn, annotation_fn = args.structure, args.matrix, args.annotation
	alpha, beta = args.a, args.b

	reg, gamma = True, 0.0
	obj_value = get_objective_value(X_fn, C_fn, annotation_fn, alpha, beta, reg, gamma)
	print(f'Objective value (without regularizer): {obj_value}')

	
	

	
