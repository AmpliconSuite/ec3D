import numpy as np
import argparse
from sklearn.metrics import euclidean_distances
from scipy import stats


chr_idx = {'1': 0, '2': 1, '3': 2, '4': 3,
        '5': 4, '6': 5, '7': 6, '8': 7,
        '9': 8, '10': 9, '11': 10, '12': 11,
        '13': 12, '14': 13, '15': 14, '16': 15,
        '17': 16, '18': 17, '19': 18, '20': 19,
        '21': 20, '22': 21, 'X': 22, 'Y': 23, 'M': 24,
	'chr1': 0, 'chr2': 1, 'chr3': 2, 'chr4': 3,
	'chr5': 4, 'chr6': 5, 'chr7': 6, 'chr8': 7,
	'chr9': 8, 'chr10': 9, 'chr11': 10, 'chr12': 11,
	'chr13': 12, 'chr14': 13, 'chr15': 14, 'chr16': 15,
	'chr17': 16, 'chr18': 17, 'chr19': 18, 'chr20': 19,
	'chr21': 20, 'chr22': 21, 'chrX': 22, 'chrY': 23, 'chrM': 24}


def read_ecDNA_cycle(fn, res):
	intrvls = []
	fp = open(fn, 'r')
	for line in fp:
		s = line.strip().split()
		if s[0] in chr_idx:
			intrvls.append([s[0], round(float(s[1]) / res) * res, round(float(s[2]) / res) * res, s[3]])
	fp.close()
	return intrvls


def reorder_bins(matrix, intrvls, res):
	"""
	Correct the order of bins for segment with orientation '-'
	"""
	start = 0
	for i in range(len(intrvls)):
		intrvl_size = (intrvls[i][2] - intrvls[i][1]) // res
		if intrvls[i][3] == '-':
			matrix[:, start: start + intrvl_size] = matrix[:, start: start + intrvl_size][:, ::-1]
			matrix[start: start + intrvl_size, :] = matrix[start: start + intrvl_size, :][::-1, :]
		start += intrvl_size
	return matrix


def rmsd(X, Y):
	"""
	Calculate the RMSD between X and Y
	X, Y are two N * 3 matrix
	Return:
		RMSD: float
	"""
	n, _ = X.shape
	RMSD = (((X - Y) ** 2).sum() / n) ** 0.5
	return RMSD


def pearson(mat1, mat2): 
	## Pearson Correlation measures the similarity in shape between two profiles
	assert mat1.shape == mat2.shape
	#convert to vectors
	vec1 = mat1.flatten()
	vec2 = mat2.flatten()

	#remove zeroes
	nonzero = [i for i in range(len(vec1)) if vec1[i] != 0 and vec2[i] != 0]
	vec1 = vec1[nonzero]
	vec2 = vec2[nonzero]

	r, p = stats.pearsonr(vec1, vec2) # spearmanr
	return r


def normalize_structure(structure):
	"""
	Normalize the structure to have unit variance.
	"""
	max_distance = np.linalg.norm(structure, axis = 0).max()
	return structure / max_distance


def getTransformation(X, Y, centering = True, scaling = True, reflection = False):
	"""
	kabsch method: Recovers transformation needed to align structure1 with structure2.
	"""
	X = X.copy()
	Y = Y.copy()
	X = X.T
	Y = Y.T

	if centering:
		centroid_X = X.mean(axis = 1, keepdims = True)
		centroid_Y = Y.mean(axis = 1, keepdims = True)
		X = X - centroid_X
		Y = Y - centroid_Y
	
	if scaling:
		X = normalize_structure(X)
		Y = normalize_structure(Y)

	C = np.dot(X, Y.transpose())
	V, _, Wt = np.linalg.svd(C)

	I = np.eye(3)
	if reflection:
		d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
		I[2, 2] = d

	U = np.dot(Wt.T, np.dot(I, V.T))
	X = np.dot(U, X)

	dx = euclidean_distances(X.T)
	dy = euclidean_distances(Y.T)
	pr = pearson(dx, dy)
	# print(rmsd(X.T, Y.T))
	return rmsd(X.T, Y.T), X.T, Y.T, pr

# 1. Scaling
def scale(points, scale_factors):
	scale_matrix = np.diag(scale_factors)
	return np.dot(points, scale_matrix)

# 2. Rotation
def rotate(points, theta_x, theta_y, theta_z):
	# Rotation matrix around x-axis
	R_x = np.array([[1, 0, 0],
					[0, np.cos(theta_x), -np.sin(theta_x)],
					[0, np.sin(theta_x), np.cos(theta_x)]])
	
	# Rotation matrix around y-axis
	R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
					[0, 1, 0],
					[-np.sin(theta_y), 0, np.cos(theta_y)]])
	
	# Rotation matrix around z-axis
	R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
					[np.sin(theta_z), np.cos(theta_z), 0],
					[0, 0, 1]])
	
	# Combined rotation matrix
	R = np.dot(R_z, np.dot(R_y, R_x))
	
	return np.dot(points, R)

# 3. Translation
def translate(points, translation_vector):
	return points + translation_vector

# randomly sample pairs of regions from the same structure
def random_sampling1():
	import csv
	import pandas as pd
	import random
	duplication = {'D458':[{'s1':9, 'e1':12, 's2':370, 'e2':373, 'size':4, 'RMSD':0.1600, 'PCC':0.9973}, 
						{'s1':115, 'e1':156, 's2':415, 'e2':374, 'size':42, 'RMSD':0.2020, 'PCC':0.9581}, 
						{'s1':236, 'e1':239, 's2':416, 'e2':419, 'size':4, 'RMSD':0.1560, 'PCC':0.9923}, 
						{'s1':240, 'e1':255, 's2':369, 'e2':354, 'size':16, 'RMSD':0.1251, 'PCC':0.9973}, 
						{'s1':341, 'e1':351, 's2':497, 'e2':507, 'size':11, 'RMSD':0.0756, 'PCC':0.9967}], 
					'H2170':[{'s1':18, 'e1':96, 's2':364, 'e2':442, 'size':79, 'RMSD':0.2759, 'PCC':0.9319}, 
			  				{'s1':260, 'e1':281, 's2':342, 'e2':363, 'size':22, 'RMSD':0.3363, 'PCC':0.8175}, 
							{'s1':97, 'e1':259, 's2':443, 'e2':605, 'size':163, 'RMSD':0.3020, 'PCC':0.7758}]}

	sample = 'D458'
	structure = np.loadtxt(f'/home/chaohuili/ecDNA_structure/results/{sample}/{sample}_5k_coordinates.txt')
	# file_writer = csv.writer(open(f'{sample}_similarity.tsv', 'w'), delimiter='\t')
	# file_writer.writerow(['Sample', 'Region1', 'Region2', 'Size(#bins)', 'RMSD', 'PCC'])
	df = pd.read_table(f'/home/chaohuili/ecDNA_structure/results/similarity/{sample}_similarity.tsv')
	for i in range(len(duplication[sample])):
		print(f'cell line: {sample}, duplication: {i+1}')
		dup = duplication[sample][i]
		structure1 = structure[dup['s1']:dup['e1']+1]
		inverse = False
		if dup['s2'] > dup['e2']:
			inverse = True
			dup['s2'], dup['e2'] = dup['e2'], dup['s2']		
		structure2 = structure[dup['s2']:dup['e2']+1]
		if inverse:
			structure2 = structure2[::-1]
		rmsd_test, _, _, pcc_test = getTransformation(structure1, structure2)
		print(f'rmsd_test: {rmsd_test}, pcc_test: {pcc_test}')

		s1, e1, size = dup['s1'], dup['e1'], dup['size']
		id = f'[{s1}, {e1}]'
		sample_size, cnt1, cnt2 = 5000, df[(df['Region1'] == id) & (df['RMSD'] < rmsd_test) & (df['PCC'] > pcc_test)].shape[0], 0

		for _ in range(0):
			# Sample random pairs
			# a, b = random.sample(list(range(size, structure.shape[0]-size)), 2)
			# if a > b:
			# 	a, b = b, a
			# structure1, structure2 = structure[a-size:a], structure[b:b+size]

			# Sample random structures
			pos = s1
			while (s1<=pos and pos<=e1) or (dup['s2']<=pos and pos<=dup['e2']):
				pos = random.choice(list(range(structure.shape[0]-size)))
			structure2 = structure[pos:pos+size]
			if inverse:
				structure2 = structure2[::-1]
			rmsd, _, _, pcc = getTransformation(structure1, structure2) # structure1 is transformed
			# file_writer.writerow([sample, f'[{s1}, {e1}]', f'[{pos}, {pos+size-1}]', size, rmsd, pcc])
			if rmsd < rmsd_test and pcc > pcc_test:
				cnt1 += 1
			# if pcc > pcc_test:
			# 	cnt2 += 1
			# print(f'RMSD: {rmsd}, PCC: {pcc}')
		p1, p2 = cnt1/sample_size, cnt2/sample_size
		print(f"p1: {p1}, p2: {p2}")

# randomly sample pairs of regions from two structures
def radom_sampling2():
	import random
	import csv
	structure1 = np.loadtxt(f'/home/chaohuili/ecDNA_structure/results/IMR575/IMR575_5k_coordinates.txt')
	structure2 = np.loadtxt(f'/home/chaohuili/ecDNA_structure/results/IMR575/IMR575_5k_coordinates.txt')
	file_writer = csv.writer(open(f'IMR575_duplication_similarity.tsv', 'w'), delimiter='\t')
	file_writer.writerow(['Sample', 'Region1', 'Region2', 'Size(#bins)', 'RMSD', 'PCC'])

	rmsd_test, _, _, pcc_test = getTransformation(structure1[6:100], structure2[100:194])
	print(f'rmsd_test: {rmsd_test}, pcc_test: {pcc_test}')

	s1, e1, s2, e2, size = 6, 99, 100, 193, 94
	sample_size, cnt1, cnt2 = 5000, 0, 0
	for _ in range(sample_size):
		# Sample random structures
		pos = s2
		while (s2<=pos and pos<=e2):
			pos = random.choice(list(range(structure2.shape[0]-size)))
		rmsd, _, _, pcc = getTransformation(structure1[6:100], structure2[pos:pos+size]) # structure1 is transformed
		file_writer.writerow(['IMR575', f'[{s1}, {e1}]', f'[{pos}, {pos+size-1}]', size, rmsd, pcc])
		if rmsd < rmsd_test and pcc > pcc_test:
			cnt1 += 1
		# if pcc > pcc_test:
		# 	cnt2 += 1
		# print(f'RMSD: {rmsd}, PCC: {pcc}')
	p1, p2 = cnt1/sample_size, cnt2/sample_size
	print(f"p1: {p1}, p2: {p2}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Compute RMSD and PCC.")
	parser.add_argument("--structure1", help = "Input the first structure.", required = True)
	parser.add_argument("--structure2", help = "Input the second structure.", required = True)
	parser.add_argument("--save", help = "Input the second structure.", type=bool, default=False, required = False)
	args = parser.parse_args()

	# random_sampling1()
	radom_sampling2()
	exit(0)
	structure1, structure2 = np.loadtxt(args.structure1), np.loadtxt(args.structure2)

	rmsd, X1, X2, pcc = getTransformation(structure1, structure2) # structure1 is transformed
	print(f'RMSD: {rmsd}, PCC: {pcc}')
	if args.save:
		np.savetxt('structure1.txt', X1)
		np.savetxt('structure2.txt', X2)

