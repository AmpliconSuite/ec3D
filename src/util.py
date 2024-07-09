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


def calculate_average_distance(array):
	diff_array = array[:, None, :] - array[None, :, :]
	distance_array = np.sqrt((diff_array**2).sum(-1))
	average_distance = np.mean(distance_array)
	return average_distance


def remove_nan_col(cord):
	index = np.isnan(cord).all(axis=1)
	cord = np.delete(cord, index, axis=0)
	return cord


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


def getTransformation(X, Y, centering = True, scaling = True, reflection = False):
	"""
	kabsch method: Recovers transformation needed to align structure1 with structure2.
	"""
	X = X.copy()
	Y = Y.copy()

	X = X.T
	Y = Y.T
	centroid_X = X.mean(axis = 1, keepdims = True)
	centroid_Y = Y.mean(axis = 1, keepdims = True)

	X = X - centroid_X
	Y = Y - centroid_Y

	if scaling:
		scale_X = np.sqrt((X ** 2).sum() / X.shape[1])
		scale_Y = np.sqrt((Y ** 2).sum() / Y.shape[1])
		to_scale = scale_Y if type(scaling) is bool else float(scaling)
		X = X / (scale_X / to_scale)
		Y = Y / (scale_Y / to_scale)

	C = np.dot(X, Y.transpose())
	V, _, Wt = np.linalg.svd(C)

	I = np.eye(3)
	if reflection:
		d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
		I[2, 2] = d

	U = np.dot(Wt.T, np.dot(I, V.T))
	X = np.dot(U, X)

	if not centering:
		X = X + centroid_Y
		Y = Y + centroid_Y
	dx = euclidean_distances(X.T)
	dy = euclidean_distances(Y.T)
	pr = pearson(dx, dy)
	# print(rmsd(X.T, Y.T))
	return rmsd(X.T, Y.T), X.T, Y.T, pr


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Compute RMSD and PCC.")
	parser.add_argument("--structure1", help = "Input the first structure.", required = True)
	parser.add_argument("--structure2", help = "Input the second structure.", required = True)
	parser.add_argument("--save", help = "Input the second structure.", type=bool, default=False, required = False)
	args = parser.parse_args()

	structure1, structure2 = np.loadtxt(args.structure1)[51:67], np.loadtxt(args.structure2)[71:87]
	rmsd, X1, X2, pcc = getTransformation(structure1, structure2)
	print(f'RMSD: {rmsd}, PCC: {pcc}')
	if args.save:
		np.savetxt('structure1.txt', X1)
		np.savetxt('structure2.txt', X2)

