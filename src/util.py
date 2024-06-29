
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


def reorder_bins(matrix, intrvls):
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

