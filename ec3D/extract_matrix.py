"""
Extract ecDNA matrix from whole genome Hi-C
"""
import sys
import numpy as np
import argparse
import cooler
import logging
import time
from iced import normalization

from util import *

def extract_matrix(cool, ecdna_cycle, resolution, output_prefix, log_fn=None, save_npy=False):
	"""
	Set up logging
	"""
	print("Extracting Hi-C submatrices of amplified regions ...")
	start_time = time.time()
	if not log_fn:
		log_fn = output_prefix + "_preprocessing.log"
	logging.basicConfig(filename = log_fn, filemode = 'w', level = logging.DEBUG, 
						format = '[%(name)s:%(levelname)s]\t%(message)s')
	logging.info("Python version " + sys.version + "\n")
	function_param = f'extract_matrix(cool=\'{cool}\', ecdna_cycle=\'{ecdna_cycle}\', resolution={resolution}, output_prefix=\'{output_prefix}\', log_fn=\'{log_fn}\', save_npy={save_npy})'
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + function_param)

	"""
	Read in ecDNA cycle
	"""
	intrvls = read_ecDNA_cycle(ecdna_cycle, resolution)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "ecDNA involves %d amplified intervals with resolution %d." %(len(intrvls), resolution))
	for intrvl in intrvls:
		logging.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + "\tAmplified interval %s." %intrvl)

	"""
	Extract the expanded matrix from the input cool file
	"""
	clr = cooler.Cooler(cool)
	chr_prefix = True
	if '1' in clr.chromnames:
		chr_prefix = False
	row_labels = dict() # Map an interval of size RES to the index of all its copies
			# Bins with negative orientation already reordered
	N = 0 # Total num bins
	for intrvl in intrvls:
		intrvl_size = (intrvl[2] - intrvl[1]) // resolution
		i = 0
		if intrvl[3] == '+':
			for b in range(intrvl[1], intrvl[2], resolution):
				try:
					row_labels[(intrvl[0], b)].append(N + i)
				except:
					row_labels[(intrvl[0], b)] = [N + i]
				i += 1
		else:
			for b in range(intrvl[1], intrvl[2], resolution):
				try:
					row_labels[(intrvl[0], b)].append(N + intrvl_size - i - 1)
				except:
					row_labels[(intrvl[0], b)] = [N + intrvl_size - i - 1]
				i += 1
		N += intrvl_size
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "There are %d bins in total in the expanded matrix." %N)
	bins = row_labels.keys()
	D = np.zeros([N, N])
	s1 = 0
	for int1 in intrvls:
		if int1[1] == int1[2]:
			continue
		intrvl_string = int1[0] + ":" + str(int1[1]) + "-" + str(int1[2])
		if not chr_prefix:
			intrvl_string = intrvl_string[3:]
		s2 = 0
		for int2 in intrvls:
			if int2[1] == int2[2]:
				continue
			intrvl_string_ = int2[0] + ":" + str(int2[1]) + "-" + str(int2[2])
			if not chr_prefix:
				intrvl_string_ = intrvl_string_[3:]
			mat_ = clr.matrix(balance = False, sparse = True).fetch(intrvl_string, intrvl_string_)
			for i, j, v in zip(mat_.row, mat_.col, mat_.data):
        			D[s1 + i][s2 + j] = v
			s2 += (int2[2] - int2[1]) // resolution
		s1 += (int1[2] - int1[1]) // resolution
	D = reorder_bins(D, intrvls, resolution) # D: N * N
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Extracted the expanded matrix from the input cool file.")

	"""
	Create the collapsed matrix, ICE normalization
	"""
	idx_dedup = [row_labels[bin][0] for bin in bins]
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "There are %d bins in total in the collapsed matrix." %len(idx_dedup))
	idx_dedup_argsort = np.argsort(idx_dedup)
	idx_dedup_sorted = [idx_dedup[i] for i in idx_dedup_argsort]
	D_dedup = D[np.ix_(idx_dedup_sorted, idx_dedup_sorted)]
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Created the collapsed matrix from expanded matrix.")
	if save_npy:
		np.save(output_prefix + "_raw_collapsed_matrix.npy", D_dedup)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the raw collapsed matrix to %s." %(output_prefix + "_raw_collapsed_matrix.npy"))
	else:
		np.savetxt(output_prefix + "_raw_collapsed_matrix.txt", D_dedup)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the raw collapsed matrix to %s." %(output_prefix + "_raw_collapsed_matrix.txt"))
	N_dedup = normalization.ICE_normalization(D_dedup, counts_profile = np.array([len(row_labels[bin]) for bin in bins])[idx_dedup_argsort])
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Normalized the collapsed matrix.")
	if save_npy:
		np.save(output_prefix + "_collapsed_matrix.npy", N_dedup)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the normalized collapsed matrix to %s." %(output_prefix + "_collapsed_matrix.npy"))
	else:
		np.savetxt(output_prefix + "_collapsed_matrix.txt", N_dedup)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the normalized collapsed matrix to %s." %(output_prefix + "_collapsed_matrix.txt"))
	fp = open(output_prefix + "_annotations.bed", 'w')
	for bin in bins:
		fp.write("%s\t%d\t%d\t" %(bin[0], bin[1], bin[1] + resolution))
		for idx in row_labels[bin][:-1]:
			fp.write("%d\t" %idx)
		fp.write("%d\n" %row_labels[bin][-1])
	fp.close()
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the annotation of bins to %s." %(output_prefix + "_annotations.bed"))
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")
	print("Matrix extraction is done. The collapsed matrix is saved to %s." %(output_prefix + "_collapsed_matrix.npy" if save_npy else output_prefix + "_collapsed_matrix.txt"))

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = "Extract Hi-C matrix correspond to ecDNA intervals.")
	parser.add_argument("--cool", help = "Input whole genome Hi-C map, in *.cool format.", required = True)
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed (chr, start, end, orientation) format.", required = True)
	parser.add_argument("--resolution", help = "Bin size.", type = int, required = True)
	parser.add_argument("--output_prefix", help = "Prefix of the output files.", required = True)
	parser.add_argument("--log_fn", help = "Name of log file.")
	parser.add_argument("--save_npy", help = "Save matrices to *.npy format", action = "store_true")
	
	args = parser.parse_args()
	extract_matrix(**vars(args))
	

