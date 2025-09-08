"""
Visualize significant interactions within ecDNA
Author: Biswanath Chowdhury
"""
import os
import sys
import argparse
import time
import numpy as np

try:
	from ec3d.util import create_logger, read_ecDNA_cycle
except:
	from util import create_logger, read_ecDNA_cycle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = [10, 10]
rcParams['pdf.fonttype'] = 42

def plot_significant_interactions(ecdna_cycle, resolution, matrix, output_prefix,
								  annotation = None, interactions = None, sv_list = None,
								  fontsize = 24, min_segment_ratio = 8, plot_collapsed_matrix = False,
								  log_fn = None):
	"""
	Set up logging
	"""
	print("Plotting significant interactions ...")
	start_time = time.time()
	log_fn = ""
	if not log_fn:
		log_fn = output_prefix + "_visualize_interactions.log"
	else:
		log_fn = log_fn
	logger = create_logger('plot_interactions.py', log_fn)
	logger.info("Python version " + sys.version + "\n")
	function_param = f'plot_significant_interactions(ecdna_cycle=\'{ecdna_cycle}\', resolution={resolution}, matrix=\'{matrix}\', output_prefix=\'{output_prefix}\', annotation=\'{annotation}\', interactions=\'{interactions}\', sv_list=\'{sv_list}\', fontsize={fontsize}, min_segment_ratio={min_segment_ratio}, plot_collapsed_matrix={plot_collapsed_matrix}, log_fn=\'{log_fn}\')'
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + function_param)

	"""
	Read in ecDNA cycle
	"""
	res = resolution
	intrvls = read_ecDNA_cycle(ecdna_cycle, res)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "ecDNA involves %d amplified intervals with resolution %d." %(len(intrvls), res))
	
	"""
	Optional: Read in annotations
	"""
	bins = []
	row_labels = dict() # Map an interval of size RES to the index of all its copies
	N = 0
	idx_dedup = []
	if plot_collapsed_matrix or sv_list:
		if annotation == 'None':
			raise ValueError("Annotation file is required.")
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
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix annotations.")
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Expanded ecDNA matrix size: %d." %N)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Collapsed ecDNA matrix size: %d." %len(row_labels))
	if plot_collapsed_matrix:
		idx_dedup = sorted([row_labels[bin][0] for bin in bins])
		idx_dedup_map = {row_labels[bin][0]: bin for bin in bins}
		intrvls = []
		bin0 = idx_dedup_map[idx_dedup[0]]
		intrvl = [bin0[0], bin0[1], bin0[1] + res]
		for i in range(1, len(idx_dedup)):
			if idx_dedup_map[idx_dedup[i]][1] == idx_dedup_map[idx_dedup[i - 1]][1] + res:
				intrvl[2] += res
			elif idx_dedup_map[idx_dedup[i]][1] == idx_dedup_map[idx_dedup[i - 1]][1] - res:
				intrvl[1] -= res
			else:
				intrvls.append(intrvl)
				intrvl = [idx_dedup_map[idx_dedup[i]][0], idx_dedup_map[idx_dedup[i]][1], idx_dedup_map[idx_dedup[i]][1] + res]
		intrvls.append(intrvl)	
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Reset amplified intervals according to collapsed matrix.")

	# Calculate the midpoints for x-tick and y-tick labels
	start_pos = []
	mid_pos = []
	xticklbl = []
	yticklbl = []
	num_bins = 0
	for intrvl in intrvls:
		intrvl_len = (intrvl[2] - intrvl[1]) // res
		if intrvl_len >= min_segment_ratio:
			start_label = f"{intrvl[1]/1e6:.1f}".rstrip('0').rstrip('.') if intrvl[1] % 1e6 != 0 else f"{int(intrvl[1]/1e6)}"
			end_label = f"{intrvl[2]/1e6:.1f}".rstrip('0').rstrip('.') if intrvl[2] % 1e6 != 0 else f"{int(intrvl[2]/1e6)}"
			xticklbl.append(f"{intrvl[0]}:{start_label}-{end_label}Mb")
			yticklbl.append(num_bins)
		else:
			xticklbl.append('')  # Empty string for segments too small to label
		mid_pos.append(num_bins + intrvl_len * 0.5)
		num_bins += intrvl_len
		start_pos.append(num_bins)
		logger.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + "\tAmplified interval %s." %intrvl)
	yticklbl.append(num_bins)

	"""
	Load Hi-C matrix
	"""
	data = np.array([])
	if matrix.endswith(".txt"):
		data = np.loadtxt(matrix)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix without duplication, in txt format.")
	elif matrix.endswith(".npy"):
		data = np.load(matrix)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix without duplication, in npy format.")
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")
	if plot_collapsed_matrix:
		if data.shape[0] != len(row_labels) or data.shape[1] != len(row_labels):
			raise OSError("Input matrix must be in size %d * %d." %(len(row_labels), len(row_labels)))

	"""
	Visualize Hi-C and interactions
	"""
	fontsize = fontsize  
	fig, ax = plt.subplots()
	log_data = np.ma.log10(np.ma.masked_where(data == 0.0, data))

	# Display heatmap with bin numbers
	im = ax.matshow(log_data, cmap = 'YlOrRd')
	if plot_collapsed_matrix:
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Plotted collapsed matrix.")
	else:
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Plotted expanded matrix.")

	# Create a color bar with consistent font size
	cbar = fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
	cbar.ax.tick_params(labelsize = fontsize)  # Set font size for color bar ticks

	# Draw lines at the bin edges
	for pos in start_pos[: -1]:
		ax.axhline(y = pos, color = 'black', linestyle = '--', linewidth = 0.4, clip_on = False)
		ax.axvline(x = pos, color = 'black', linestyle = '--', linewidth = 0.4, clip_on = False)

	# Plot significant interactions
	if interactions:
		si_x, si_y = [], []
		fp = open(interactions, 'r')
		if plot_collapsed_matrix:
			idx_map = dict()
			for bin in bins:
				for idx in row_labels[bin]:
					idx_map[idx] = idx_dedup.index(row_labels[bin][0])
			for line in fp:
				s = line.strip().split('\t')
				try:
					bin1 = idx_map[int(s[0])]
					bin2 = idx_map[int(s[1])]
					if bin1 > bin2:
						bin1, bin2 = bin2, bin1
					si_x.append(bin2)
					si_y.append(bin1)
				except:
					pass
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Mapped significant interactions to collapsed matrix.")
		else:
			for line in fp:
				s = line.strip().split('\t')
				try:
					si_x.append(int(s[1]))
					si_y.append(int(s[0]))
				except:
					pass
		fp.close()
		ax.plot(si_x, si_y, 'o', markeredgewidth = 1, ms = 5, markerfacecolor = "None", markeredgecolor = 'b')
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Plotted significant interactions.")

	# Optional: plot additional SVs
	if sv_list:	
		sv_x, sv_y = [], []
		fp = open(sv_list, 'r')
		for line in fp:
			s = line.strip().split('\t')
			try:
				sv_1 = (s[0], int(round(int(s[1]) / res)) * res)
				sv_2 = (s[2], int(round(int(s[3]) / res)) * res)
				if sv_1 in row_labels and sv_2 in row_labels:
					if plot_collapsed_matrix:
						n1 = idx_dedup.index(row_labels[sv_1][0])
						n2 = idx_dedup.index(row_labels[sv_2][0])
						if n1 > n2:
							n1, n2 = n2, n1
						sv_x.append(n1)
						sv_y.append(n2)
					else:
						for n1 in row_labels[sv_1]:
							for n2 in row_labels[sv_2]:
								if n1 > n2:
									n1, n2 = n2, n1
								sv_x.append(n1)
								sv_y.append(n2)
			except:
				pass
		fp.close()
		ax.plot(sv_x, sv_y, 's', markeredgewidth = 1, ms = 6, markerfacecolor = "None", markeredgecolor = 'k')
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Plotted additional structural variations.")

	# Adjust ticks on the axis
	ax.set_xticks(mid_pos, minor = False)
	ax.set_xticklabels(xticklbl, rotation = 45, ha = 'left',fontsize = fontsize, minor = False)
	ax.set_yticks(yticklbl, minor = False)
	ax.set_yticklabels(yticklbl, fontsize = fontsize, minor = False)
	ax.set_ylabel("Bins (%dKb resolution)" %(res // 1000), fontsize = fontsize)
	if plot_collapsed_matrix:
		ax.set_xlabel(output_prefix.split('/')[-1] + "_collapsed_matrix", fontsize = fontsize)
	else:
		ax.set_xlabel(output_prefix.split('/')[-1] + "_expanded_matrix", fontsize = fontsize)

	plt.tight_layout()
	if plot_collapsed_matrix:
		plt.savefig(output_prefix + "_collapsed_matrix.pdf")
		plt.savefig(output_prefix + "_collapsed_matrix.png", dpi = 150)	
	else:
		plt.savefig(output_prefix + "_expanded_matrix.pdf")
		plt.savefig(output_prefix + "_expanded_matrix.png", dpi = 150)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the plot to pdf and png.")
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")
	print('Significant interactions plot is done. Significant interactions are visualized in %s and %s.' %(output_prefix + "_collapsed_matrix.pdf", output_prefix + "_collapsed_matrix.png" if plot_collapsed_matrix else output_prefix + "_expanded_matrix.pdf"))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Visualize significant interactions identified in ecDNA.")
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed (chr, start, end, orientation) format.", required = True)
	parser.add_argument("--resolution", help = "Bin size.", type = int, required = True)
	parser.add_argument("--matrix", help = "Input collapsed/expanded Hi-C matrix, in *.txt or *.npy format", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--annotation", help = "Annotation of bins in the input matrix.")
	parser.add_argument("--interactions", help = "Significant interactions to visualize.")
	parser.add_argument("--sv_list", help = "Optional, additional SVs not as part of the ecDNA structure.")
	parser.add_argument("--fontsize", help = "Tick fontsizes in the plot.", type = int, default = 24)
	parser.add_argument("--min_segment_ratio", help = "Do not show labels for segments smaller than min_segment_ratio * resolution.", type = int, default = 8)
	parser.add_argument("--plot_collapsed_matrix", help = "Visualize interactions and optional SVs on collapsed matrix.", action = 'store_true')
	parser.add_argument("--log_fn", help = "Name of log file.")
	
	args = parser.parse_args()
	plot_significant_interactions(**vars(args))