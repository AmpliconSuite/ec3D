"""
Visualize significant interactions within ecDNA
Author: Biswanath Chowdhury
"""
import os
import sys
import logging
import argparse
import time
import numpy as np

from util import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = [10, 10]
rcParams['pdf.fonttype'] = 42

"""
Todo:
(1) add --rotate-to-min
(2) complete sv and interaction visualization on collapsed matrix
(3) complete the output to logging
(4) remove diagonal SVs
"""
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Visualize significant interactions identified in ecDNA.")
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed (chr, start, end, orientation) format.", required = True)
	parser.add_argument("--resolution", help = "Bin size.", type = int, required = True)
	parser.add_argument("--matrix", help = "Input collapsed/expanded Hi-C matrix, in *.txt or *.npy format", required = True)
	parser.add_argument("--annotation", help = "Annotation of bins in the input matrix.")
	parser.add_argument("--interactions", help = "Significant interactions to visualize.")
	parser.add_argument("--sv_list", help = "Optional, additional SVs not as part of the ecDNA structure.")
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--fontsize", help = "Tick fontsizes in the plot.", type = int, default = 24)
	parser.add_argument("--min_segment_ratio", help = "Do not show labels for segments smaller than min_segment_ratio * resolution.", type = int, default = 8)
	parser.add_argument("--plot_collapsed_matrix", help = "Visualize interactions and optional SVs on collapsed matrix.", action = 'store_true')
	parser.add_argument("--log_fn", help = "Name of log file.")
	start_time = time.time()
	args = parser.parse_args()

	"""
	Set up logging
	"""
	log_fn = ""
	if not args.log_fn:
		log_fn = args.output_prefix + "_visualize_interactions.log"
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
	Read in ecDNA cycle
	"""
	res = args.resolution
	intrvls = read_ecDNA_cycle(args.ecdna_cycle, res)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "ecDNA involves %d amplified intervals with resolution %d." %(len(intrvls), res))
	
	"""
	Optional: Read in annotations
	"""
	bins = []
	row_labels = dict() # Map an interval of size RES to the index of all its copies
	N = 0
	if args.plot_collapsed_matrix or args.sv_list:
		if args.annotation == 'None':
			print("Annotation file is required.")
			os.abort()
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
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded ecDNA matrix annotations.")
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Expanded ecDNA matrix size: %d." %N)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Collapsed ecDNA matrix size: %d." %len(row_labels))
	if args.plot_collapsed_matrix:
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
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Reset amplified intervals according to collapsed matrix.")

	# Calculate the midpoints for x-tick and y-tick labels
	start_pos = []
	mid_pos = []
	xticklbl = []
	yticklbl = []
	num_bins = 0
	for intrvl in intrvls:
		intrvl_len = (intrvl[2] - intrvl[1]) // res
		if intrvl_len >= args.min_segment_ratio:
			start_label = f"{intrvl[1]/1e6:.1f}".rstrip('0').rstrip('.') if intrvl[1] % 1e6 != 0 else f"{int(intrvl[1]/1e6)}"
			end_label = f"{intrvl[2]/1e6:.1f}".rstrip('0').rstrip('.') if intrvl[2] % 1e6 != 0 else f"{int(intrvl[2]/1e6)}"
			xticklbl.append(f"{intrvl[0]}:{start_label}-{end_label}Mb")
			yticklbl.append(num_bins)
		else:
			xticklbl.append('')  # Empty string for segments too small to label
		mid_pos.append(num_bins + intrvl_len * 0.5)
		num_bins += intrvl_len
		start_pos.append(num_bins)
		logging.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + "\tAmplified interval %s." %intrvl)

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
	if args.plot_collapsed_matrix:
		print (data.shape)
		if data.shape[0] != len(row_labels) or data.shape[1] != len(row_labels):
			raise OSError("Input matrix must be in size %d * %d." %(len(row_labels), len(row_labels)))

	"""
	Visualize Hi-C and interactions
	"""
	fontsize = args.fontsize  
	fig, ax = plt.subplots()
	log_data = np.ma.log10(np.ma.masked_where(data == 0.0, data))

	# Display heatmap with bin numbers
	im = ax.matshow(log_data, cmap = 'YlOrRd')

	# Create a color bar with consistent font size
	cbar = fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
	cbar.ax.tick_params(labelsize = fontsize)  # Set font size for color bar ticks

	# Draw lines at the bin edges
	for pos in start_pos[: -1]:
		ax.axhline(y = pos, color = 'black', linestyle = '--', linewidth = 0.4, clip_on = False)
		ax.axvline(x = pos, color = 'black', linestyle = '--', linewidth = 0.4, clip_on = False)

	# Plot significant interactions
	if args.interactions:
		si_x, si_y = [], []
		fp = open(args.interactions, 'r')
		for line in fp:
			s = line.strip().split(',')
			try:
				si_x.append(int(s[1]))
				si_y.append(int(s[0]))
			except:
				pass
		fp.close()
		ax.plot(si_x, si_y, 'o', markeredgewidth = 1, ms = 5, markerfacecolor = "None", markeredgecolor = 'b')

	# Optional: plot additional SVs
	if args.sv_list:
				
		sv_x, sv_y = [], []
		fp = open(args.sv_list, 'r')
		for line in fp:
			s = line.strip().split('\t')
			try:
				sv_1 = (s[0], int(round(int(s[1]) / res)) * res)
				sv_2 = (s[2], int(round(int(s[3]) / res)) * res)
				if sv_1 in row_labels and sv_2 in row_labels:
					for n1 in row_labels[sv_1]:
						for n2 in row_labels[sv_2]:
							if n1 <= n2:
								sv_x.append(n1)
								sv_y.append(n2)
							else:
								sv_x.append(n2)
								sv_y.append(n1)
			except:
				pass
		fp.close()
		ax.plot(sv_x, sv_y, 's', markeredgewidth = 1, ms = 6, markerfacecolor = "None", markeredgecolor = 'k')

	# Adjust ticks on the axis
	ax.set_xticks(mid_pos, minor = False)
	ax.set_xticklabels(xticklbl, rotation = 45, ha = 'left',fontsize = fontsize, minor = False)
	ax.set_yticks(yticklbl[1:], minor = False)
	ax.set_yticklabels(yticklbl[1:], fontsize = fontsize, minor = False)
	ax.set_ylabel("Bins (%dKb resolution)" %(res // 1000), fontsize = fontsize)
	ax.set_xlabel("")

	plt.tight_layout()
	if args.plot_collapsed_matrix:
		plt.savefig(args.output_prefix + "_collapsed_matrix.pdf")
		plt.savefig(args.output_prefix + "_collapsed_matrix.png", dpi = 150)
	else:
		plt.savefig(args.output_prefix + "_expanded_matrix.pdf")
		plt.savefig(args.output_prefix + "_expanded_matrix.png", dpi = 150)

