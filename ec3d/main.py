import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import hic2cool
import numpy as np
from threadpoolctl import threadpool_limits

try:
	from ec3d.extract_matrix import extract_matrix
	from ec3d.spatial_structure import reconstruct_3D_structure
	from ec3d.expand_matrix import expand_matrix
	from ec3d.significant_interactions import identify_significant_interactions
	from ec3d.plot_interactions import plot_significant_interactions
	from ec3d.plot_structure import plot_3D_structure
except:
	from extract_matrix import extract_matrix
	from spatial_structure import reconstruct_3D_structure
	from expand_matrix import expand_matrix
	from significant_interactions import identify_significant_interactions
	from plot_interactions import plot_significant_interactions
	from plot_structure import plot_3D_structure

def main():
	parser = argparse.ArgumentParser(description = "Compute the 3D coordinates from Hi-C.")
	parser.add_argument("--hic", help = "Input whole genome Hi-C map, in *.cool or *.hic format.", required = True)
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed (chr, start, end, orientation) format.", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files", required = True)
	parser.add_argument("--resolution", help = "Bin siz.", type = int, required = True)
	parser.add_argument("--ref", help = "One of {hg19, hg38, GRCh38, mm10}.", choices=['hg19', 'hg38', 'GRCh38', 'mm10'], default='hg38')
	parser.add_argument("--num_threads", help = "Maximal number of threads (default 8) that can be used by ec3D", type = int, default=8)
	parser.add_argument("--save_npy", help = "Save matrices to *.npy format.", action = "store_true")
	args = parser.parse_args()
	
	# Extract Hi-C submatrices of amplified regions
	hic_fn = args.hic
	if not (hic_fn.endswith('.cool') or hic_fn.endswith('.hic') or '.mcool' in hic_fn):
		raise ValueError("The input Hi-C file must be in .cool or .hic format.")
	elif hic_fn.endswith('.hic'):
		new_hic_fn = hic_fn.split('/')[-1][:-4] + '.cool'
		hic2cool.hic2cool_convert(hic_fn, new_hic_fn, args.resolution)
		hic_fn = new_hic_fn
	extract_matrix(hic_fn, args.ecdna_cycle, args.resolution, args.output_prefix, save_npy=args.save_npy)

	# Reconstruct 3D structure
	matrix_fn = args.output_prefix + ('_collapsed_matrix.npy' if args.save_npy else '_collapsed_matrix.txt')
	annotation_fn = args.output_prefix + '_annotations.bed'
	with threadpool_limits(limits=args.num_threads, user_api='blas'):
		reconstruct_3D_structure(matrix_fn, annotation_fn, args.output_prefix, save_npy=args.save_npy)

	# Generate expanded matrix
	dup_flag = False
	fp = open(args.output_prefix + "_annotations.bed", 'r')
	for line in fp:
		s = line.strip().split()
		if len(s) > 4:
			dup_flag = True
	fp.close()
	if dup_flag:
		fp = open(args.output_prefix + "_hyperparameters.txt", 'r')
		params = dict()
		for line in fp:
			s = line.strip().split('\t')
			params[s[0]] = s[1]
		fp.close()
		# Expand Hi-C matrix for duplication
		raw_matrix_fn = args.output_prefix + ('_raw_collapsed_matrix.npy' if args.save_npy else '_raw_collapsed_matrix.txt')
		structure_fn = args.output_prefix + ('_coordinates.npy' if args.save_npy else '_coordinates.txt')
		expand_matrix(raw_matrix_fn, annotation_fn, structure_fn, float(params['alpha']), float(params['beta']), args.output_prefix, save_npy=args.save_npy)
	else:
		if args.save_npy:
			collapsed_matrix = np.load(args.output_prefix + '_collapsed_matrix.npy')
			np.save(args.output_prefix + '_expanded_matrix.npy', collapsed_matrix)
		else:
			collapsed_matrix = np.loadtxt(args.output_prefix + '_collapsed_matrix.txt')
			np.savetxt(args.output_prefix + '_expanded_matrix.txt', collapsed_matrix)

	# Identify significant interactions
	expanded_matrix = args.output_prefix + ('_expanded_matrix.npy' if args.save_npy else '_expanded_matrix.txt')
	identify_significant_interactions(args.output_prefix, expanded_matrix)

	# Plot significant interactions
	interactions_fn = args.output_prefix + "_significant_interactions.tsv"
	plot_significant_interactions(args.ecdna_cycle, int(args.resolution), expanded_matrix, args.output_prefix, args.save_npy, interactions=interactions_fn)
	
	# Plot 3D structure
	structure_fn = args.output_prefix + ('_coordinates.npy' if args.save_npy else '_coordinates.txt')
	clusters_fn = args.output_prefix + "_clustered_bins.tsv"
	plot_3D_structure(structure_fn, args.output_prefix, interactions=interactions_fn, clusters=clusters_fn, annotation=annotation_fn, ref=args.ref)

	print ("Finished.")

if __name__ == "__main__":
	main()