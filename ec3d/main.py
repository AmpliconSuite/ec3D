import warnings
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)

import sys
import os
import argparse
import hic2cool
import numpy as np

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


def run_ec3d_all_steps(hic_fn, ecdna_cycle, res, output_prefix, num_threads = None, ref = 'hg38', save_npy = False):
	# Extract Hi-C submatrices of amplified regions
	converted_hic_fn = ""
	if not (hic_fn.endswith(".cool") or hic_fn.endswith(".hic") or ".mcool" in hic_fn):
		raise ValueError("The input Hi-C file must be in .cool or .hic format.")
	elif hic_fn.endswith(".hic"):
		converted_hic_fn = hic_fn.split('/')[-1][:-4] + ".cool"
		hic2cool.hic2cool_convert(hic_fn, converted_hic_fn, res)
		extract_matrix(converted_hic_fn, ecdna_cycle, res, output_prefix, save_npy)
	else:
		extract_matrix(hic_fn, ecdna_cycle, res, output_prefix, save_npy)
	if hic_fn.endswith(".hic"):
		try:
			os.remove(converted_hic_fn)
		except:
			pass

	# Reconstruct 3D structure
	matrix_fn = output_prefix + ("_collapsed_matrix.npy" if save_npy else "_collapsed_matrix.txt")
	annotation_fn = output_prefix + "_annotations.bed"
	if num_threads:
		from threadpoolctl import threadpool_limits
		with threadpool_limits(limits = num_threads, user_api = 'blas'):
			alpha, beta, X = reconstruct_3D_structure(matrix_fn, annotation_fn, output_prefix, save_npy = save_npy)
	else:
		alpha, beta, X = reconstruct_3D_structure(matrix_fn, annotation_fn, output_prefix, save_npy = save_npy)
	
	# Generate expanded matrix
	dup_flag = False
	fp = open(output_prefix + "_annotations.bed", 'r')
	for line in fp:
		s = line.strip().split()
		if len(s) > 4:
			dup_flag = True
	fp.close()
	if dup_flag:
		raw_matrix_fn = output_prefix + ("_raw_collapsed_matrix.npy" if save_npy else "_raw_collapsed_matrix.txt")
		expand_matrix(raw_matrix_fn, annotation_fn, X, alpha, beta, output_prefix, save_npy = save_npy)
	else:
		if save_npy:
			collapsed_matrix = np.load(output_prefix + '_collapsed_matrix.npy')
			np.save(output_prefix + '_expanded_matrix.npy', collapsed_matrix)
		else:
			collapsed_matrix = np.loadtxt(output_prefix + '_collapsed_matrix.txt')
			np.savetxt(output_prefix + '_expanded_matrix.txt', collapsed_matrix)

	# Identify significant interactions
	expanded_matrix = output_prefix + ('_expanded_matrix.npy' if save_npy else '_expanded_matrix.txt')
	identify_significant_interactions(output_prefix, expanded_matrix)

	# Plot significant interactions
	interactions_fn = output_prefix + "_significant_interactions.tsv"
	plot_significant_interactions(ecdna_cycle, res, expanded_matrix, output_prefix, save_npy, interactions = interactions_fn)
	
	# Plot 3D structure
	structure_fn = output_prefix + ('_coordinates.npy' if save_npy else '_coordinates.txt')
	clusters_fn = output_prefix + "_clustered_bins.tsv"
	plot_3D_structure(structure_fn, output_prefix, interactions = interactions_fn, clusters = clusters_fn, 
			annotation = annotation_fn, ref = ref)

	print ("Finished.")


def process_arguments():
	parser = argparse.ArgumentParser(description = "ec3D: 3D structure analysis tookit of ecDNA.")

	subparser_ = any(arg in ["extract_matrix", "reconstruct", "expand_matrix", "significant_interactions",
					"plot_interactions", "plot_structure"] for arg in sys.argv)

	parser.add_argument("--hic", "--cool", dest = "input", help = "Input whole genome Hi-C map, in *.cool or *.hic format.", required = not subparser_)
	parser.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed format.", required = not subparser_)
	parser.add_argument("--output_prefix", help = "Prefix of output files", required = not subparser_)
	parser.add_argument("--resolution", help = "Bin size.", type = int, required = not subparser_)
	parser.add_argument("--ref", help = "One of {hg19, hg38, GRCh38, mm10}.", choices = ['hg19', 'hg38', 'GRCh38', 'mm10'], default = 'hg38')
	parser.add_argument("--num_threads", help = "Maximum number of threads that can be used by ec3D.", type = int)
	parser.add_argument("--save_npy", help = "Save matrices to *.npy format.", action = "store_true")

	subparsers = parser.add_subparsers(dest = "mode", help = "Run individual steps of ec3D.")

	parser_extract_matrix = subparsers.add_parser("extract_matrix", description = "Extract Hi-C matrix correspond to ecDNA intervals.")
	parser_extract_matrix.add_argument("--hic", "--cool", dest = "input", help = "Input whole genome Hi-C map, in *.cool or *.hic format.", required = True)
	parser_extract_matrix.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed format.", required = True)
	parser_extract_matrix.add_argument("--resolution", help = "Bin size.", type = int, required = True)
	parser_extract_matrix.add_argument("--output_prefix", help = "Prefix of the output files.", required = True)
	parser_extract_matrix.add_argument("--log_fn", help = "Name of log file.")
	parser_extract_matrix.add_argument("--save_npy", help = "Save matrices to *.npy format", action = "store_true")
	
	parser_reconstruct = subparsers.add_parser("reconstruct", description = "Compute the 3D coordinates from Hi-C.")
	parser_reconstruct.add_argument("--matrix", help = "Input collapsed Hi-C matrix, in *.txt or *.npy format.", required = True)
	parser_reconstruct.add_argument("--raw_matrix", help = "Raw, collapsed Hi-C matrix, in *.txt or *.npy format.")
	parser_reconstruct.add_argument("--annotation", help = "Annotation of bins in the input matrix.", required = True)
	parser_reconstruct.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser_reconstruct.add_argument("--log_fn", help = "Name of log file.")
	parser_reconstruct.add_argument("--reg", help = "Regularizer weight.", type = float, default = 0.05)
	parser_reconstruct.add_argument("--init_alpha", help = "An initial guess of alpha, for initialization and MDS.", type = float, default = -3.0)
	parser_reconstruct.add_argument("--num_repeats", help = "Number of repetitions with random initial structures.", type = int, default = 5)
	parser_reconstruct.add_argument("--num_threads", help = "Maximum number of threads that can be used by ec3D.", type = int)
	parser_reconstruct.add_argument("--save_repeats", help = "Save the reconstructed structure in each repeat.", action = "store_true")
	parser_reconstruct.add_argument("--max_rounds", help = "Maximum number of rounds for Poisson model.", type = int, default = 1000)
	parser_reconstruct.add_argument("--gt_structure", help = "Input the ground truth structure, in *.txt or *.npy format, for calculating RMSD and PCC.")
	parser_reconstruct.add_argument("--save_npy", help = "Save matrices to *.npy format", action = "store_true")

	parser_expand_matrix = subparsers.add_parser("expand_matrix", description = "Compute the expanded Hi-C from collapsed Hi-C.")
	parser_expand_matrix.add_argument("--raw_matrix", help = "Raw, collapsed Hi-C matrix, in *.txt or *.npy format.", required = True)
	parser_expand_matrix.add_argument("--annotation", help = "Annotation of bins in the input matrix.", required = True)
	parser_expand_matrix.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format.", required = True)
	parser_expand_matrix.add_argument("--alpha", help = "The value of alpha from 3D structure computation.", required = True, type = float)
	parser_expand_matrix.add_argument("--beta", help = "The value of beta from 3D structure computation.", required = True, type = float)
	parser_expand_matrix.add_argument("--strategy", help = "One option from redist/poisson/hybrid.", default = "redist", choices = ['redist', 'hybrid', 'poisson'])
	parser_expand_matrix.add_argument("--log_fn", help = "Name of log file.")

	parser_si = subparsers.add_parser("significant_interactions", description = "Identify significant interactions from ecDNA.")
	parser_si.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser_si.add_argument("--matrix", help = "Input expanded Hi-C matrix, in *.txt or *.npy format.")
	parser_si.add_argument("--pval_cutoff", help = "P-value cutoff as significant interactions.", type = float, default = 0.05)
	parser_si.add_argument("--model", help = "Statistical model used to computet the P-values.", default = "global_nb",
				choices = ['distance_ratio', 'local', 'global_poisson', 'global_nb'])
	parser_si.add_argument("--padding", help = "Pad expanded Hi-C matrix with certain values, for HiCCUPS interaction calling.", 
				default = "average", choices = ['zero', 'average', 'cyclic'])
	parser_si.add_argument("--genomic_distance_model", help = "Model of genomic distance between two bins.", default = "circular",
				choices = ['circular', 'linear', 'reference'])
	parser_si.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format.")
	parser_si.add_argument("--annotation", help = "Annotation of bins in the input matrix.")
	parser_si.add_argument("--max_pooling", help = "Only keep significant interactions larger than their neighbors.", action = 'store_true')
	parser_si.add_argument("--exclude", help = "Exclude significant interactions at given indices.", type = int, nargs = '+')
	parser_si.add_argument("--log_fn", help = "Name of log file.")

	parser_plot_si = subparsers.add_parser("plot_interactions", description = "Visualize significant interactions identified in ecDNA.")
	parser_plot_si.add_argument("--ecdna_cycle", help = "Input ecDNA intervals, in *.bed format.", required = True)
	parser_plot_si.add_argument("--resolution", help = "Bin size.", type = int, required = True)
	parser_plot_si.add_argument("--matrix", help = "Input collapsed/expanded Hi-C matrix, in *.txt or *.npy format", required = True)
	parser_plot_si.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser_plot_si.add_argument("--annotation", help = "Annotation of bins in the input matrix.")
	parser_plot_si.add_argument("--interactions", help = "Significant interactions to visualize.")
	parser_plot_si.add_argument("--sv_list", help = "Optional, additional SVs not as part of the ecDNA structure.")
	parser_plot_si.add_argument("--fontsize", help = "Tick fontsizes in the plot.", type = int, default = 24)
	parser_plot_si.add_argument("--min_segment_ratio", help = "Do not show labels for segments smaller than min_segment_ratio * resolution.", type = int, default = 8)
	parser_plot_si.add_argument("--plot_collapsed_matrix", help = "Visualize interactions and optional SVs on collapsed matrix.", action = 'store_true')
	parser_plot_si.add_argument("--log_fn", help = "Name of log file.")

	parser_plot_structure = subparsers.add_parser("plot_structure", description = "Visualize the 3D structure of ecDNA.")
	parser_plot_structure.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format", required = True)
	parser_plot_structure.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser_plot_structure.add_argument("--interactions", help = "Significant interactions to visualize.")
	parser_plot_structure.add_argument("--clusters", help = "Clusters of significant interactions to visualize.")
	parser_plot_structure.add_argument("--annotation", help = "Annotation of bins in the input matrix.")
	parser_plot_structure.add_argument("--ref", help = "One of {hg19, hg38, GRCh38, mm10}.", choices=['hg19', 'hg38', 'GRCh38', 'mm10'], default='hg38')
	parser_plot_structure.add_argument("--download_gene", help = "Download gene list from UCSC Genome Browser.", action = 'store_true')
	parser_plot_structure.add_argument("--gene_fn", help = "Parse user provided gene list, in *.gff of *.gtf format.")
	parser_plot_structure.add_argument("--noncyclic", help = "Noncyclic structure, will not connect the first and last nodes in 3D plot.", action = 'store_true')
	parser_plot_structure.add_argument("--plot_axis", help = "Plot the three axes and grid.", action = 'store_true')
	parser_plot_structure.add_argument("--save_png", help = "Save the structure plot additionally into a *.png file.", action = 'store_true')
	parser_plot_structure.add_argument("--log_fn", help = "Name of log file.")

	args = parser.parse_args()

	if args.mode == "extract_matrix":
		print ("s")
		hic_fn = args.input
		converted_hic_fn = ""
		if not (hic_fn.endswith('.cool') or hic_fn.endswith('.hic') or '.mcool' in hic_fn):
			raise ValueError("The input Hi-C file must be in .cool or .hic format.")
		elif hic_fn.endswith('.hic'):
			converted_hic_fn = hic_fn.split('/')[-1][:-4] + '.cool'
			hic2cool.hic2cool_convert(hic_fn, converted_hic_fn, args.resolution)
			extract_matrix(converted_hic_fn, args.ecdna_cycle, args.resolution, args.output_prefix, save_npy = args.save_npy)
		else:
			extract_matrix(hic_fn, args.ecdna_cycle, args.resolution, args.output_prefix, save_npy = args.save_npy)
		if hic_fn.endswith('.hic'):
			try:
				os.remove(converted_hic_fn)
			except:
				pass
	elif args.mode == "reconstruct":
		dup_flag = False
		fp = open(args.annotation, 'r')
		for line in fp:
			s = line.strip().split()
			if len(s) > 4:
				dup_flag = True
		fp.close()
		if args.num_threads:
			from threadpoolctl import threadpool_limits
			with threadpool_limits(limits = args.num_threads, user_api = 'blas'):
				alpha, beta, X = reconstruct_3D_structure(args.matrix, args.annotation, args.output_prefix, log_fn = args.log_fn, 
								reg = args.reg, init_alpha = args.init_alpha, num_repeats = args.num_repeats, 
								save_repeats = args.save_repeats, max_rounds = args.max_rounds, 
								gt_structure = args.gt_structure, save_npy = args.save_npy)
		else:
			alpha, beta, X = reconstruct_3D_structure(args.matrix, args.annotation, args.output_prefix, log_fn = args.log_fn, 
							reg = args.reg, init_alpha = args.init_alpha, num_repeats = args.num_repeats, 
							save_repeats = args.save_repeats, max_rounds = args.max_rounds, 
							gt_structure = args.gt_structure, save_npy = args.save_npy)
		if dup_flag:
			if args.raw_matrix:
				expand_matrix(args.raw_matrix, args.annotation, X, alpha, beta, args.output_prefix, save_npy = args.save_npy)
			else:
				print ("Raw Hi-C matrix not found, skipped matrix expansion.")
		else:
			collapsed_matrix = np.array([])
			if args.matrix.endswith(".txt"):
				collapsed_matrix = np.loadtxt(args.matrix)
			elif args.matrix.endswith(".npy"):
				collapsed_matrix = np.load(args.matrix)
			if args.save_npy:
				np.save(args.output_prefix + "_expanded_matrix.npy", collapsed_matrix)
			else:
				np.savetxt(args.output_prefix + "_expanded_matrix.txt", collapsed_matrix)
	elif args.mode == "expand_matrix":
		expand_matrix(args.raw_matrix, args.annotation, args.structure, args.alpha, args.beta, 
				args.output_prefix, strategy = args.strategy, log_fn = args.log_fn, save_npy = args.save_npy)
	elif args.mode == "significant_interactions":
		identify_significant_interactions(args.output_prefix, matrix = args.matrix, pval_cutoff = args.pval_cutoff, 
				model = args.model, padding = args.padding, genomic_distance_model = args.genomic_distance_model, 
				significant_interactions = None, structure = args.structure, annotation = args.annotation, 
				max_pooling = args.max_pooling, exclude = args.exclude, log_fn = args.log_fn)
	elif args.mode == "plot_interactions":
		plot_significant_interactions(args.ecdna_cycle, args.resolution, args.matrix, args.output_prefix,
				annotation = args.annotation, interactions = args.interactions, sv_list = args.sv_list,
				fontsize = args.fontsize, min_segment_ratio = args.min_segment_ratio, 
				plot_collapsed_matrix = args.plot_collapsed_matrix, log_fn = args.log_fn)
	elif args.mode == "plot_structure":
		plot_3D_structure(args.structure, args.output_prefix, interactions = args.interactions, clusters = args.clusters,
				annotation = args.annotation, ref = args.ref, download_gene = args.download_gene, gene_fn = args.gene_fn, 
				noncyclic = args.noncyclic, plot_axis = args.plot_axis, save_png = args.save_png, log_fn = args.log_fn)
	else:
		run_ec3d_all_steps(args.input, args.ecdna_cycle, args.resolution, args.output_prefix, 
				num_threads = args.num_threads, ref = args.ref, save_npy = args.save_npy)


if __name__ == "__main__":
	process_arguments()

