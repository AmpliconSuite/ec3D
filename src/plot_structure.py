"""
Visualize 3D structures of an ecDNA
Author: Biswanath Chowdhury
"""
import sys
import os
import argparse
import time
import logging
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot

"""
Todo: add scatter plot option
	each gene has only one legend
"""


def plotstr_significant_interactions_dup_genes(pos, bin2gene, gene_colors, si, clusters, output_prefix, save_png = False):
	num_nodes = len(pos)
	fig = make_subplots(specs=[[{'type': 'scatter3d'}]])

	# Create a trace for nodes
	node_trace = go.Scatter3d(
		x = pos[:, 0],
		y = pos[:, 1],
		z = pos[:, 2],
		mode = 'markers + text',
		marker = dict(size = 3, color = 'blue'),
		text = [f'{i}' for i in range(num_nodes)],
		textfont = dict(size = 12),
		name = 'Nodes',
		visible = 'legendonly'
	)
	fig.add_trace(node_trace)
    
	all_edges_x = []
	all_edges_y = []
	all_edges_z = []
	for i in range(-1, num_nodes - 1):
		all_edges_x += [pos[i][0], pos[i + 1][0], None]
		all_edges_y += [pos[i][1], pos[i + 1][1], None]
		all_edges_z += [pos[i][2], pos[i + 1][2], None]
	base_edge_trace = go.Scatter3d(
		x = all_edges_x,
		y = all_edges_y,
		z = all_edges_z,
		mode = 'lines',
		line = dict(color = 'gray', width = 4.0),
		name = 'Edges',
		visible = True,  
		showlegend = False
	)
	fig.add_trace(base_edge_trace)
    
	edges_by_gene = dict()
	for bin in bin2gene.keys():
		try:
			edges_by_gene[bin2gene[bin]['gene']]['bins'].append(bin)
		except:
			edges_by_gene[bin2gene[bin]['gene']] = {'bins': [bin], 'strand': bin2gene[bin]['strand']}
	for gene in edges_by_gene.keys():
		edges_by_gene[gene]['bins'] = sorted(edges_by_gene[gene]['bins'])
		ranges_ = []
		range_ = [edges_by_gene[gene]['bins'][0], edges_by_gene[gene]['bins'][0]]
		for i in range(1, len(edges_by_gene[gene]['bins'])):
			if edges_by_gene[gene]['bins'][i] != range_[1] + 1:
				ranges_.append(range_)
				range_ = [edges_by_gene[gene]['bins'][i], edges_by_gene[gene]['bins'][i]]
			else:
				range_[1] += 1
		ranges_.append(range_)
		edges_by_gene[gene]['bins'] = ranges_
        
	for gene, ranges in edges_by_gene.items():
		for gene_range in ranges['bins']:
			edge_color = gene_colors.get(gene, 'gray')
			strand = ranges['strand']
			gene_name_with_strand = f"{gene} ({strand})"
			visible = 'legendonly' if gene.startswith(('LOC', 'LINC', 'MIR')) else True
			edge_x = []
			edge_y = []
			edge_z = []
			for bin_num in range(gene_range[0], gene_range[1]):
				edge_x += [pos[bin_num][0], pos[bin_num + 1][0], None]
				edge_y += [pos[bin_num][1], pos[bin_num + 1][1], None]
				edge_z += [pos[bin_num][2], pos[bin_num + 1][2], None]

			edge_trace = go.Scatter3d(
				x = edge_x,
				y = edge_y,
				z = edge_z,
				mode = 'lines',
				line = dict(width = 8.0, color = edge_color),
				name = gene_name_with_strand,  # Use gene name with strand in the legend
				showlegend = True,
				visible = visible
			)
			fig.add_trace(edge_trace)

			start_bin_num = gene_range[0]
			fig.add_trace(go.Scatter3d(
				x = [pos[start_bin_num][0]],
				y = [pos[start_bin_num][1]],
				z = [pos[start_bin_num][2]],
				mode = 'text',
				text = [gene],  # Use gene name without strand in the plot
				textfont = dict(color = edge_color, size = 10),
				showlegend = False,
				visible = visible
			))

	clusters_ = dict()
	for idx, row in clusters.iterrows():
		try:
			clusters_[row['cluster']].append(row['bin'])
		except:
			clusters_[row['cluster']] = [row['bin']]
	cluster_colors = [f'rgb({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)})'
			for cluster in clusters_.keys()]
	for cluster_id in sorted(clusters_.keys()):
		nodes = clusters_[cluster_id]
		cluster_edges_x, cluster_edges_y, cluster_edges_z = [], [], []
		for idx in range(len(si)):
			i, j = si['bin1'].values[idx], si['bin2'].values[idx]
			if i in nodes and j in nodes:
				cluster_edges_x.extend([pos[i][0], pos[j][0], None])
				cluster_edges_y.extend([pos[i][1], pos[j][1], None])
				cluster_edges_z.extend([pos[i][2], pos[j][2], None])
		# Add a trace for the cluster's significant interactions
		fig.add_trace(go.Scatter3d(
			x = cluster_edges_x,
			y = cluster_edges_y,
			z = cluster_edges_z,
			mode = 'lines',
			line = dict(color = cluster_colors[cluster_id], width = 1.5),  # or use a consistent color if preferred
			name = f'Cluster {cluster_id} Sig Interactions',
			visible = 'legendonly'
		))

	fig.write_html(output_prefix + "_ec3d.html")
	if save_png:
		fig.write_image(output_prefix + "_ec3d.png", dpi = 150)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Visualize the 3D structure of ecDNA.")
	parser.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format", required = True)
	parser.add_argument("--interactions", help = "Significant interactions to visualize.", required = True)
	parser.add_argument("--clusters", help = "Clusters of significant interactions to visualize.", required = True)
	parser.add_argument("--annotation", help = "Annotation of bins in the input matrix.", required = True)
	parser.add_argument("--ref", help = "One of {hg19, hg38, GRCh38, mm10}.", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--filter_interactions_by_percentile", help = ".")
	parser.add_argument("--log_fn", help = "Name of log file.")
	start_time = time.time()
	args = parser.parse_args()

	"""
	Set up logging
	"""
	log_fn = ""
	if not args.log_fn:
		log_fn = args.output_prefix + "_plot_structure.log"
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
	Read 3D coordinates
	"""
	X = np.array([])
	if args.structure.endswith(".txt"):
		X = np.loadtxt(args.structure)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded resolved 3D structure, in txt format.")
	elif args.structure.endswith(".npy"):
		X = np.load(args.structure)
		logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded resolved 3D structure, in npy format.")
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")

	"""
	Read significant interactions
	"""
	si = pd.read_csv(args.interactions)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded significant interactions.")
	logging.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + "Significant interactions: %s" %si)

	"""
	Read annotations and map bins to gene names
	"""
	oncogenes = dict()
	oncogene_fn = os.getcwd().replace("src", "data_repo") + "/"
	if args.ref == 'hg19' or args.ref == 'GRCh37':
		oncogene_fn += "AC_oncogene_set_GRCh37.gff"
	elif args.ref == 'hg38' or args.ref == 'GRCh38':
		oncogene_fn += "AC_oncogene_set_hg38.gff"
	elif args.ref == 'mm10' or args.ref == 'GRCm38':
		oncogene_fn += "AC_oncogene_set_mm10.gff"
	else:
		print("Reference must be one from {hg19, hg38, GRCh38, mm10}.")
		os.abort()
	fp = open(oncogene_fn, 'r')
	for line in fp:
		s = line.strip().split()
		if "chr" not in s[0]:
			s[0] = "chr" + s[0]
		try:
			oncogenes[s[0]].append([int(s[3]), int(s[4]), s[6], s[-1].split(';')[2][5:]])
		except:
			oncogenes[s[0]] = [[int(s[3]), int(s[4]), s[6], s[-1].split(';')[2][5:]]]
	fp.close()
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Parsed oncogene names and strand from %s." %(oncogene_fn))
	
	bin2gene = dict()
	unique_genes = set()
	fp = open(args.annotation, 'r')
	for line in fp:
		s = line.strip().split()
		for gene_intrvl in oncogenes[s[0]]:
			if int(s[1]) <= gene_intrvl[1] and gene_intrvl[0] <= int(s[2]):
				for i in range(3, len(s)):
					bin2gene[int(s[i])] = {'gene': gene_intrvl[3], 'strand': gene_intrvl[2]}
					unique_genes.add(gene_intrvl[3])	
	fp.close()
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Mapped the following bins to genes.")
	for bin_num in bin2gene:
		logging.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + \
				"Bin number: %d; Gene name: %s; Strand: %s" %(bin_num, bin2gene[bin_num]['gene'], bin2gene[bin_num]['strand']))

	# Assign colors to genes
	gene_colors = dict()
	for gene in unique_genes:
		gene_colors[gene] = f'rgb({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)})'
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Assigned one distinct color to each gene name.")
	
	# Read in and visualize clusters
	clusters = pd.read_csv(args.clusters)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded clusters of significant interactions.")
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Clusters: %s." %clusters)
	plotstr_significant_interactions_dup_genes(X, bin2gene, gene_colors, si, clusters, args.output_prefix)
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the structure plot to %s." %(args.output_prefix + "_ec3d.html"))
	logging.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")
	    
	