"""
Visualize 3D structures of an ecDNA
Author: Biswanath Chowdhury
"""
import sys
import os
import argparse
import time
import warnings
import numpy as np
#from scipy.optimize import minimize
import hashlib
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#from plotly.offline import plot

try:
	from ec3d.util import create_logger, chr_idx
except:
	from util import create_logger, chr_idx

"""
def bending_energy(coords):
	energy = 0
	for i in range(1, len(coords) - 1):
		v1 = coords[i] - coords[i - 1]
		v2 = coords[i + 1] - coords[i]
		cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
		angle_change = np.arccos(np.clip(cos_theta, -1.0, 1.0))
		energy += angle_change ** 2
	return energy

def objective_function(coords, alpha):
	#Objective function that includes the bending energy.
	coords = coords.reshape(-1, 3)
	return alpha * bending_energy(coords)

def refine_coordinates(initial_coords, alpha=0.1):
	#Refines the 3D coordinates to minimize the bending energy.
	result = minimize(objective_function, initial_coords.ravel(), args=(alpha,), method='L-BFGS-B')
	refined_coords = result.x.reshape(len(initial_coords), 3)
	return refined_coords
"""

def add_arrow(fig, gene, start_point, end_point, visible, color = 'rgb(255,0,0)', size = 0.4):
	# Calculate the direction of the arrow
	direction = np.array(end_point) - np.array(start_point)
	length = np.linalg.norm(direction)
	direction = direction / length  # Normalize the direction vector

	# Define the arrow shaft
	arrow_shaft_end = np.array(end_point) - direction * size * length

	left = np.cross(direction, [0, 0, 1])
	if np.linalg.norm(left) == 0:  # direction is parallel to [0, 0, 1]
		left = np.cross(direction, [0, 1, 0])  # Use a different vector if parallel

	left = left / np.linalg.norm(left) * size * length   # Normalize and scale the left vector

	# Calculate the right vector as perpendicular to both direction and left
	right = np.cross(direction, left)
	right = right / np.linalg.norm(right) * size * length  

	# Add the lines representing the arrowhead sides
	for side in [left, right]:
		arrow_side_end = arrow_shaft_end + side
		fig.add_trace(go.Scatter3d(x = [arrow_side_end[0], end_point[0]],
					y = [arrow_side_end[1], end_point[1]],
					z = [arrow_side_end[2], end_point[2]],
					mode = 'lines',
					line = dict(color = color, width = 4),
					legendgroup = gene,
					showlegend = False,
					visible = visible
		))


def plotstr_significant_interactions_and_genes(pos, breakpoints, bins, bin2gene, redundant_genes, gene_colors, si, clusters, 
	output_prefix, noncyclic = False, show_background = False, save_png = False):
	num_nodes = len(pos)
	center = np.mean(pos, axis = 0)
	pos = pos - center
	up = pos[1] - pos[0]
	eye = np.cross(pos[1] - pos[0], pos[2] - pos[1])
	eye = eye / np.linalg.norm(eye) * 2.0
	
	camera = dict(eye = dict(x = eye[0], y = eye[1], z = eye[2]), # camera position
		up = dict(x = up[0], y = up[1], z = up[2]),
		center = dict(x = 0.0, y = 0.0, z = 0.0)
	)
	fig = make_subplots(specs=[[{'type': 'scatter3d'}]])

	# Create a trace for nodes
	node_trace = go.Scatter3d(
		x = pos[:, 0],
		y = pos[:, 1],
		z = pos[:, 2],
		mode = 'markers + text',
		marker = dict(size = 3, color = 'blue'),
		text = [f'{i}' if i % 5 == 0 else '' for i in range(num_nodes)],
		textfont = dict(size = 12),
		name = 'Nodes',
		visible = 'legendonly'
	)
	fig.add_trace(node_trace)
    
	concordant_edges_x, discordant_edges_x = [], []
	concordant_edges_y, discordant_edges_y = [], []
	concordant_edges_z, discordant_edges_z = [], []
	for i in range(-1, num_nodes - 1):
		if noncyclic and i == -1:
			continue
		if (i, i + 1) not in breakpoints:
			concordant_edges_x += [pos[i][0], pos[i + 1][0], None]
			concordant_edges_y += [pos[i][1], pos[i + 1][1], None]
			concordant_edges_z += [pos[i][2], pos[i + 1][2], None]
		else:
			discordant_edges_x += [pos[i][0], pos[i + 1][0], None]
			discordant_edges_y += [pos[i][1], pos[i + 1][1], None]
			discordant_edges_z += [pos[i][2], pos[i + 1][2], None]
	concordant_edge_trace = go.Scatter3d(
		x = concordant_edges_x,
		y = concordant_edges_y,
		z = concordant_edges_z,
		mode = 'lines',
		line = dict(color = 'gray', width = 6.0),
		name = 'Edges',
		visible = True,  
		showlegend = False
	)
	fig.add_trace(concordant_edge_trace)
	discordant_edge_trace = go.Scatter3d(
		x = discordant_edges_x,
		y = discordant_edges_y,
		z = discordant_edges_z,
		mode = 'lines',
		line = dict(color = 'black', width = 6.0, dash = 'dashdot'),
		name = 'Edges',
		visible = True,  
		showlegend = False
	)
	fig.add_trace(discordant_edge_trace)
	breakpoint_x = []
	breakpoint_y = []
	breakpoint_z = []
	for bp in breakpoints:
		breakpoint_x.append((pos[bp[0]][0] + pos[bp[1]][0]) / 2)
		breakpoint_y.append((pos[bp[0]][1] + pos[bp[1]][1]) / 2)
		breakpoint_z.append((pos[bp[0]][2] + pos[bp[1]][2]) / 2)
	breakpoint_trace = go.Scatter3d(
		x = breakpoint_x,
		y = breakpoint_y,
		z = breakpoint_z,
		mode = 'markers',
		marker = dict(size = 2.5, symbol = 'x', color = 'red'),
		name = 'breakpoints',
		showlegend = True
	)
	fig.add_trace(breakpoint_trace)
    
	edges_by_gene = dict()
	for bin in bin2gene.keys():
		for gene in bin2gene[bin].keys():
			try:
				edges_by_gene[gene]['bins'].append(bin)
			except:
				edges_by_gene[gene] = {'bins': [bin], 'strand': bin2gene[bin][gene]}
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
		if gene.startswith(('LOC', 'LINC', 'MIR')):
			continue
		edge_color = gene_colors.get(gene, 'gray')
		strand = ranges['strand']
		gene_name_with_strand = f"{gene} ({strand})"
		visible = 'legendonly' if (gene.startswith(('LOC', 'LINC', 'MIR')) or gene in redundant_genes) else True
		edge_x = []
		edge_y = []
		edge_z = []
		for gene_range in ranges['bins']:
			if gene_range[0] == gene_range[1]:
				continue
			for bin_num in range(gene_range[0], gene_range[1]):
				edge_x += [pos[bin_num][0], pos[bin_num + 1][0], None]
				edge_y += [pos[bin_num][1], pos[bin_num + 1][1], None]
				edge_z += [pos[bin_num][2], pos[bin_num + 1][2], None]
			start_bin_num = gene_range[0]
			fig.add_trace(go.Scatter3d(
				x = [pos[start_bin_num][0]],
				y = [pos[start_bin_num][1]],
				z = [pos[start_bin_num][2]],
				mode = 'text',
				text = [gene],  # Use gene name without strand in the plot
				textfont = dict(color = edge_color, size = 10),
				legendgroup = gene,
				showlegend = False,
				visible = visible
			))
			if (strand == '+' and bins[gene_range[1]][1] > bins[gene_range[1] - 1][1]) or (strand == '-' and bins[gene_range[1]][1] < bins[gene_range[1] - 1][1]):
				add_arrow(fig, gene, pos[gene_range[1] - 1], pos[gene_range[1]], visible, color = edge_color)
			else:
				add_arrow(fig, gene, pos[gene_range[0] + 1], pos[gene_range[0]], visible, color = edge_color)
		if not edge_x:
			continue
		edge_trace = go.Scatter3d(
				x = edge_x,
				y = edge_y,
				z = edge_z,
				mode = 'lines',
				line = dict(width = 12.0, color = edge_color),
				name = gene_name_with_strand,  # Use gene name with strand in the legend
				legendgroup = gene,
				showlegend = True,
				visible = visible
		)
		fig.add_trace(edge_trace)

	clusters_ = dict()
	for c in clusters:
		try:
			clusters_[int(c[1])].append(int(c[0]))
		except:
			clusters_[int(c[1])] = [int(c[0])]
	cluster_colors = [f'rgb({np.random.randint(0,255)}, {np.random.randint(0,255)}, {np.random.randint(0,255)})'
			for cluster in clusters_.keys()]
	for cluster_id in sorted(clusters_.keys()):
		nodes = clusters_[cluster_id]
		cluster_edges_x, cluster_edges_y, cluster_edges_z = [], [], []
		for si_item in si:
			i, j = int(si_item[0]), int(si_item[1])
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
	
	if not show_background:
		fig.update_layout(
			margin = dict(l = 10, r = 10, t = 20, b = 10),
			scene_dragmode = 'orbit',
			scene_camera = camera, 
			scene = dict(
				xaxis = dict(visible = False),
				yaxis = dict(visible = False),
				zaxis =dict(visible = False)
			)
		)
	else:
		fig.update_layout(
			margin = dict(l = 10, r = 10, t = 20, b = 10),
			scene_dragmode = 'orbit',
			scene_camera = camera
		)
	fig.write_html(output_prefix + "_ec3d.html")
	if save_png:
		fig.write_image(output_prefix + "_ec3d.png", scale = 2.5)

def plot_3D_structure(structure, output_prefix, interactions = None, clusters = None,
					  annotation = None, ref = 'hg38', download_gene = False, gene_fn = None, 
					  noncyclic = False, plot_axis = False, save_png = False, log_fn = None):
	"""
	Set up logging
	"""
	print('Plotting 3D structure ...')
	start_time = time.time()
	if not log_fn:
		log_fn = output_prefix + "_visualize_structure.log"
	logger = create_logger('plot_structure.py', log_fn)
	logger.info("Python version " + sys.version + "\n")
	function_param = f'plot_3D_structure(structure=\'{structure}\', output_prefix=\'{output_prefix}\', interactions=\'{interactions}\', ' + \
		f'clusters=\'{clusters}\', annotation=\'{annotation}\', ref=\'{ref}\', download_gene={download_gene}, gene_fn=\'{gene_fn}\', ' + \
		f'noncyclic={noncyclic}, plot_axis={plot_axis}, save_png={save_png}, log_fn=\'{log_fn}\')'
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + function_param)
	"""
	Read 3D coordinates
	"""
	X = np.array([])
	if structure.endswith(".txt"):
		X = np.loadtxt(structure)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded resolved 3D structure, in txt format.")
	elif structure.endswith(".npy"):
		X = np.load(structure)
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded resolved 3D structure, in npy format.")
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")

	"""
	Read significant interactions
	"""
	si = []
	if interactions:
		if not clusters:
			warnings.warn("Need clusters of significant interactions to visualize. Ignoring input significant interactions.")
		else:
			i = 0
			fp = open(interactions, 'r')
			for line in fp:
				if i > 0:
					si.append(line.strip().split('\t'))
				i += 1
			fp.close()
			logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded significant interactions.")
			logger.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + "Significant interactions: %s" %si)

	"""
	Read annotations and map bins to gene names
	"""
	oncogenes = dict()
	oncogene_fn = oncogene_fn = os.path.dirname(os.path.realpath(__file__))+"/"+"data_repo/"
	if download_gene and not gene_fn:
		wget_command = "wget -P " + oncogene_fn
		if ref == 'hg19' or ref == 'GRCh37':
			wget_command += " https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/hg19.ncbiRefSeq.gtf.gz"
		elif ref == 'hg38' or ref == 'GRCh38':
			wget_command += " https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.ncbiRefSeq.gtf.gz"
		elif ref == 'mm10' or ref == 'GRCm38':
			wget_command += " https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/genes/mm10.ncbiRefSeq.gtf.gz"
		else:
			print("Reference must be one from {hg19, hg38, GRCh38, mm10}.")
			os.abort()
		print (wget_command)
		os.system(wget_command)
		gunzip_command = ("gunzip " + oncogene_fn + "*.gz")
		os.system(gunzip_command)
		if ref == 'hg19' or ref == 'GRCh37':
			oncogene_fn += "hg19.ncbiRefSeq.gtf"
		elif ref == 'hg38' or ref == 'GRCh38':
			oncogene_fn += "hg38.ncbiRefSeq.gtf"
		elif ref == 'mm10' or ref == 'GRCm38':
			oncogene_fn += "mm10.ncbiRefSeq.gtf"
	elif gene_fn:
		if download_gene:
			warnings.warn("Will use the specified gene list, downloading disabled.")	
		oncogene_fn = gene_fn
	else: # Use the oncogene files provided in data_repo
		if ref == 'hg19' or ref == 'GRCh37':
			oncogene_fn += "AC_oncogene_set_GRCh37.gff"
		elif ref == 'hg38' or ref == 'GRCh38' or not annotation:
			oncogene_fn += "AC_oncogene_set_hg38.gff"
		elif ref == 'mm10' or ref == 'GRCm38':
			oncogene_fn += "AC_oncogene_set_mm10.gff"
		else:
			print("Reference must be one from {hg19, hg38, GRCh38, mm10}.")
			os.abort()
 
	fp = open(oncogene_fn, 'r')
	for line in fp:
		s = line.strip().split('\t')
		if "chr" not in s[0]:
			s[0] = "chr" + s[0]
		if s[0] not in chr_idx:
			continue
		if s[0] not in oncogenes:
			oncogenes[s[0]] = dict()
		gene_name = ""
		for token in s[-1].split(';'):
			if "Name" in token:
				gene_name = token[5:]
				break
			if "gene_name" in token:
				gene_name = token.strip()[11:-1]
				break
		if gene_name not in oncogenes[s[0]]:
			oncogenes[s[0]][gene_name] = [int(s[3]), int(s[4]), s[6]]
		else:
			oncogenes[s[0]][gene_name][0] = min(int(s[3]), oncogenes[s[0]][gene_name][0])
			oncogenes[s[0]][gene_name][1] = max(int(s[4]), oncogenes[s[0]][gene_name][1])
	fp.close()
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Parsed oncogene names and strand from %s." %(oncogene_fn))
	
	bins = dict()
	res = -1
	bin2gene = dict()
	unique_genes = set()
	redundant_genes = set()
	if annotation:
		fp = open(annotation, 'r')
		for line in fp:
			s = line.strip().split('\t')
			if res < 0:
				res = int(s[2]) - int(s[1])
			for i in range(3, len(s)):
				bins[int(s[i])] = [s[0], int(s[1])]
			for (gene, gene_intrvl) in oncogenes[s[0]].items():
				if int(s[1]) <= gene_intrvl[1] and gene_intrvl[0] <= int(s[2]):
					for i in range(3, len(s)):
						if int(s[i]) in bin2gene:
							for gene_ in bin2gene[int(s[i])].keys():
								if len(gene_) > len(gene):
									redundant_genes.add(gene_)
								elif len(gene_) == len(gene):
									redundant_genes.add(gene_)
									redundant_genes.add(gene)
							bin2gene[int(s[i])][gene] = gene_intrvl[2]
						else:
							bin2gene[int(s[i])] = {gene: gene_intrvl[2]}
						unique_genes.add(gene)
		fp.close()
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Mapped the following bins to genes.")
		for bin_num in bin2gene:
			for gene in bin2gene[bin_num].keys():
				logger.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + \
						"Bin number: %d; Gene name: %s; Strand: %s" %(bin_num, gene, bin2gene[bin_num][gene]))

	breakpoints = [(-1, 0)]
	if noncyclic:
		breakpoints = []
	if len(bins) > 0:
		for i in range(len(bins) - 1):
			if bins[i][0] != bins[i + 1][0]:
				breakpoints.append((i, i + 1))
			elif abs(bins[i + 1][1] - bins[i][1]) != res:
				breakpoints.append((i, i + 1))
			else:
				if 1 < i < len(bins) - 2 and bins[i][0] == bins[i - 1][0] and bins[i + 1][0] == bins[i + 2][0] and \
					abs(bins[i][1] - bins[i - 1][1]) == res and abs(bins[i + 2][1] - bins[i + 1][1]) == res and \
					bins[i][1] - bins[i - 1][1] != bins[i + 2][1] - bins[i + 1][1]: #foldbacks
					breakpoints.append((i, i + 1))
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Extracted %d breakpoint from annotation file." %(len(breakpoints)))
		logger.debug("#TIME " + '%.4f\t' %(time.time() - start_time) + "breakpoints %s" %(breakpoints))	
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Identified breakpoints from annotation file.")

	# Assign colors to genes
	gene_colors = dict()
	for gene in unique_genes:
		color = str(hashlib.sha1(gene.encode('utf-8')).hexdigest())[-6:]
		gene_colors[gene] = f'#{color}'
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Assigned one distinct color to each gene name.")
	
	# Read in and visualize clusters
	clusters = []
	if clusters:
		if not interactions:
			print("Significant interactions are required to visualize clusters.")
			os.abort()
		i = 0
		fp = open(clusters, 'r')
		for line in fp:
			if i > 0:
				clusters.append(line.strip().split('\t'))
			i += 1
		fp.close()
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Loaded clusters of significant interactions.")
		logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Clusters: %s." %clusters)

	noncyclic_ = False
	plot_axis_ = False
	save_png = False
	if noncyclic:
		noncyclic_ = True
	if plot_axis:
		plot_axis_ = True
	if save_png:
		save_png = True
	plotstr_significant_interactions_and_genes(X, breakpoints, bins, bin2gene, redundant_genes, gene_colors, si, clusters, output_prefix, 
		noncyclic = noncyclic_, show_background = plot_axis_, save_png = save_png)
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Saved the structure plot to %s." %(output_prefix + "_ec3d.html"))
	logger.info("#TIME " + '%.4f\t' %(time.time() - start_time) + "Total runtime.")
	print('3D structure visualization is done. The plot is saved to %s.' %(output_prefix + "_ec3d.html"))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Visualize the 3D structure of ecDNA.")
	parser.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of output files.", required = True)
	parser.add_argument("--interactions", help = "Significant interactions to visualize.")
	parser.add_argument("--clusters", help = "Clusters of significant interactions to visualize.")
	parser.add_argument("--annotation", help = "Annotation of bins in the input matrix.")
	parser.add_argument("--ref", help = "One of {hg19, hg38, GRCh38, mm10}.", choices=['hg19', 'hg38', 'GRCh38', 'mm10'], default='hg38')
	parser.add_argument("--download_gene", help = "Download gene list from UCSC Genome Browser.", action = 'store_true')
	parser.add_argument("--gene_fn", help = "Parse user provided gene list, in *.gff of *.gtf format.")
	parser.add_argument("--noncyclic", help = "Noncyclic structure, will not connect the first and last nodes in 3D plot.", action = 'store_true')
	parser.add_argument("--plot_axis", help = "Plot the three axes and grid.", action = 'store_true')
	parser.add_argument("--save_png", help = "Save the structure plot additionally into a *.png file.", action = 'store_true')
	parser.add_argument("--log_fn", help = "Name of log file.")
	
	args = parser.parse_args()
	plot_3D_structure(**vars(args))