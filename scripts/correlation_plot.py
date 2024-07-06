"""
Visualize relationship between distances
Author: Biswanath Chowdhury
"""
import argparse
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy.stats import spearmanr, pearsonr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['pdf.fonttype'] = 42
   
import plotly.graph_objs as go


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = ".")
	parser.add_argument("--matrix", help = "Input expanded Hi-C matrix, in *.txt or *.npy format", required = True)
	parser.add_argument("--structure", help = "The 3D structure of ecDNA, in *.txt or *.npy format", required = True)
	parser.add_argument("--output_prefix", help = "Prefix of the output file.", required = True)
	parser.add_argument("--include_genomic_distance", help = "Plot the correlations between spatial distance, genomic distance and interaction frequencies.", action = "store_true")
	parser.add_argument("--fontsize", help = "Tick fontsizes in the plot.", type = int, default = 24)
	args = parser.parse_args()
	
	"""
	Read expanded matrix
	"""
	D = np.array([])
	if args.matrix.endswith(".txt"):
		D = np.loadtxt(args.matrix)
	elif args.matrix.endswith(".npy"):
		D = np.load(args.matrix)
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")
	
	"""
	Read 3D coordinates
	"""
	X = np.array([])
	if args.structure.endswith(".txt"):
		X = np.loadtxt(args.structure)
	elif args.structure.endswith(".npy"):
		X = np.load(args.structure)
	else:
		raise OSError("Input matrix must be in *.txt or *.npy format.")

	assert X.shape[0] == D.shape[0]
	N = D.shape[0]
	dis = euclidean_distances(X) 
	distances = []
	interaction_freqs = []
	genomic_distances = []
	for i in range(N):
		for j in range(i + 1, N):
			if D[i][j] > 0:
				distances.append(dis[i][j])
				interaction_freqs.append(D[i][j])  
				genomic_distances.append(min(abs(i - j), N - abs(i - j)))

	if args.include_genomic_distance:
		spearman_corr_interaction_spatial, _ = spearmanr(np.log10(interaction_freqs), np.log10(distances))
		pearson_corr_interaction_spatial, _ = pearsonr(np.log10(interaction_freqs), np.log10(distances))
		spearman_corr_interaction_genomic, _ = spearmanr(np.log10(interaction_freqs), np.log10(genomic_distances))
		pearson_corr_interaction_genomic, _ = pearsonr(np.log10(interaction_freqs), np.log10(genomic_distances))
    
		# Create text for correlations
		corr_text = f'Spearman correlation (Int_freq-Spatial_dist): {spearman_corr_interaction_spatial:.2f}<br> \
				Pearson correlation (Int_freq-Spatial_dist): {pearson_corr_interaction_spatial:.2f}<br> \
				Spearman correlation (Int_freq-Genomic_dist): {spearman_corr_interaction_genomic:.2f}<br> \
				Pearson correlation (Int_freq-Genomic_dist): {pearson_corr_interaction_genomic:.2f}'

		# Create the 3D scatter plot
		fig = go.Figure(data = [go.Scatter3d(
				x = interaction_freqs,
				y = distances,
				z = genomic_distances,
				mode = 'markers',
				marker = dict(size = 2,
					color = (genomic_distances), # set color to genomic distance
					colorscale = 'Viridis',   # choose a colorscale
					opacity = 0.8,
					colorbar = dict(title = "genomic distance", len = 0.5, y = 0.8)
				)
		)])
		fig.update_layout(scene = dict(xaxis = dict(type = 'log', title = 'Hi-C interaction frequency'),
						yaxis = dict(type = 'log', title = 'Spatial distance'),
						zaxis = dict(type = 'log', title = 'Genomic distance')),
						width = 700,
						margin = dict(l = 50, # left margin
							r = 50, # right margin
							b = 10, # bottom margin
							t = 10, # top margin
						),
						annotations = [dict(showarrow = False,
								x = 0, 
								y = 1.05, 
								xref = 'paper',
								yref = 'paper',
								text = corr_text,
								xanchor = "left",
								xshift = 10,
								opacity = 0.7)]
		)
		fig.write_html(args.output_prefix + '_interactions_vs_distances.html')
	else:
		spearman_corr, _ = spearmanr(np.log(interaction_freqs), np.log(distances))
		pearson_corr, _ = pearsonr(np.log(interaction_freqs), np.log(distances))

		# Plot the results
		plt.figure(figsize = (12, 8))
    
		# Create a color map for genomic distance
		cmap = plt.get_cmap('coolwarm')
		norm = plt.matplotlib.colors.Normalize(vmin = min(genomic_distances), vmax = max(genomic_distances))

		# Scatter plot with color representing genomic distance
		plt.scatter(interaction_freqs, distances, c = genomic_distances, cmap = cmap, norm = norm, alpha = 0.5)

		plt.xscale('log')
		plt.yscale('log')
		plt.xticks(fontsize = args.fontsize)
		plt.yticks(fontsize = args.fontsize)
		plt.xlabel('Hi-C interaction frequency', fontsize = args.fontsize)
		plt.ylabel('Spatial distance', fontsize = args.fontsize)
		plt.title(args.output_prefix + ' interactions vs spatial distance', fontsize = args.fontsize)
    
		# Add a colorbar
		cbar = plt.colorbar()
		cbar.set_label('Genomic Distance', fontsize = args.fontsize)
		cbar.ax.tick_params(labelsize = args.fontsize) 
    
		# Add text 
		plt.text(0.02, min(distances) + 0.08, f'Spearman correlation: {spearman_corr:.2f}', transform = plt.gca().transAxes, fontsize = args.fontsize)
		plt.text(0.02, min(distances) + 0.02, f'Pearson correlation: {pearson_corr:.2f}', transform = plt.gca().transAxes, fontsize = args.fontsize)

		plt.tight_layout()
		plt.savefig(args.output_prefix + '_interactions_vs_distance.png', dpi = 150)
		plt.savefig(args.output_prefix + '_interactions_vs_distance.pdf')






