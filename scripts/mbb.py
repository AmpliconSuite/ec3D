import sys
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Define a new namedtuple for BoundingBox
from collections import namedtuple

BoundingBox = namedtuple('BoundingBox', ('volume',
					'length',
					'width',
					'height',
					'corner_points'
                                        ))


def rotate_points(points, angle, axis):
	R = np.eye(3)
	c, s = np.cos(angle), np.sin(angle)
	if axis == 0:  # x-axis
		R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
	elif axis == 1:  # y-axis
		R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
	elif axis == 2:  # z-axis
		R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
	return np.dot(points, R.T), R.T


def minimum_bounding_box(points, hull_edges, step = 100):
	min_volume = float('inf')
	min_box = None
	R1, R2, R3 = None, None, None
	
	for i in range(len(hull_edges)):
		for j in range(i + 1, len(hull_edges)):
			e1 = points[hull_edges[i][1]] - points[hull_edges[i][0]]
			e2 = points[hull_edges[j][1]] - points[hull_edges[j][0]]
			edge = np.cross(e1, e2)
			edge = edge / np.linalg.norm(edge)

			# Rotate points so that edge aligns with x-axis
			angle1 = np.arctan2(edge[2], edge[0])
			rotated_points, R1_ = rotate_points(points, angle1, 1)
			rotated_edge, R1__ = rotate_points(edge, angle1, 1)
			angle2 = np.arctan2(rotated_edge[1], rotated_edge[0])
			rotated_points_, R2_ = rotate_points(rotated_points, angle2, 2)
            
			# Rotate around x-axis to minimize height
			for theta in np.linspace(0, np.pi * 0.5, step):
				rotated_points_x, R3_ = rotate_points(rotated_points_, theta, 0)
				height = np.max(rotated_points_x[:, 2]) - np.min(rotated_points_x[:, 2])
				width = np.max(rotated_points_x[:, 1]) - np.min(rotated_points_x[:, 1])
				length = np.max(rotated_points_x[:, 0]) - np.min(rotated_points_x[:, 0])

				volume = length * height * width

				if volume < min_volume:
					min_volume = volume
					min_box = rotated_points_x
					R1, R2, R3 = R1_, R2_, R3_

	min_x, max_x = np.min(min_box[:, 0]), np.max(min_box[:, 0])
	min_y, max_y = np.min(min_box[:, 1]), np.max(min_box[:, 1])
	min_z, max_z = np.min(min_box[:, 2]), np.max(min_box[:, 2])
    
	corners = np.array([[min_x, min_y, min_z], [max_x, min_y, min_z],
			[max_x, max_y, min_z], [min_x, max_y, min_z],
			[min_x, min_y, max_z], [max_x, min_y, max_z],
			[max_x, max_y, max_z], [min_x, max_y, max_z]])
	corners = corners @ np.linalg.inv(R3) @ np.linalg.inv(R2) @ np.linalg.inv(R1)

	return BoundingBox(volume = min_volume, length = max_x - min_x, width = max_y - min_y, height = max_z - min_z, corner_points = corners)
	
    
def minimum_bounding_box_pca(points):
	# Apply PCA
	pca = PCA(n_components = 3)
	points_pca = pca.fit_transform(points)

	# Get the minimum and maximum values on each axis of the transformed points
	min_pca = np.min(points_pca, axis = 0)
	max_pca = np.max(points_pca, axis = 0)

	# Compute the lengths along each axis
	l = max_pca[0] - min_pca[0]
	w = max_pca[1] - min_pca[1]
	h = max_pca[2] - min_pca[2]

	# Compute the volume of the bounding box
	vol = l * w * h

	# Compute the corners of the bounding box in the space of the transformed points
	corners_pca = np.array([[min_pca[0], min_pca[1], min_pca[2]],
				[max_pca[0], min_pca[1], min_pca[2]],
				[max_pca[0], max_pca[1], min_pca[2]],
				[min_pca[0], max_pca[1], min_pca[2]],
				[min_pca[0], min_pca[1], max_pca[2]],
				[max_pca[0], min_pca[1], max_pca[2]],
				[max_pca[0], max_pca[1], max_pca[2]],
				[min_pca[0], max_pca[1], max_pca[2]]])

	# Transform the corners of the bounding box back to the original space
	corners = pca.inverse_transform(corners_pca)

	return BoundingBox(volume = vol, length = l, width = w, height = h, corner_points = corners)


def create_lines_from_corners(corners):
	lines = [[0, 1, 2, 3, 0, 4, 5, 6, 7, 4], [1, 5], [2, 6], [3, 7]]
	return [[corners[i] for i in line] for line in lines]


if __name__ == "__main__":
	# Read in structure
	points = np.loadtxt(sys.argv[1])
	print ("The input structure has %d points." %len(points))

	# Compute the convex hull
	hull = ConvexHull(points)
	hull_points = points[hull.vertices]
	hull_edges = set([])
	for face in hull.simplices:
		hull_edges.add((min(face[0], face[1]), max(face[0], face[1])))
		hull_edges.add((min(face[0], face[2]), max(face[0], face[2])))
		hull_edges.add((min(face[1], face[2]), max(face[1], face[2])))
	print ("Convex hull has %d points." %len(hull_points))
	print ("Convex polyhedron has %d edges." %len(hull_edges))

	# Compute the minimum bounding box
	if sys.argv[2] == 'pca':
		bbox = minimum_bounding_box_pca(points)
		print("Dimensions of the PCA bounding box:")
		print(f"Volume: {bbox.volume}")
		print(f"Length: {bbox.length}")
		print(f"Width: {bbox.width}")
		print(f"Height: {bbox.height}")
		print(f"Corners: {bbox.corner_points}")
	else:
		bbox = minimum_bounding_box(points, list(hull_edges))
		print("Dimensions of the optimal bounding box:")
		print(f"Volume: {bbox.volume}")
		print(f"Length: {bbox.length}")
		print(f"Width: {bbox.width}")
		print(f"Height: {bbox.height}")
		print(f"Corners: {bbox.corner_points}")

	# Create a scatter plot of the points
	scatter = go.Scatter3d(x = points[:,0], 
				y = points[:,1], 
				z = points[:,2], 
				mode = 'markers',
				name = 'points',
				marker = dict(size = 2)
	)

	original_bbox_lines = create_lines_from_corners(bbox.corner_points)
	original_bbox_line_plots = []
	for line in original_bbox_lines:
		x = [point[0] for point in line]
		y = [point[1] for point in line]
		z = [point[2] for point in line]
		if sys.argv[2] == 'pca':
			original_bbox_line_plots.append(go.Scatter3d(x = x, y = y, z = z,
									mode = 'lines',
									name = 'pca_bbox',
									line = dict(color = 'green')
									)
							)
		else:
			original_bbox_line_plots.append(go.Scatter3d(x = x, y = y, z = z,
									mode = 'lines',
									name = 'optimal_bbox',
									line = dict(color = 'green')
									)
							)
	fig = go.Figure(data = [scatter] + original_bbox_line_plots) 
	if sys.argv[2] == 'pca':
		fig.update_layout(title = f"MBB-PCA", scene = dict(xaxis_title = 'X', yaxis_title = 'Y', zaxis_title = 'Z'))
	else:
		fig.update_layout(title = f"MBB", scene = dict(xaxis_title = 'X', yaxis_title = 'Y', zaxis_title = 'Z'))

	# Create the figure and add the scatter plot and lines
	fig.write_html(sys.argv[3])

