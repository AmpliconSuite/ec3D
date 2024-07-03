import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import random

def calculate_distance_matrix(coords):
    coords = np.array(coords)
    distance_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
    return distance_matrix

def distance_to_interaction(distance_matrix, a=-3, b=1, noise = False):
    """
    Convert pairwise distances to interaction counts using a power-law decay.

    Parameters:
    - distance_matrix: Pairwise distances between all points.
    - a, b: Parameters of the power-law decay.

    Returns:
    - hic_matrix: Simulated Hi-C interaction matrix.
    """
    # Mask zero distances to avoid divide-by-zero errors
    masked_distance_matrix = np.where(distance_matrix == 0, distance_matrix+0.01, distance_matrix)
    hic_matrix = b * np.power(masked_distance_matrix, a)
    if noise == True:
        for i in range(len(hic_matrix)):
            for j in range(i, len(hic_matrix[0])):
                hic_matrix[i, j] = np.random.poisson(hic_matrix[i, j])
                hic_matrix[j, i] = hic_matrix[i, j]
    return hic_matrix

# One pinch k = 1
def one_pinch(num_bins, pinch_strength = 0.9):
    def x(t):
        return np.cos(t)

    def y(t):
        if np.sin(t) > 0: # symmetric about x-axis
            return np.sin(t) - pinch_strength * np.sin(t)**4
        else:
            return np.sin(t) + pinch_strength * np.sin(t)**4

    def z(t):
        return np.cos(t)**2

    # Define the derivative functions (dx/dt, dy/dt, dz/dt)
    def dx_dt(t):
        return -np.sin(t)

    def dy_dt(t):
        if np.sin(t) > 0:
            return np.cos(t) - 4 * pinch_strength * np.sin(t)**3 * np.cos(t)
        else:
            return np.cos(t) + 4 * pinch_strength * np.sin(t)**3 * np.cos(t)

    def dz_dt(t):
        return -2 * np.cos(t) * np.sin(t)

    # Define the integrand function
    def integrand(t):
        return np.sqrt(dx_dt(t)**2 + dy_dt(t)**2 + dz_dt(t)**2)

    # Calculate the arc length
    N = 10000
    arc_length = np.zeros(N)
    t_values_fine = np.linspace(0, 2 * np.pi, N)
    for i in range(1, N):
        arc_length[i], _ = quad(integrand, t_values_fine[i-1], t_values_fine[i])
    arc_length = np.cumsum(arc_length)
    
    # Interpolate to find t values for equal arc length
    interp_func = interp1d(arc_length, t_values_fine)
    arc_length_even = np.linspace(0, arc_length[-1], num_bins+1)
    t_values_even = interp_func(arc_length_even)

    # The average step length
    step_arc_length = arc_length[-1] / num_bins
    # Calculate the coordinates for evenly spaced points
    sampled_points = []
    for t in t_values_even[:-1]:
        sampled_points.append([x(t), y(t), z(t)])
    return step_arc_length, sampled_points

# Two pinches k = 2
def two_pinches(num_bins, strength1 = 0.9, strength2 = 0.9):
    def x(t):
        if np.cos(t) > 0: # symmetric about y-axis
            return np.cos(t) - strength1 * np.cos(t)**4
        else:
            return np.cos(t) + strength1 * np.cos(t)**4

    def y(t):
        if np.sin(t) > 0: # symmetric about x-axis
            return np.sin(t) - strength2 * np.sin(t)**4
        else:
            return np.sin(t) + strength2 * np.sin(t)**4

    def z(t):
        return np.sin(t)**2

    # Define the derivative functions (dx/dt, dy/dt, dz/dt)
    def dx_dt(t):
        if np.cos(t) > 0:
            return -np.sin(t) + 4 * strength1 * np.cos(t)**3 * np.sin(t)
        else:
            return -np.sin(t) - 4 * strength1 * np.cos(t)**3 * np.sin(t)

    def dy_dt(t):
        if np.sin(t) > 0:
            return np.cos(t) - 4 * strength2 * np.sin(t)**3 * np.cos(t)
        else:
            return np.cos(t) + 4 * strength2 * np.sin(t)**3 * np.cos(t)

    def dz_dt(t):
        return 2 * np.sin(t) * np.cos(t)

    # Define the integrand function
    def integrand(t):
        return np.sqrt(dx_dt(t)**2 + dy_dt(t)**2 + dz_dt(t)**2)

    # Calculate the cumulative arc length function
    def cumulative_arc_length(start, end):
        result, _ = quad(integrand, start, end)
        return result

    # Calculate the arc length
    N = 10000
    arc_length = np.zeros(N)
    t_values_fine = np.linspace(0, 2 * np.pi, N)
    for i in range(1, N):
        arc_length[i], _ = quad(integrand, t_values_fine[i-1], t_values_fine[i])
    arc_length = np.cumsum(arc_length)
    
    # Interpolate to find t values for equal arc length
    interp_func = interp1d(arc_length, t_values_fine)
    arc_length_even = np.linspace(0, arc_length[-1], num_bins+1)
    t_values_even = interp_func(arc_length_even)

    # The average step length
    step_arc_length = arc_length[-1] / num_bins
    # Calculate the coordinates for evenly spaced points
    sampled_points = []
    for t in t_values_even[:-1]:
        sampled_points.append([x(t), y(t), z(t)])
    return step_arc_length, sampled_points

# Three pinches k = 3
def three_pinches(num_bins, strength = 0.9):
    def x(t):
        return np.cos(t) * (1 + strength * np.cos(3 * t))/4

    def y(t):
        return np.sin(t) * (1 + strength * np.cos(3 * t))/4

    def z(t):
        return np.sin(3 * t)/4

    # Define the derivative functions (dx/dt, dy/dt, dz/dt)
    def dx_dt(t):
        return (-np.sin(t) * (1 + strength * np.cos(3 * t)) + np.cos(t) * (-3 * strength * np.sin(3 * t)))/4

    def dy_dt(t):
        return (np.cos(t) * (1 + strength * np.cos(3 * t))/2 + np.sin(t) * (-3 * strength * np.sin(3 * t)))/4

    def dz_dt(t):
        return 3 * np.cos(3 * t)/4

    # Define the integrand function
    def integrand(t):
        return np.sqrt(dx_dt(t)**2 + dy_dt(t)**2 + dz_dt(t)**2)
        
    # Calculate the arc length
    N = 10000
    arc_length = np.zeros(N)
    t_values_fine = np.linspace(0, 2 * np.pi, N)
    for i in range(1, N):
        arc_length[i], _ = quad(integrand, t_values_fine[i-1], t_values_fine[i])
    arc_length = np.cumsum(arc_length)
    
    # Interpolate to find t values for equal arc length
    interp_func = interp1d(arc_length, t_values_fine)
    arc_length_even = np.linspace(0, arc_length[-1], num_bins+1)
    t_values_even = interp_func(arc_length_even)

    # The average step length
    step_arc_length = arc_length[-1] / num_bins
    # Calculate the coordinates for evenly spaced points
    sampled_points = []
    for t in t_values_even[:-1]:
        sampled_points.append([x(t), y(t), z(t)])
    return step_arc_length, sampled_points

def generate_local_fold(step_length, start_coor, end_coor, num_points, seed = -1, ref_delta = np.array([])):
    r = step_length # step length
    if seed != -1:
        random.seed(seed)
    t = random.uniform(0, 2*np.pi)
    p = random.uniform(0, 2*np.pi)
    end_coor = end_coor - np.array([r*np.sin(t)*np.cos(p), r*np.sin(t)*np.sin(p), r*np.cos(t)])
    
    coordinates = [start_coor]
    delta = (np.array(end_coor) - np.array(start_coor))*2/num_points

    step_vectors, comp_vectors = [], []
    if ref_delta.size == 3:
        rotation_axis = np.cross(ref_delta, delta)
        rotation_axis = normalize(rotation_axis)
        angle_of_rotation = angle_between_vectors(ref_delta, delta)
    for _ in range(num_points//2):
        t = random.uniform(0, 2*np.pi)
        p = random.uniform(0, 2*np.pi)
        vector = np.array([r*np.sin(t)*np.cos(p), r*np.sin(t)*np.sin(p), r*np.cos(t)])
        if ref_delta.size == 3:
            vector = rotate_vector(vector, rotation_axis, angle_of_rotation)
        step_vectors.append(vector)
        comp_vectors.append(delta - vector)
    step_vectors = step_vectors + comp_vectors
    # Generate coordinates
    for d in step_vectors:
        next_coordinate = coordinates[-1] + d
        coordinates.append(next_coordinate)
    # print(len(coordinates), coordinates[-1], end_coor)
    return coordinates, delta

def save_structure(coordinates, filename):
    with open(filename, 'w') as txt_file:
        for [x_coord, y_coord, z_coord] in coordinates:
            txt_file.write(f"{x_coord}\t{y_coord}\t{z_coord}\n")

def simulate_duplications(total_num_bins, local_folds):
    local_folds_by_size = {}
    for size in [2 * i for i in range(8, 12)]:
        local_folds_by_size[size] = []
    for i, local_fold in enumerate(local_folds):
        size = local_fold[1]
        if size != 4:
            local_folds_by_size[size].append(i)
    # randomly choose local fold pairs of the same size
    num_duplicated_pairs = random.randint(1, round(total_num_bins*0.5/50))
    keys = list(local_folds_by_size.keys())
    duplicated_pairs = []

    while len(duplicated_pairs) < num_duplicated_pairs:
        # Randomly choose a key
        key = random.choice(keys)
        local_folds_list = local_folds_by_size[key]
        # Ensure there are enough elements to form 1 pair
        if len(local_folds_list) < 2:
            # Remove the key if not enough elements to form a pair
            keys.remove(key)
            continue
        # Randomly select a pair from the list
        random.shuffle(local_folds_list)
        pair = (local_folds_list.pop(), local_folds_list.pop())
        if pair[0] > pair[1]:
            pair = (pair[1], pair[0])
        duplicated_pairs.append(pair)
        # Update the list in the dictionary after removing used elements
        local_folds_by_size[key] = local_folds_list
        # If the list is now too small to form a pair, remove the key from future consideration
        if len(local_folds_list) < 2:
            keys.remove(key)
    # Generate the duplication list
    bin_dict = {}
    for i in range(total_num_bins):
        bin_dict[i] = []
    for (pos1, pos2) in duplicated_pairs:
        i, region_size = 0, local_folds[pos1][1]
        pos1, pos2 = local_folds[pos1][0], local_folds[pos2][0]
        while i < region_size:
            bin_dict[pos1+i].append(pos2+i)
            del bin_dict[pos2+i]
            i += 1
    duplications = [[bin]+copies for bin,copies in bin_dict.items()]
    return duplications, duplicated_pairs

# vector rotation
def normalize(v):
    return v / np.linalg.norm(v)

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    return np.arccos(cos_theta)

def rotate_vector(v, axis, angle):
    """
    Rotate vector v by angle around the given axis using Rodrigues' rotation formula.

    Parameters:
    v (np.array): Vector to be rotated
    axis (np.array): Axis of rotation
    angle (float): Angle in radians

    Returns:
    np.array: Rotated vector
    """
    axis = normalize(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    dot_product = np.dot(axis, v)
    cross_product = np.cross(axis, v)

    rotated_v = cos_theta * v + sin_theta * cross_product + (1 - cos_theta) * dot_product * axis
    return rotated_v