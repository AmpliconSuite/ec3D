import random
import numpy as np
import csv
from util import *

# simulate the ecDNA 3D structure and Hi-C matrix
def random_para(k):
    # total number of bins {250, 500, 750}
    total_num_bins = 250 * k
    # alpha (-3, -0.75) and beta (1, 10)
    alpha = round(random.choice(np.linspace(-3, -0.75, num=226)), 2)
    beta = round(random.choice(np.linspace(1, 10, num=901)), 2)
    # local folds [(starting bin, number of bins) ...]
    pos = random.randint(0, 5)
    local_folds = []
    while pos < total_num_bins:
        num_lf_bins = random.choice([2 * i for i in range(8, 12)])
        lf_diff = random.choice([4, 4, 4, 22 + (k-1)*4]) # distance between adjacent local folds
        if pos + num_lf_bins + lf_diff >= total_num_bins:
            break
        local_folds.append((pos, num_lf_bins))
        pos += num_lf_bins
        if lf_diff != 4:
            next_pos = pos + lf_diff
            pos += 4
            while pos + 4 < next_pos:
                local_folds.append((pos, 4))
                pos += 8
            pos = next_pos
        else:
            pos += lf_diff
    while pos + 4 < total_num_bins-1:
        local_folds.append((pos, 4))
        pos += 8
    return total_num_bins, alpha, beta, local_folds

if __name__ == "__main__":
    parameter_file = csv.writer(open('simulation_parameters.csv', 'w'))
    parameter_file.writerow(['case_id', '#pinches', '#bins', 'alpha', 'beta', '#duplicated_bins', 'same_duplication'])
    case_id = 1
    for num_pinches in [1, 2, 3]:
        for k in [1, 2, 3]: # total number of bins - 250, 500, 750
            for rep in range(10):
                random.seed(case_id)
                print(f'simulating structure {case_id}')
                # generate and save random parameters
                total_num_bins, alpha, beta, local_folds = random_para(k)
                
                # generate the base structure
                num_base_bins = total_num_bins - sum([entry[1] for entry in local_folds])
                pinch_strength = random.choice([0.01 * i for i in range(90, 100)])

                if num_pinches == 1:
                    step_length, coordinates = one_pinch(num_base_bins, pinch_strength)
                elif num_pinches == 2:
                    step_length, coordinates = two_pinches(num_base_bins, pinch_strength, pinch_strength)
                else:
                    step_length, coordinates = three_pinches(num_base_bins, pinch_strength)

                # Simulate duplication
                duplications, duplicated_pairs = simulate_duplications(total_num_bins, local_folds)
                duplicated_ref = {}
                print(local_folds)
                # create seeds for local folds
                selected_seeds = [1, 3, 4, 7, 8, 11, 17, 20, 29, 32, 34, 36, 47, 50, 53, 54, 58, 59, 62, 65, 70, 74, 77, 82, 86, 97]
                # random.seed(case_id)
                seeds = random.choices(selected_seeds, k = len(local_folds))
                if rep < 5: # same duplicated substructure
                    for (i, j) in duplicated_pairs:
                        seed = random.choice(selected_seeds)
                        seeds[i] = seed
                        seeds[j] = seed
                        duplicated_ref[j] = i
                        print(f'same [{local_folds[i][0]}:{local_folds[i][0]+local_folds[i][1]}], [{local_folds[j][0]}:{local_folds[j][0]+local_folds[j][1]}]')
                else: # different duplicated substructures
                    for (i, j) in duplicated_pairs:
                        seeds[i],  seeds[j] = random.sample(selected_seeds, k = 2)
                        print(f'different [{local_folds[i][0]}:{local_folds[i][0]+local_folds[i][1]}], [{local_folds[j][0]}:{local_folds[j][0]+local_folds[j][1]}]')
                # Add local folds
                deltas = [np.array([]) for _ in range(len(local_folds))]
                for idx, (pos, num_lf_bins) in enumerate(local_folds):
                    seed = seeds[idx]
                    ref_delta = np.array([])
                    if idx in duplicated_ref.keys():
                        ref_delta = deltas[duplicated_ref[idx]]
                    local_fold, deltas[idx] = generate_local_fold(step_length, coordinates[pos], coordinates[pos+1], num_lf_bins, seed, ref_delta)
                    coordinates  = coordinates[:pos] + local_fold + coordinates[pos+1:]
                
                # Save the 3D structure
                save_structure(coordinates, f'simulation{case_id}_coordinates.txt')
                
                # Generate and save the Hi-C matrix
                distance_matrix = calculate_distance_matrix(coordinates)
                hic_matrix = distance_to_interaction(distance_matrix, alpha, beta, True)
                np.savetxt(f'simulation{case_id}_expanded_matrix.txt', hic_matrix, delimiter='\t', fmt='%.18e')

                # Collapse the Hi-C matrix by summation        
                removed_idx = []
                collapsed_matrix = np.copy(hic_matrix)
                for bin_idx in duplications:
                    i = bin_idx[0]
                    for j in bin_idx[1:]:
                        removed_idx.append(j)
                        # Add row j to row i
                        collapsed_matrix[i] += hic_matrix[j]
                        # Add column j to column i
                        collapsed_matrix[:, i] += hic_matrix[:, j]
                        # Add the self-interaction
                        collapsed_matrix[i, i] += hic_matrix[j, j]
                # Remove duplication from the result matrix
                collapsed_matrix = np.delete(collapsed_matrix, removed_idx, axis=0)
                collapsed_matrix = np.delete(collapsed_matrix, removed_idx, axis=1)
                np.savetxt(f'simulation{case_id}_collapsed_matrix.txt', collapsed_matrix, delimiter='\t', fmt='%.18e')
                # Generate the annotation file
                with open(f'simulation{case_id}_annotations.bed', 'w') as file:
                    rows = []
                    genome_pos, bin_size = 128460000, 10000
                    for bin_idx in duplications:
                        rows.append(['chr8', genome_pos, genome_pos+bin_size, *bin_idx])
                        genome_pos += bin_size
                    writer = csv.writer(file, delimiter='\t')
                    writer.writerows(rows)
                # Save the simulation parameters
                num_duplicated_bins = 0
                for (i, j) in duplicated_pairs:
                    num_duplicated_bins += local_folds[i][1] + local_folds[j][1]
                same_duplication = len(duplicated_ref) > 0
                parameter_file.writerow([case_id, num_pinches, total_num_bins, alpha, beta, num_duplicated_bins, same_duplication])
                case_id += 1
                