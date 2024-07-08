Simulating ecDNA 3D structures and Hi-C matrices. 

### Step 1 - Simulating ecDNA 3D structures
We defined a ```pinch``` as two subregions on an ecDNA that are genomically distant but spatially close to each other. Givne the definition, we designed three circular base structures with 1, 2, and 3 pinches, respectively. Then we added local folds into the base structures by random walks. As a result, we simulated ecDNA 3D structures with local and global folds as follows. 

#### Parameters
* Number of pinches: {1, 2, 3}
* Size of a full structure (#bins): N = {250, 500, 750}
* Size of a local fold (#bins): randomly from {16, 18, 20, 22}

![alt text](https://github.com/AmpliconSuite/ec3D/blob/main/images/simulated_structures.png)

### Step 2 - Generating Hi-C matrices
After simulating a 3D structure, we generated the corresponding Hi-C matrix by c_ij = Poisson_Sampling(lambda = beta*(d_ij)^alph), where d_ij is the Euclidean distance between bin i and j. Then we got Hi-C matrices as follows

#### Parameters
* Alpha: randomly from [-3, -0.75]
* Beta: randomly from [1, 10]

![alt text](https://github.com/AmpliconSuite/ec3D/blob/main/images/simulated_HiC.png)

### Step 3 - Simulating duplication on ecDNA
Given a simulated structure, we randomly chose n pairs of local folds, which could have the same or different conformations. Then we collapsed the Hi-C by adding the interaction frequencies of duplicated bins. As a result, we got the ecDNA structure with duplication and the corresponding collapsed Hi-C matrix. 

#### Parameters
* Number of pairs of duplicated local folds: n is a random interger from [1, ceiling(N/100)]
* Same or different duplicated conformations: True or False