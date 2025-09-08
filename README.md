# ec3D: reconstructing the 3D structure of extrachromosomal DNA molecules
[![GitHub release](https://img.shields.io/github/release/AmpliconSuite/ec3D.svg)](https://github.com/AmpliconSuite/ec3D/releases)
[![PyPI version](https://badge.fury.io/py/ec3D.svg)](https://badge.fury.io/py/ec3D)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Ec3D is a computational method for reconstructing 3D structures of ecDNA and analyzing significant interactions from high-throughput chromatin capture (Hi-C) data.

## Dependencies 
- Cooler (https://cooler.readthedocs.io/)
- ICED (https://members.cbio.mines-paristech.fr/~nvaroquaux/iced/)
- SciPy (https://scipy.org/)
- scikit-learn (https://scikit-learn.org)
- autograd (https://github.com/HIPS/autograd)
- NetworkX (https://networkx.org/)
- python-louvain (https://python-louvain.readthedocs.io/en/latest/)
- Plotly (https://plotly.com/python/) and Kaleido (https://pypi.org/project/kaleido/)

## Installation
Ec3D can be installed and run on most modern Unix-like operating systems (e.g. Ubuntu 18.04+, CentOS 7+, macOS). It requires python>=3.8 and the above dependencies. There are a few options for installing ec3D as follows. 
### Option A: Installing from PyPI
```
pip install ec3D
```
### Option B: Local installation with venv
1. Pull the source code
    ```
    git clone git@github.com:AmpliconSuite/ec3D.git
    cd /path/to/ec3D
    ```
2. Create a virtual environment
    ```
    python3 -m venv ec3D_venv
    source ec3D_venv/bin/activate
    ```
3. Install dependencies locally
    ```
    pip install .
    ```
### Option C: Local installation with Conda
1. Pull the source code
    ```
    git clone git@github.com:AmpliconSuite/ec3D.git
    cd /path/to/ec3D
    ```
2. Create a Conda environment with dependencies
    ```
    conda env create -f environment.yml
    conda activate ec3D_env
    ```
## Running
### Batch mode
The easiest way to run ec3D is running in **batch** mode, performing all following steps (i.e., [Preprocessing Hi-C](#step-1---preprocessing-hi-c), [Reconstructing the 3D structure of ecDNA](#step-2---reconstructing-the-3d-structure-of-ecdna), [Identifying significant interactions](#step-3---identifying-significant-interactions), and [Visualization](#step-4---visualization)), with default parameters. For custom parameter adjustments, it is best to run through individual steps.
```
ec3D --hic <FILE> --ecdna_cycle <FILE> --resolution <INT> --output_prefix <STRING>
```
or
```
python3 /path/to/ec3D/ec3d/main.py --hic <FILE> --ecdna_cycle <FILE> --resolution <INT> --output_prefix <STRING>
```
The only required input for ec3D is a Hi-C matrix, in ```*.[m]cool``` or ```*.hic``` format, and an ecDNA cycle, in browser extensible data (```*.bed```) format. Of course, you will need to specify the resolution to work with, and a prefix of the desired output. After a successful reconstruction, ```ec3D``` will write all default output files in the following steps into the path specified in ```--output_prefix```.
- ```--hic <FILE>```, Hi-C matrix, in ```*.[m]cool``` or ```*.hic``` format. Usually [cooler](https://cooler.readthedocs.io/) will organize multiple cool files with different resolutions in ```*.mcool``` format, and you will need to add a suffix ```::/resolutions/<RESOLUTION>``` to specify the resolution you want to work with. If ```*.hic``` is provided, it will be converted to ```.cool``` by ec3D. 
- ```--ecdna_cycle <FILE>```, ecDNA intervals, in ```*.bed``` (with chr, start, end stored in the first three columns and strand orientation stored in the 6-th column) format, see below for an example.
- ```--resolution <INT>```, resolution, which should match the resolution (i.e., bin size) of the input ```*.cool``` file. Each ```ec3D``` run must only work with a single fixed resolution.
- ```--output_prefix <STRING>```, prefix of the output matrix files and annotation file ```*_annotations.bed```. Note that if these file is desired to be written to a different directory, then a path/directory should also be included as a prefix. 
- ```--num_threads <INT>```, maximal number of threads (optional, default 8) that can be used by ec3D. Note that the multi-core utilization arises from parallelized BLAS operations for reconstructing 3D structures, and ec3D takes advantage of multiple CPU cores for intensive computation. Adding a control limit on available number of threads can save computational resources but may increase running time. 

### Sample run of ec3D
As a test sample, you can download the processed Hi-C dataset for D458 (a pediatric medulloblastoma cell line) from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7697651), and convert it to ```*.mcool``` format (ec3D can also convert it to ```*.cool``` internally with a specific resolution):
```
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7697nnn/GSM7697651/suppl/GSM7697651%5FD458.allValidPairs.hic
hic2cool convert GSM7697651_D458.allValidPairs.hic D458.mcool
```
Then, use [this ecDNA cycle](https://github.com/AmpliconSuite/ec3D/blob/main/sample/D458_ecDNA.bed) below to kick off a ec3D batch run.
```
mkdir sample_output
ec3D --cool D458.mcool::/resolutions/5000 --ecdna_cycle D458_ecDNA.bed --output_prefix ./sample_output/D458 --resolution 5000
```

### Step 1 - Preprocessing Hi-C
Given the whole genome Hi-C (in ```*.cool``` format) and the ecDNA cycle (in ```*.bed``` format), output the Hi-C matrix corresponding to ecDNA to work on in the following steps, and the annotation of each bin at a given resolution. 
#### Usage
```python3 extract_matrix.py [Required arguments] [Optional arguments]```
#### Required arguments
- ```--cool <FILE>```, Hi-C matrix, in ```*.cool``` format.
- ```--ecdna_cycle <FILE>```, ecDNA intervals, in ```*.bed``` format.
- ```--resolution <INT>```, Resolution, which should match the resolution of the input ```*.cool``` file.
- ```--output_prefix <STRING>```, Prefix of the output matrix files and annotation file ```*_annotations.bed```.
#### Optional arguments
- ```--save_npy```, Save output matrices in ```*.npy``` format. Note that by default, the ecDNA matrices are saved in ```*.txt``` format for easier readability, even though they are less compact.
#### [Example ecDNA cycle file](https://github.com/AmpliconSuite/ec3D/blob/main/sample/D458_ecDNA.bed) (from [D458](https://www.ncbi.nlm.nih.gov/sra/SRX21566415)):
```
#chr	start	end	cycle_id	weight	orientation	iscyclic
chr8	127293093	127948583	1	1.000000	+	True
chr8	127872074	128441212	1	1.000000	-	True
chr8	128505994	128527415	1	1.000000	+	True
chr8	127293093	127455873	1	1.000000	-	True
chr8	129033869	129085009	1	1.000000	-	True
chr8	127850424	127870604	1	1.000000	+	True
chr14	56593089	56794203	1	1.000000	+	True
chr14	56797826	56986857	1	1.000000	-	True
chr8	127957689	128012988	1	1.000000	-	True
chr8	128443842	128452940	1	1.000000	+	True
chr8	128458992	129085009	1	1.000000	+	True
```
- The provided ecDNA structure in a cycle bed file may include duplicated segments in its records, e.g., ```chr8 127293093 127948583``` and ```chr8 127872074 128441212```.  We refer to _collapsed_ matrix as the Hi-C matrix where each duplicated segment occurs only one time; and _expanded_ matrix as the Hi-C matrix representing the structure of ecDNA where all duplicated segments occur as many times as they are duplicated. Ec3D will automatically process cycle files containing duplicated segments and reconstruct the underlying ecDNA structures, regardless of whether the input from ```--ecdna_cycle``` contains duplication or not. 
- Please note that ec3D accepts the UCSC (e.g., "chr1", "chr2", "chrX") and Ensembl ("1", "2", "X") conventions of chromosome names, which should be consistent in the Hi-C data and ecDNA cycle file. 
#### Output of Step 1
Outputs from ```extract_matrix.py``` include 
- (i) the ICE normalized collapsed matrix ```*_ice_normalized.npy```;
- (ii) the raw collapsed matrix ```*_original_matrix.npy```; and
- (iii) an annotation file ```*_annotations.bed``` which maps each bin to the indices in the expanded matrix. Example annotation file from D458 ecDNA, with 10K resolution:
```
chr8	128460000	128470000	0
chr8	128470000	128480000	1
chr8	128480000	128490000	2
chr8	128490000	128500000	3
chr8	128500000	128510000	4
chr8	128510000	128520000	5	186
chr8	128520000	128530000	6	187
chr8	128530000	128540000	7
chr8	128540000	128550000	8
chr8	128550000	128560000	9
...
```
### Step 2 - Reconstructing the 3D structure of ecDNA
As a main functionality of ec3D, given the **(ICE) normalized** collapsed Hi-C matrix corresponding to ecDNA and annotations, compute the 3D coordinates for each fixed resolution bin on the ecDNA, and an expanded matrix in the existence of a duplicated segment. 
#### Usage
```python3 spatial_structure.py [Required arguments] [Optional arguments]```
#### Required arguments
- ```--matrix <FILE>```, ICE normalized, collapsed matrix in ```*.txt``` or ```*.npy``` format (```*_ice_normalized.txt/npy```) from ```extract_matrix.py```.
- ```--annotation <FILE>```, Annotation of bins in the input matrix (```*_annotations.bed```) from ```extract_matrix.py```.
- ```--output_prefix <STRING>```, Prefix of the output structure and expanded matrix file. Again if these file is desired to be written to a different directory, then a path/directory should also be included.
#### Optional arguments
ec3D optimizes the Poisson likelihood with the l-BFGS algorithm implemented in [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html). In order to tune the optimization process, it allows to reset the following parameters: 
- ```--reg <FLOAT>```, Weight parameter of the regularization term, which controls the **variance** of Euclidean distance between every pair of consecutive bins; default value is 0.05.
- ```--init_alpha <FLOAT>```, An initial value of alpha, the power law decay parameter of Hi-C with respect to Euclidean distance, for iterative optimization of the Poisson likelihood evaluated with X (the 3D coordinates) and alpha; default value is -3.0.
  - Note that ec3D works with another parameter beta, which controls the scale of Hi-C interactions relative to power law decay. This parameter can be estimated automatically from given alpha, so there is no need to reset it.
- ```--num_repeats <INT>``` - To overcome potential local minimums, ec3D starts with random initial structures and performs optimization multiple times. ```--num_repeats``` controls the number of times the optimization is repeated (with random initial structures). Default value is 5.
- ```--max_rounds <INT>``` - ec3D performs iterative optimization of the Poisson likelihood evaluated with the 3D coordinates and alpha, ```--max_rounds``` controls how many times this iterative process is repeated. Default value is 1000.
- ```--max_iter_likelihood <INT>```, Maximum number of L-BFGS iterations for estimating the 3D coordinates, default value is 10000.
- ```--max_iter_exponent <INT>```, Maximum number of L-BFGS iterations for estimating alpha, default is 5000.

And finally, to work with ```*.npy``` format, you can again specify
- ```--save_npy``` to save the output numpy matrices in ```*.npy``` format.
#### Output of Step 2
After ```spatial_structure.py``` is completed, the following files will be written to the path specified in ```--output_prefix```:
- (i) Most importantly, ```*_coordinates.txt/npy```, the optimal ecDNA structure reconstruction as 3D coordinates for each bin in the expanded matrix, sorted by the indices of bins specified in the input annotation file;
- (ii) The optimal values of alpha and beta in ```*_hyperparameters.txt```; and
- (iii) The expanded Hi-C matrix, ```*_expanded_matrix.txt/npy```, which is computed from the **raw** collapsed Hi-C matrix, annotation file and structure file, and finally ICE normalized, to be used in the following steps, e.g., identifying significant interactions. 

### Step 3 - Identifying significant interactions 
The capability of ec3D to expand the ecDNA Hi-C matrix additionally enables its identification of significant interactions within an expanded matrix. Given the (ICE) normalized, expanded matrix corresponding to ecDNA, ```significant_interactions.py``` provides an option to identify significant interactions from a  **circular genome**, and separate significant interactions from different causes (SV breakpoints, conformational changes, and candidate _trans_-interactions between different ecDNA molecules).
#### Usage
```python3 significant_interactions.py [Required arguments] [Optional arguments]```
#### Required arguments
- ```--matrix <FILE>```, ICE normalized, expanded matrix in ```*.txt``` or ```*.npy``` format (```*_expanded_matrix.txt/npy```) from ```spatial_structure.py```.
- ```--output_prefix <STRING>```, Prefix of the output significant interactions and clusterings.
#### Optional arguments
- ```--model <distance_ratio|global_poisson|global_nb>``` - The statistical model and quantities used to compute significant interactions, default setting is ```global_nb```.
  - If ```global_poisson``` or ```global_nb``` is specified, ec3D will compute significant interactions based on Hi-C interactions, and at each genomic distance specified in ```--genomic_distance_model```. The only difference is the statistical model of significant interactions - ```global_poisson``` fits a Poisson distribution of interactions at each genomic distance; while ```global_nb```  fits a Negative Binomial distribution, which can better capture overdispersion in Hi-C interactions.
  - If ```distance_ratio``` is specified, significant interactions will be computed based on the **ratio** between Euclidean distance and genomic distance. This model is designed to capture interactions within a single 3D structure, or minimize the impact of significant interactions due to alternative 3D structures/conformations and trans-interactions between different ecDNA copies.
- ```--genomic_distance_model <circular|linear|reference>```, Model of genomic distance between two bins, default is ```circular```.
  - ```circular``` - Genomic distance between bin ```i``` and bin ```j``` equals to ```min(|i - j|, N - |i - j|)``` where ```N``` is the size of the expanded matrix (i.e., number of fixed resolution bins in ecDNA). In other words, interactions in the ```i```-th diagonal and ```N - i```-th diagonal are considered to have the same genomic distance. When applied to ecDNA Hi-C, this setting will minimize the impact of significant interactions due to circularization of ecDNA, and prioritize interactions due to  conformational changes.
  - ```linear``` - Genomic distance between bin ```i``` and bin ```j``` equals to ```|i - j|```. This setting can be applied to normal chromosome Hi-C.
  - ```reference``` - Genomic distance between bin ```i``` and bin ```j``` equals to their distance on the reference genome, bounded by the ecDNA size, i.e., if ```i``` and ```j``` are from different chromosomes or their distance is larger than the size of the input ecDNA, their genomic distance is reset to the ecDNA size. When applied to ecDNA Hi-C, this genomic distance model will capture significant interactions due to circularization of ecDNA, or SV breakpoint joining remote chromosomal segments, potentially suggesting functional impacts/alterations (for example, enhancer hijacking).
- ```--structure <FILE>```, The reconstructed 3D structure of ecDNA in *.txt or *.npy format (```*_coordinates.txt/npy```), only required when ```distance_ratio``` is specified in ```--model```, otherwise ignored.
- ```--annotation <FILE>```, Annotation of bins in the input matrix (```*_annotations.bed```), only required when ```distance_ratio``` is specified in ```--model```, otherwise ignored.
- ```--pval_cutoff <FLOAT>```, (Adjusted) P-value cutoff for significant interactions, default value is 0.05.
- ```--max_pooling```, Only keep significant interactions larger than their top, bottom, left and right neighbors.
- ```--exclude <INT1 INT2, ...>```, Exclude significant interactions at given indices in the output and subsequent clustering process.
#### Output of Step 3
- (i) A ```*.tsv``` file describing the significant interactions computed from the input matrix ```*_significant_interactions.tsv```. Columns represent indices of the two bins involved in each significant interaction (given by annotation file); (normalized) Hi-C interaction frequencies; Poisson/Negative Binomial P-values and adjusted P-values, respectively. An example ```*_significant_interactions.tsv``` from D458 ecDNA:
```
bin1	bin2	interaction	p_value	q_value
0	348	36.323594	0.000428	0.039651
0	349	40.438812	0.000011	0.002524
0	350	37.250238	0.000248	0.027096
1	346	39.230045	0.000054	0.008436
1	347	41.097461	0.000037	0.006483
1	349	43.453360	0.000009	0.002167
1	350	40.806230	0.000011	0.002524
...
```
- (ii) (Louvain) Clustering of significant interactions, in the form of a ```*.tsv``` file ```*_clustered_bins.tsv```, which maps each bin involved in a significant interaction to the corresponding cluster of that bin.

### Step 4 - Visualization
ec3D by default supports 2 stylistic plotting functionalities, respectively clarifying the 3D structure (output from Step [2](#step-2---reconstructing-the-3d-structure-of-ecdna)) and the significant interaction of an ecDNA (output from Step [3](#step-3---identifying-significant-interactions)).

#### 4-1 Plotting 3D structures
Usage: ```python3 plot_structure.py [Required arguments] [Optional arguments]```
#### Required arguments
- ```--structure <FILE>```, the structure reconstruction ```*_coordinates.txt/npy``` output from ```spatial_structure.py```.
- ```--output_prefix <STRING>```, Prefix of the output plot(s).
#### Optional arguments
By default, only the structure will be rendered in the output html/png (see the left plot below). To facilitate interpretation and downstream analysis, one useful option is to plot the (onco)genes amplified on ecDNA (see the middle plot below), and for this you will need to specify  
- ```--annotation```, Annotation of bins (```*_annotations.bed```) used in reconstructing the 3D structure.
- ```--ref <hg19|GRCh37|hg38|GRCh38|mm10>```, The reference genome of the input ecDNA cycles (which should match the reference genome used for Hi-C preprocessing with e.g., HiC-pro). Currently, ec3D supports three reference genomes: hg19/GRCh37, hg38/GRCh38, and mm10.
- One of ```--download_gene``` or ```--gene_fn <FILE>```. By default, ec3D will plot the selected oncogenes given in its ```data_repo``` if neither of ```--download_gene``` or ```--gene_fn``` is specified. If ```--download_gene``` is specified, ec3D will download the list of NCBI RefSeq genes provided by UCSC genome browser, and plot all genes in this list that (even partially) overlap with the ecDNA. You may also provide a local, custom gene list, in ```*.gff``` or ```*.gtf``` format, in ```--gene_fn```.  

Another plotting option provided by ```plot_structure.py``` is to include the significant interactions as well as their clusters (see the right plot below), and for this purpose you will need
- ```--interactions <FILE>```, The ```*_significant_interactions.tsv``` output from Step 3. 
- ```--clusters <FILE>```, The ```*_clustered_bins.tsv``` output from Step 3.
![alt text](https://github.com/AmpliconSuite/ec3D/blob/main/images/D458_3D.png)

Finally, ec3d additionally provides an option ```--noncyclic``` to plot a non-cyclic structure, including the 3D structure of a normal chromosomal segment. 

#### Output of Step 4-1
A single ```*_ec3d.html``` image, which allows to freely rotate the structure, and show/hide certain elements such as genes, bin numbers, breakpoints (that form the ecDNA), interactions and clusters. Adding ```--save_png``` can additionally take a screenshot of the structure in default view and save it as ```*_ec3d.png``` with the prefix specified in ```--output_prefix```.

#### 4-2 Plotting significant interactions
```python3 plot_interactions.py [Required arguments] [Optional arguments]```

#### Required arguments
The minimum requirement of a significant interaction plot is just the ecDNA cycle and the corresponding expanded matrix, so that only the Hi-C matrix (corresponding to ecDNA intervals) is plotted.  
- ```--ecdna_cycle <FILE>```, ecDNA intervals in ```*.bed``` format (the same file used in [Preprocessing Hi-C](#step-1---preprocessing-hi-c)).
- ```--resolution <INT>```, Resolution used in the above reconstrcutions.
-	```--matrix <FILE>```, ICE normalized and expanded matrix in ```*.txt``` or ```*.npy``` format (```*_expanded_matrix.txt/npy```). Can also input a collapsed matrix here, see below for the ```--plot_collapsed_matrix``` option.
- ```--output_prefix <STRING>```, Prefix of the output plot(s).

#### Optional arguments
- ```--interactions <FILE>```, The ```*_significant_interactions.tsv``` output from Step 3, to be visualized. Significant interactions will be plotted in the upper triangular part of the matrix. 
- ```--sv_list <FILE>```, List of SV breakpoints, in [AmpliconClassifier](https://github.com/AmpliconSuite/AmpliconClassifier) ```*_SV_summary.tsv``` format. The first four columns should define the breakpoint in the form of ```chr1  pos1  chr2  pos2```. Breakpoints forming the ecDNA will appear on the diagonals. Addtional SV breakpoints that do not appear on the main ecDNA species will be plotted in the lower triangular part of the matrix. Also note that when these additional SV happen on a duplicated segment, they will occur (and be plotted) multiple times in an expanded matrix. 
- ```--annotation <FILE>```, Annotation ```*.bed``` file, requested when either ```--sv_list``` or ```--plot_collapsed_matrix``` is specified.
- ```--plot_collapsed_matrix```, Plot significant interactions on top of a collapsed matrix. Ec3D assumes that the input significant interactions are defined on an expanded matrix, so this option requires an annotation file to map interactions to indices in collapsed matrix. 
	
#### Output of Step 4-2
- Running ```plot_interactions.py``` will produce a copy of ```*_expanded/collapsed_matrix.png```, and another copy of ```*_expanded/collapsed_matrix.pdf``` images with the prefix specified in ```--output_prefix```. See below for examples. Top left: plot of expanded matrix; Top right: plot of expanded matrix along with significant interactions; Bottom left: plot of collapsed matrix along with significant interactions; Bottom right: plot of expanded matrix with significant interactions and additional SVs identified by AmpliconArchitect.
![alt text](https://github.com/AmpliconSuite/ec3D/blob/main/images/D458_5000.png)
