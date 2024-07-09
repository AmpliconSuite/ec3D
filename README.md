# ec3D: reconstructing the 3D structure of extrachromosomal DNA molecules

## Dependencies 
- Cooler (https://cooler.readthedocs.io/)
- ICED (https://members.cbio.mines-paristech.fr/~nvaroquaux/iced/)
- SciPy (https://scipy.org/)
- scikit-learn (https://scikit-learn.org)
- autograd (https://github.com/HIPS/autograd)
- NetworkX (https://networkx.org/)
- python-louvain (https://python-louvain.readthedocs.io/en/latest/)
- pandas (https://pandas.pydata.org/)
- Plotly (https://plotly.com/python/) and Kaleido (https://pypi.org/project/kaleido/)

## Installation

## Running
### Batch mode
```
python3 ec3D.py --cool <path of *.cool file>
--ecdna_cycle <path of *.bed file>
--resolution 10000
--output_prefix <prefix of output, including path>
```

### Step 1 - Preprocessing Hi-C
Given the whole genome Hi-C (in ```*.cool``` format) and the ecDNA cycle (in ```*.bed``` format), output the Hi-C matrix corresponding to ecDNA to work on in the following steps, and the annotation of each bin at a given resolution. Example command:
```
python3 extract_matrix.py --cool <path of *.cool file>
--ecdna_cycle <path of *.bed file>
--resolution 10000
--output_prefix <prefix of output, including path>
```
- Example cycles file (from D458):
```
#chr	start	end	orientation	cycle_id	iscyclic	weight
chr8	128458992	129085009	+	1	True	1.000000
chr8	127293093	127948583	+	1	True	1.000000
chr8	127872074	128441212	-	1	True	1.000000
chr8	128505994	128527415	+	1	True	1.000000
chr8	127293093	127455873	-	1	True	1.000000
chr8	129033869	129085009	-	1	True	1.000000
chr8	127850424	127870604	+	1	True	1.000000
chr14	56593089	56794203	+	1	True	1.000000
chr14	56797826	56986857	-	1	True	1.000000
chr8	127957689	128012988	-	1	True	1.000000
chr8	128443842	128452940	+	1	True	1.000000
```
- The provided ecDNA structure in a cycle bed file may include duplicated segments in its records, e.g., ```chr8 127293093 127948583``` and ```chr8 127872074 128441212```.  We refer to _collapsed_ matrix as the Hi-C matrix where each duplicated segment occurs only one time; and _expanded_ matrix as the Hi-C matrix representing the structure of ecDNA where all duplicated segments occur as many times as they are duplicated. 
- Outputs from ```extract_matrix.py``` include (i) the ICE normalized collapsed matrix *_ice_normalized.npy, (ii) the raw collapsed matrix *_original_matrix.npy, and (iii) an annotation file ```*_annotations.bed``` which maps each bin to the indices in the expanded matrix. Example annotation file from D458:
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
Given the **(ICE) normalized** collapsed Hi-C matrix corresponding to ecDNA and the annotations, compute the 3D coordinates for each bin in the expanded matrix, and the expanded matrix itself. Example command: 
```
python3 spatial_structure.py --matrix <*.npy output from the last step>
--annotation <*.bed output from the last step>
--output_prefix <prefix of output, including path>
--log_fn <logs for optimization>
```
- ```spatial_structure.py``` will output a single ```*_coordinates.npy``` file storing the 3D coordinates for each bin in the expanded matrix sorted by the indices of bins specified in the annotation file.
- To compute the expanded Hi-C matrix, run ```expand_hic.py``` which takes the **raw** collapsed Hi-C matrix, the annotation file and the ```*_coordinates.npy``` above as input, expands the matrix, runs ICE normalization, and saves the normalized expanded matrix to an ```*.npy``` file. 
```
python3 expand_hic.py --raw_matrix <*.npy output from the last step>
--annotation <*.bed output from the last step>
--structure <*_coordinates.npy>
--output_prefix <prefix of output, including path>
```
### Step 3 - Identifying significant interactions 
Given the (ICE) normalized expanded matrix corresponding to ecDNA, identify the significant interactions from the matrix **under the assumption that the underlying genome is circular**. Example command: 
```
python3 significant_interactions.py --matrix <*.npy output from the last step>
--output_prefix <prefix of output, including path>
--pval_cutoff <float between (0, 1) indicating the P-value cutoff as significant interactions>
--log_fn <logs for significant interaction>
```
- The single output by ```significant_interactions.py``` is a ```*_significant_interactions.csv``` file indicating the significant interactions in the input matrix where the indices of bins are given by the annotation file. Example ```*_significant_interactions.csv```:
```
bin1,bin2,count,distance,p_value,q_value
2,26,50.325278,24,0.000067,0.013997
3,26,61.347507,23,0.000005,0.001745
7,89,24.410618,82,0.000267,0.039469
7,355,14.389584,292,0.000160,0.027056
13,109,24.362287,96,0.000061,0.012875
13,346,14.232664,307,0.000142,0.024759
...
```
### Step 4 - Visualization
ec3D by default will output 3 stylistic plots clarifying the 3D structure as well as the significant interaction of an ecDNA.
```
python3 plot_interactions.py --ecdna_cycle <path of *.bed file>
--matrix <*.npy output from expand_hic.py>
--annotation <*.bed output from the last step>
--interactions <*_significant_interactions.csv from the last step>
--sv_list <optional AmpliconArchitect or AmpliconClassifier SV list>
--output_prefix <prefix of output, including path>
```
- Running ```plot_interactions.py``` will produce the following output image.
![alt text](https://github.com/kyzhu/ecDNA-3D-structure/blob/main/images/D458-5000.png)
```
python3 plot_structures.py --structure <*_coordinates.npy>
--interactions <*_significant_interactions.csv from the last step>
--output_prefix <prefix of output, including path>
--log_fn <logs for optimization>
```
- Running ```plot_structures.py``` will produce the following two images.
![alt text](https://github.com/kyzhu/ecDNA-3D-structure/blob/main/images/D458_3D.png)
![alt text](https://github.com/kyzhu/ecDNA-3D-structure/blob/main/images/hic_dis_correlation.png)
