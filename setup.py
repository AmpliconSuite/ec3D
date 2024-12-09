from setuptools import setup, find_packages


if __name__ == "__main__":
	setup(
		name = 'ec3d',
		version = '0.0.1',    
		description = 'Reconstructing the 3D structure of extrachromosomal DNA and other focal amplifications',
		url = 'https://github.com/AmpliconSuite/ec3D',
		author = 'Kaiyuan Zhu, Chaohui Li, Biswanath Chowdhary',
		author_email = 'kaiyuan-zhu@ucsd.edu, chl221@ucsd.edu',
		license = 'BSD 3-clause',
		packages = find_packages('src'),
		install_requires = ['cooler',
				'iced',
				'scipy',
				'scikit-learn',
				'autograd',
				'networkx',
				'python-louvain',
				'plotly',
				'kaleido'],

		classifiers = ['Development Status :: 1 - Planning',
				'Intended Audience :: Science/Research',
				'License :: OSI Approved :: BSD License',  
				'Operating System :: POSIX :: Linux',        
				'Programming Language :: Python :: 3',
				'Programming Language :: Python :: 3.7',],
	)
