from setuptools import setup, find_packages


if __name__ == "__main__":
	setup(
		name = 'ec3D',
		version = '0.0.1',    
		description = 'Reconstructing the 3D structure of extrachromosomal DNA and other focal amplifications',
		url = 'https://github.com/AmpliconSuite/ec3D',
		author = 'Kaiyuan Zhu, Chaohui Li, Biswanath Chowdhary',
		author_email = 'kaiyuan-zhu@ucsd.edu, chl221@ucsd.edu',
		license = 'BSD 3-clause',
		packages=find_packages(where="."),
    	package_dir={"": "."},
		include_package_data=True,
		package_data={
        "ec3D": ["data_repo/*"],
    },
		install_requires = ['cooler>=0.10.2',
				'iced>=0.5.10',
				'scipy>=1.10.1',
				'scikit-learn>=1.3.2',
				'autograd>=1.6.2',
				'networkx>=3.1',
				'python-louvain>=0.16',
				'plotly>=5.24.1',
				'kaleido>=0.2.1',
				'statsmodels>=0.14.5',
				'matplotlib>=3.7.1'],
		python_requires=">=3.8",
		entry_points={
			"console_scripts": [
				"ec3D=ec3D.main:main",
			],
		},
		classifiers = ['Development Status :: 1 - Planning',
				'Intended Audience :: Science/Research',
				'License :: OSI Approved :: BSD License',  
				'Operating System :: POSIX :: Linux',        
				'Programming Language :: Python :: 3',
				'Programming Language :: Python :: 3.7',],
	)
