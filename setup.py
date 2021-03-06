#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, find_namespace_packages



with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()
	


Version = '1.3'


with open('requirements_dev.txt') as requirements_file:
    read_lines = requirements_file.readlines()
    requirements = [line.rstrip('\n') for line in read_lines]
	

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(


    author="Philipe Riskalla Leal",
    author_email='leal.philipe@gmail.com',
    classifiers=[
        'Topic :: Education',                                        # this line follows the options given by: [1]
        "Topic :: Scientific/Engineering",    # this line follows the options given by: [1]		
        'Intended Audience :: Education',
		"Intended Audience :: Science/Research",
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
		'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="""This library operates Space-Time Match-Up operations over Netcdf-Xarray datasets and Geopandas-GeoDataFrames. \
				 It is a mandatoryr step for areas of study as geography, epidemiology, sociology, remote sensing, ecology, etc.""",
    
    install_requires=requirements,
    license="MIT license",
    
    include_package_data=True,
	
	python_requires='>=3.4',  # Your supported Python ranges
	
    keywords='time space Match Up xarray geopandas space-time reduction',
    name='time_space_reductions',
	
	packages=find_packages(exclude='/tests/*', include='/time_space_reductions/*'),
	package_dir = {'': 'time_space_reductions'},
    
    setup_requires=setup_requirements,
    test_suite='nose.collector',
    tests_require=['nose'],
    url='https://github.com/PhilipeRLeal/time_space_reductions',
    version=Version,
    zip_safe=False,
)




# [1]: https://pypi.org/classifiers/