#!/usr/bin/env python
import setuptools

VERSION = '0.1.0'

# Read in requirements
with open('docker/requirements.txt', 'r') as rf:
    # Setuptools can't deal with git+https references
    required_packages = rf.readlines()


# Fill in all this for the specific project
setuptools.setup(
    name='fair_conformal',
    version=VERSION,
    description='Code to reproduce fair conformal predictors',
    author='clu5',
    maintainer='clu5',
    platforms=['Linux'],
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    # Requirements match exactly the dockerfile requirements
    install_requires=required_packages,
    # Everything in the packages resources directory is added as a resource
    # for inclusion wherever the package is installed
    # This is intended for config files
    package_data={
        '': ['configs/*']
    },
    # Set up the main entrypoint script for the entire project
    # This defines the ml_proj command-line command
    entry_points={
        'console_scripts': [
            'fair_conformal=fair_conformal.main:main'
        ]
    }
)
