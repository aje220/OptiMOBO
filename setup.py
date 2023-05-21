import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='OptiMOBO',
   version='0.1.0',
   license='MIT',
   description='Solve multi-objective problems using Bayesian optimisation.',
   author='aje220',
   author_email='aje220@exeter.ac.uk',
   packages=['OptiMOBO'],  #same as name
   install_requires=[required], #external packages as dependencies
)