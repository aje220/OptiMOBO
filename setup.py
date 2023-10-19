from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
   name='optimobo',
   version='0.2.1',
   license='MIT',
   description='Solve multi-objective problems using Bayesian optimisation.',
   long_description=long_description,
   long_description_content_type='text/markdown',
   author='aje220',
   author_email='alexjacevans@gmail.com',
   packages=find_packages(),  #same as name
   install_requires=['numpy<=1.23.5', 'pymoo>=0.6.0', 'pygmo>=2.0.0', 'scipy>=1.9.0', 'gpy>=1.10.0'], #external packages as dependencies
)