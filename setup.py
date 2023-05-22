from setuptools import setup

setup(
   name='optimobo',
   version='0.1.2',
   license='MIT',
   description='Solve multi-objective problems using Bayesian optimisation.',
   author='aje220',
   author_email='aje220@exeter.ac.uk',
   packages=['optimobo'],  #same as name
   install_requires=['numpy>=1.20.0', 'pymoo>=0.6.0', 'pygmo>=2.0.0', 'scipy>=1.9.0', 'scikit-learn>=1.0.0'], #external packages as dependencies
)