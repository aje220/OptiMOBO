# MOBO
Solve multi-objective problems using multi-surrogate and mono-surrogate methods.

This repo contains implementations of multi-objective bayesian optimisation (MOBO) methods. 
The methods include: 
* **Mono-surrogate.** This uses a single model to optimise. Objective vectors are aggregated into a single scalar value and a Gaussian process is built upon the scalarised values.
* **Multi-surrogate.** This method uses multiple models. One model for each objective. Multi-objective acquisition functions are used to identify new sample points.

The two methods are written as two classes `MultiSurrogateOptimiser` and `MonoSurrogateOptimiser`.
They are designed to solve problems that inherit from the `ElementwiseProblem` shown in the library [pymoo](https://pymoo.org/index.html).

An example of how they can be used:
```
problem = MyProblem()
optimiser = MultiSurrogateOptimiser(problem, [661,12],[0,0])
results = optimiser.solve(n_iterations=100, display_pareto_front=True, n_init_samples=10, acquisition_func="_PBI_")
```
The output `results` is a tuple containing:
* Solutions on the Pareto front approximation.
* The corresponding inputs to the solutions on the Pareto front.
* All evaluated solutions.
* All inputs used in the search.

#### Features
* Plotting of Pareto front approximation
* EITCH, EIPBI, EHVI, mulit-objective acquisition functions.

#### Requirements
* Numpy
* Scipy
* pygmo
* pymoo
