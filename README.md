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
import numpy as np
from optimisers import MultiSurrogateOptimiser
from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2
        out["F"] = [f1, f2]

problem = MyProblem()
optimi = MultiSurrogateOptimiser(problem, [661,12], [0,0])
out = optimi.solve(n_iterations=10, display_pareto_front=True, n_init_samples=10)
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
