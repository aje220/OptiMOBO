# OptiMOBO
Solve bi-objective multi-objective problems using multi-objective bayesian optimisation (MOBO).

This repo contains implementations of two MOBO methods. They are designed to solve bi-objective minimisation problems, constraints are not supported.
The methods include:
* **Mono-surrogate.** This uses a single model to optimise. Objective vectors are aggregated into a single scalar value and a Gaussian process is built upon the scalarised values.
* **Multi-surrogate.** This method uses multiple models. One model for each objective. Multi-objective acquisition functions are used to identify new sample points.

The two methods are written as two classes `MultiSurrogateOptimiser` and `MonoSurrogateOptimiser`.
They are designed to solve problems that inherit from the `Problem` class presented in the library [pymoo](https://pymoo.org/index.html).

#### Examples 
The following code defines a bi-objective problem, MyProblem, and uses multi-surrogate Bayesian optimisation (utilising Tchebicheff aggregation as an acquisition function) to solve.
```python
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
optimi = MultiSurrogateOptimiser(problem, [0,0], [700,12])
out = optimi.solve(n_iterations=100, display_pareto_front=True, n_init_samples=20, sample_exponent=3, acquisition_func=Tchebicheff([0,0],[700,12]))
```

Will return a Pareto set approximation:

![MyProblem](docs/media/dadsdaa.png "MyProblem Pareto Approximation")

For the multi-objective benchmark problem DTLZ5:
```python
from pymoo.problems import get_problem
problem = get_problem("dtlz5", n_obj=2, n_var=5)
optimi = MultiSurrogateOptimiser(problem, [0,0], [1.3,1.3])
out = optimi.solve(n_iterations=100, display_pareto_front=True, n_init_samples=20, sample_exponent=3, acquisition_func=Tchebicheff([0,0],[1.3,1.3])) 
```

Will return:

![DTLZ5](docs/media/adwdawdssss.png "DTLZ5 Pareto front approximation")




The output `results` is a tuple containing:
* Solutions on the Pareto front approximation.
* The corresponding inputs to the solutions on the Pareto front.
* All evaluated solutions.
* All inputs used in the search.

## Key Features
#### Mono and multi-surrogate:
Two optimisers based on differing methods. 

#### Choice of acquisition/aggragation functions:
In mono-surrogate MOBO, scalarisation functions are used to aggregate objective vectors in a single value that can be used by the optimsier.
In multi-surrogate MOBO, scalarisation functions are used as convergence measures to select sample points.
This package contains 10 scalarisation functions that can be used in the above mentioned contexts.
Options Include:
* Weighted Sum
* Tchebicheff
* Modified Tchebicheff
* Augmented Tchebicheff
* Weighted Norm
* Weighted Power
* Weighted Product
* PBI
* IPBI
* Exponential Weighted Criterion

They are written so they can be used in any context.

#### Experimental Parameters
Various experimental parameters can be customised.


## Requirements
* numpy
* scipy
* pygmo
* pymoo
