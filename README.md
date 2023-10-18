# OptiMOBO
Solve multi-objective optimisation problems using multi-objective bayesian optimisation (MOBO).

This repo is a toolbox for solving expensive to evaluate multi-objective problems. It contains implementations of MOBO methods;
the methods include:

 
* **Generic Mono-surrogate.** This uses a single model to optimise. Objective vectors are aggregated into a single scalar value and a Gaussian process is built upon the scalarised values.
* **Generic Multi-surrogate.** This method uses multiple models, one model for each objective. Multi-objective acquisition functions are used to identify new sample points. Scalarisation functions can be exploited as multi-objective acquisition functions in this paradigm.
* **ParEGO.** A mono-surrogate method proposed in 2006. This uses evolutionary operators to select new sample points.
* **ParEGO-C1/C2.** Mono-surrogate methods that feature constraint handling (see example for how to define constraints).
* **EMO.** Multi-surrogate method, implemented using Hypervolume-based PoI as an acquisition method.
* **KEEP.** An extension of ParEGO that includes a second surrogate model to improve selection of sample points.
* **Multi-objective TuRBO.** Designed for single objective high dimensional problems. It has been adapted for unconstrained multi-objective problems via the addition of a mono-surrogate method.

The methods are written as classes.
They are designed to solve problems that inherit from the `Problem` class.
Problems are defined by implementing the ```_evaluate()``` and ```_evaluate_constraints()``` methods.

#### Examples 
The following code defines a bi-objective problem, MyProblem, and uses multi-surrogate Bayesian optimisation (utilising Tchebicheff aggregation as an acquisition function) to solve.
```python
import numpy as np
import optimobo.scalarisations as sc
import optimobo.algorithms.optimisers as opti
from optimobo.problem import ElementwiseProblem

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
optimi = opti.MultiSurrogateOptimiser(problem, [0,0], [700,12])
out = optimi.solve(budget=100, n_init_samples=20, sample_exponent=3, acquisition_func=sc.Tchebicheff([0,0],[700,12]))
out.plot_pareto_front()
plt.show()
```

Will return a Pareto set approximation:

![MyProblem](docs/media/Myproblem.png "MyProblem Pareto Approximation")

For a constrained problem the constraint functions are defined in a seperate method:
```python
from optimobo.algorithms.cparego import ParEGO_C1
from optimobo.problem import Problem
import optimobo.scalarisations as sc

class BNH(Problem):

    def __init__(self):
        super().__init__(n_var=2, 
                        n_obj=2, 
                        n_ieq_constr=2, 
                        vtype=float)
                        self.xl = np.zeros(self.n_var)
                        self.xu = np.array([5.0, 3.0])

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 4 * x[:, 0] ** 2 + 4 * x[:, 1] ** 2
        f2 = (x[:, 0] - 5) ** 2 + (x[:, 1] - 5) ** 2
        out["F"] = [f1, f2]
    
    def _evaluate_constraints(self, x, out, *args, **kwargs):
        g1 = (1 / 25) * ((x[:, 0] - 5) ** 2 + x[:, 1] ** 2 - 25)
        g2 = -1 / 7.7 * ((x[:, 0] - 8) ** 2 + (x[:, 1] + 3) ** 2 - 7.7)
        out["G"] = [g1, g2]
        
problem = BNH()
optimi = ParEGO_C2(problem, [0,0], [150,60])
out = optimi.solve(sc.Tchebicheff([0,0], [150,60]), budget=50, n_init_samples=20)
out.plot_pareto_front()
plt.show()
```

Will return:

![DTLZ5](docs/media/BNH_objective_space.png "BNH Pareto front approximation")




The output of each of the ```(algorithm).solve()``` method is an object containing these attributes:
* ```results.pf_approx``` Objective vectors on the Pareto front approximation. 
* ```results.pf_inputs``` The corresponding inputs to the solutions on the Pareto front, the best performing solutions.
* ```results.ysample``` All evaluated solutions/objective vectors.
* ```results.xsample``` All inputs/solutions used in the search. 
* ```results.hypervolume_convergence``` How the hypervolume changes from iteration to iteration.

For algorithms with constraint handling more information is included:
* ```results.pf_approx``` calculated from only feasible solutions.
* ```results.pf_inputs``` calculated from only feasible solutions.
* ```results.X_feasible``` and ```results.X_infeasible``` Solutions.
* ```results.y_feasible``` and ```results.y_infeasible``` Objective vectors.

#### Other information

<!-- When calling the ```optimiser.solve``` function for a ```MonoSurrogateOptimiser``` object, an aggregation function must be defined. -->

* For a ```MultiSurrogateOptimiser``` object a scalarisation function can be chosen as a convergence measure. However, if left default, the optimiser will use Expected Hypervolume Improvement (EHVI) to solve the problem. It should be noted that EHVI only works in 2 and 3 (crudely) dimensions.

## Installation
Can be installed via:

`pip install optimobo`

pygmo can struggle with Microsoft Windows; if you are using Windows anaconda is recommended.

## Key Features

#### Choice of acquisition/aggregation functions:
In mono-surrogate MOBO, scalarisation functions are used to aggregate objective vectors in a single value that can be used by the optimsier.
In multi-surrogate MOBO, scalarisation functions are used as convergence measures to select sample points.
This package contains 10 scalarisation functions that can be used in the above mentioned contexts.
Options Include:
* Weighted Sum (WS)
* Tchebicheff (TCH)
* Modified Tchebicheff (MTCH)
* Augmented Tchebicheff (ATCH)
* Weighted Norm (WN)
* Weighted Power (WPO)
* Weighted Product (WPR)
* Penality Boundary Intersection (PBI)
* Inverted PBI (IPBI)
* Quadratic PBI (QPBI)
* Exponential Weighted Criterion (EWC)
* Angle Penalised Distance (APD)

They are written so they can be used in any context.
Their contours (with a weighting of `[0.5, 0.5]` and upper and lower bounds of `[-500, -450] [500, 1000]`) can be seen here:
![ScalarisationContours](docs/media/scalarisations.png "Contours of the scalarisation functions, in 2D.")

#### Utility Functions
Aside from the algorithms and scalarisations themselves this package includes implementations of useful functions. They are found in the util_functions file and include:
* **WFG:** A function to calculate the hypervolume of a set of objective vectors.
* **Exclusive Hypervolume** Calculate the exclusive hypervolume of an objective vector.
* **Inclusive Hypervolume** Calculate the inclusive hypervolume of an objective vector.
* **Modified WFG:** A modified version of WFG that can break down a non-dominated space into cells and returns the coordinates of each cell (in the objective space). Currently only works in 2D.
* **generate_latin_hypercube_samples** A method of producing latin hypercube samples in any ```n``` dimensional space.
* **EHVI** The expected hypervolume improvement of an objective vector. 2D and crudely in 3D.
* **Expected decomposition:** Scalarisation functions can be used as multi-objective acqusition functions. Scalarisation functions can be used as a performance measure, this quality can be exploited to measure the convergence of solution in the objective space. Optimising the expected decomposition of a problem can identify new sample points.

#### Experimental Parameters
Various experimental parameters can be customised:
* Number of iterations/budget. How many expensive function evalutaions used in the optimisation process.
* Number of initial samples

See the implementations of each algorithm for details.

#### Visualisation
By calling ```result.plot_pareto_front()``` from a result object method, the program will use matplotlib to plot the objective space of the problem at after the final iteration.
The hypervolume convergence can be displayed also. Calling ```result.plot_hv_convergence()``` will plot how the hypervolume changes iteration to iteration.

## Requirements
See requirements.txt.
Due to a bug in GPy, an older version of numpy is required.
You can install the required version with: ```pip install numpy==1.23.5```