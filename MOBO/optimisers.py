
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import differential_evolution
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.core.problem import ElementwiseProblem


from util_functions import EHVI, calc_pf, chebyshev, PBI, EIPBI, EITCH
from scalarisations import ExponentialWeightedCriterion, IPBI, PBI, chebyshev, WeightedNorm, WeightedPower, WeightedProduct, AugmentedTchebicheff, ModifiedTchebicheff, chebyshev

class MultiSurrogateOptimiser:
    """
    Class that allows optimisation of multi-objective problems using a multi-surrogate methodology.
    This method creates multiple probabalistic models, one for each objective.
    Constraints not supported.
    """

    def __init__(self, test_problem, max_point, ideal_point):
        self.test_problem = test_problem
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.upper = test_problem.xu
        self.lower = test_problem.xl


    def _objective_function(self, problem, x):
        return problem.evaluate(x)

    
    def _get_proposed(self, function, models, ideal_point, max_point, min_val, ref_dir, cache):
        """
        Function to retieve the next sample point.
        """
        def obj(X):
            return -function(X, models, ref_dir, ideal_point, max_point, min_val, cache)

        x = list(zip(self.lower, self.upper))
        res = differential_evolution(obj, x)
        return res.x, res.fun, ref_dir


    def _get_proposed_EHVI(self, function, models, ideal_point, max_point, ysample, cache):
        """
        Function to retrieve the next sample point using EHVI.
        This is a seperate function to _get_proposed as the parameters are different.
        
        """
        def obj(X):
            return -function(X, models, ideal_point, max_point, ysample, cache)

        # x = [(bounds[0], bounds[1])] * n_var
        x = list(zip(self.lower, self.upper))
        res = differential_evolution(obj, x)
        return res.x, res.fun


    
    def solve(self, n_iterations=100, display_pareto_front=False, n_init_samples=5, acquisition_func="_TCH_"):
        """
        This function attempts to solve the multi-objective optimisation problem.

        Params:
            n_iterations: the number of iterations 
            display_pareto_front: bool. When set to true, a matplotlib plot will show the pareto front approximation discovered by the optimiser.
            n_init_samples: the number of initial samples evaluated before optimisation occurs.
            acquisition_function: the acqusition function used to select new sample points. Options include:
                tchebicheff aggregation "_TCH_", PBI aggregation "_PBI_", Expected Hypervolume Improvement "_EHVI_".
        
        """
        # variables/constants
        problem = self.test_problem
        assert(problem.n_obj == 2)

        # Initial samples.
        sampler = qmc.LatinHypercube(d=problem.n_var)
        Xsample = sampler.random(n=n_init_samples)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])

        # Create cached samples, this is to speed up computation in calculation of the acquisition functions.
        sampler = qmc.Sobol(d=2, scramble=True)
        sample = sampler.random_base2(m=5)
        norm_samples1 = norm.ppf(sample[:,0])
        norm_samples2 = norm.ppf(sample[:,1])
        cached_samples = np.asarray(list(zip(norm_samples1, norm_samples2)))

        # Reference directions, one of these is radnomly selected every iteration, this promotes diverity.
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=100)

        for i in range(n_iterations):

            # Create models for each objective.
            models = []
            for i in range(problem.n_obj):
                model = GaussianProcessRegressor()
                model.fit(Xsample, ysample[:,i])
                models.append(model)

            # With each iteration we select a random weight vector, this is to improve diversity.
            ref_dir = ref_dirs[np.random.randint(0,len(ref_dirs))]

            # Retrieve the next sample point.
            X_next = None
            if acquisition_func in {"_TCH_"}:
                TCH_min = np.min([chebyshev(y, ref_dir, self.ideal_point, self.max_point) for y in ysample])
                X_next, _, _ = self._get_proposed(EITCH, models, self.ideal_point, self.max_point, TCH_min, ref_dir, cached_samples)
            elif acquisition_func in {"_PBI_"}:
                PBI_min = np.min([PBI(y, ref_dir, self.ideal_point, self.max_point) for y in ysample])
                X_next, _, ref_dir = self._get_proposed(EIPBI, models, self.ideal_point, self.max_point, PBI_min, ref_dir, cached_samples)
            elif acquisition_func in {"_EHVI_"}:
                X_next, _ = self._get_proposed_EHVI(EHVI, models, self.ideal_point, self.max_point, ysample, cached_samples)
            else:
                raise(Exception())

            # Evaluate the next input.
            y_next = self._objective_function(problem, X_next)

            # Add the new sample.
            ysample = np.vstack((ysample, y_next))

            # Update archive.
            Xsample = np.vstack((Xsample, X_next))

        # Get hypervolume metric
        ref_point = self.max_point
        HV_ind = HV(ref_point=ref_point)
        pf_approx = calc_pf(ysample)
        hv = HV_ind(ysample)

        if display_pareto_front:
            plt.scatter(ysample[5:,0], ysample[5:,1], color="red", label="Samples.")
            plt.scatter(ysample[0:n_init_samples,0], ysample[0:n_init_samples,1], color="blue", label="Initial samples.")
            plt.scatter(pf_approx[:,0], pf_approx[:,1], color="green", label="PF approximation.")
            plt.scatter(ysample[-1:-5:-1,0], ysample[-1:-5:-1,1], color="black", label="Last 5 samples.")
            plt.legend()
            plt.show()

        # Identify the inputs that correspond to the pareto front solutions.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        return pf_approx, pf_inputs, ysample, Xsample



class MonoSurrogateOptimiser:
    """
    Class that enables optimisation of multi-objective problems using a mono-surrogate methodology.
    Mono-surrogate method aggregates multiple objectives into a single scalar value, this then allows optimisation of
    a multi-objective problem with a single probabalistic model.
    """
    def __init__(self, test_problem, max_point, ideal_point):
        self.test_problem = test_problem
        # self.aggregation_func = aggregation_func
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.upper = test_problem.xu
        self.lower = test_problem.xl


    def _objective_function(self, problem, x):
        return problem.evaluate(x)

    def _expected_improvement(self, X, model, opt_value, kappa=0.001):
        """
        EI, single objective acquisition function.
        """
        # import pdb; pdb.set_trace()
        # get the mean and s.d. of the proposed point
        X_aux = X.reshape(1, -1)
        mu_x, sigma_x = model.predict(X_aux, return_std=True)

        mu_x = mu_x[0]
        sigma_x = sigma_x[0]
        # compute EI at that point
        gamma_x = (mu_x - opt_value - kappa) / (sigma_x + 1e-10)
        ei = sigma_x * (gamma_x * norm.cdf(gamma_x) + norm.pdf(gamma_x))

        return ei.flatten()

    def _get_proposed(self, function, models, current_best):
        """
        Helper function to optimise the acquisition function. This is to identify the next sample point.
        """

        def obj(X):
            # print("obj called")
            return -function(X, models, current_best)

        x = list(zip(self.lower, self.upper))

        res = differential_evolution(obj, x)
        return res.x, res.fun

    def _normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    
    def solve(self, n_iterations=100, display_pareto_front=False, n_init_samples=5, aggregation_func=None):
        """
        This function contains the main flow of the multi-objective optimisation algorithm. This function attempts
        to solve the MOP.

        Params:
            n_iterations: the number of iterations 
            display_pareto_front: bool. When set to true, a matplotlib plot will show the pareto front approximation discovered by the optimiser.
            n_init_samples: the number of initial samples evaluated before optimisation occurs.
            acquisition_function: the aggregation function used to aggregate the objective vectors in a single scalar value. Options include:
                tchebicheff aggregation "_TCH_", PBI aggregation "_PBI_".
        """
        
        # ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        problem = self.test_problem
        assert(problem.n_obj==2)

        problem.n_obj = 2

        # initial weights are all the same
        weights = np.asarray([0.5]*problem.n_obj)

        # get the initial samples used to build first model
        # use latin hypercube sampling
        sampler = qmc.LatinHypercube(d=problem.n_var)
        Xsample = sampler.random(n=n_init_samples)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])

        # # Aggregate initial samples.
        # if aggregation_func in {"_PBI_"}:
        #     aggregated_samples = [PBI(i, weights, self.ideal_point, self.max_point)[0] for i in ysample]
        # elif aggregation_func in {"_TCH_"}:
        #     aggregated_samples = [chebyshev(i, weights, self.ideal_point, self.max_point) for i in ysample]
        # elif aggregation_func in {"EWC"}:
        #     aggregated_samples = [exponential_weighted_criterion(i, weights) for i in ysample]
        # elif aggregation_func in {"IPBI"}:
        #     aggregated_samples = [IPBI(i, weights, self.ideal_point, self.max_point, 5)]

        #     Exception()

        # import pdb; pdb.set_trace()
        aggregated_samples = np.asarray([aggregation_func(i, weights) for i in ysample]).flatten()

        model = GaussianProcessRegressor()

        # Fit initial model.
        model.fit(Xsample, aggregated_samples)

        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=100)

        for i in range(n_iterations):
            # import pdb; pdb.set_trace()
            # Identify the current best sample, used for search.
            current_best = aggregated_samples[np.argmin(aggregated_samples)]

            # import pdb; pdb.set_trace()

            model.fit(Xsample, aggregated_samples)

            # use the model, current best to get the next x value to evaluate.
            next_X, _ = self._get_proposed(self._expected_improvement, model, current_best)

            # Evaluate that point to get its objective values.
            next_y = self._objective_function(problem, next_X)

            # add the new sample
            ysample = np.vstack((ysample, next_y))

            ref_dir = ref_dirs[np.random.randint(0,len(ref_dirs))]
            # print("Selected weight: "+str(ref_dir))

            # aggregate new sample
            # if aggregation_func in {"_PBI_"}:
            #     agg = PBI(next_y, ref_dir, self.ideal_point, self.max_point)[0]
            # elif aggregation_func in {"_TCH_"}:
            #     agg = chebyshev(next_y, ref_dir, self.ideal_point, self.max_point)
            # else:
            #     Exception()
            agg = aggregation_func(next_X, ref_dir)
            # agg = PBI(next_y, weights, ideal_point, max_point)[0]
            # aggregated_samples.append(agg)


            aggregated_samples = np.append(aggregated_samples, agg)

            # Add the variables into the archives.
            Xsample = np.vstack((Xsample, next_X))
        
        pf_approx = calc_pf(ysample)

        if display_pareto_front:
            plt.scatter(ysample[5:,0], ysample[5:,1], color="red", label="Samples.")
            plt.scatter(ysample[0:n_init_samples,0], ysample[0:n_init_samples,1], color="blue", label="Initial samples.")
            plt.scatter(pf_approx[:,0], pf_approx[:,1], color="green", label="PF approximation.")
            plt.scatter(ysample[-1:-5:-1,0], ysample[-1:-5:-1,1], color="black", label="Last 5 samples.", zorder=10)
            plt.legend()
            plt.show()

        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        
        pf_inputs = Xsample[indicies]

        return pf_approx, pf_inputs, ysample, Xsample





class MyProblemm(ElementwiseProblem):
    
    def __init__(self):
        super().__init__(n_var=2,
                        n_obj=2,
                        # n_ieq_constr=2,
                        xl=np.array([-2,-2]),
                        xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        # g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        # g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        # out["G"] = [g1, g2]

prob = MyProblemm()
optimi = MonoSurrogateOptimiser(prob, [661,12],[0,0])


# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=ExponentialWeightedCriterion(p=1))
# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=WeightedNorm())
# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=WeightedPower())
# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=PBI([0,0], [661,12]))
# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=IPBI([0,0], [661,12]))
# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=AugmentedTchebicheff([0,0], [661,12]))
# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=ModifiedTchebicheff([0,0], [661,12]))
# out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=chebyshev([0,0], [661,12]))





out = optimi.solve(n_iterations=50, display_pareto_front=True, n_init_samples=10, aggregation_func=WeightedProduct())




import pdb; pdb.set_trace()
print(out)
