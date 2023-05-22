
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import differential_evolution
from pymoo.util.ref_dirs import get_reference_directions
# from pymoo.indicators.hv import HV
# from pymoo.core.problem import ElementwiseProblem


# from util_functions import EHVI, calc_pf, expected_decomposition
from . import util_functions


# from scalarisations import ExponentialWeightedCriterion, IPBI, PBI, Tchebicheff, WeightedNorm, WeightedPower, WeightedProduct, AugmentedTchebicheff, ModifiedTchebicheff


class MultiSurrogateOptimiser:
    """
    Class that allows optimisation of multi-objective problems using a multi-surrogate methodology.
    This method creates multiple probabalistic models, one for each objective.
    Constraints not supported.

    Param:
        test_problem: problem to be solved. Defined via pymoo.
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
                in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
    """

    def __init__(self, test_problem, ideal_point, max_point):
        self.test_problem = test_problem
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.upper = test_problem.xu
        self.lower = test_problem.xl


    def _objective_function(self, problem, x):
        """
        Wrapper function to just evaluate the objective functions.
        """
        return problem.evaluate(x)

    
    def _get_proposed_scalarisation(self, function, models, min_val, scalar_func, ref_dir, cache):
        """
        Function to retieve the next sample point.
        This is called when a scalarisation function is being used as an acquisition function.
        Params:
            function: acqusition function to optimise
            models: array of the models trained on each objective
            min_val: the best scalarisation value from the current population; according to the ref_dir and 
                the scalarisation function used. 
            scalar_func: the scalarisation function being used as a convergence measure in the acquisition
                function
            ref_dir: randomly selected weight vector (row vector)
            cache: the premade samples that will transformed over the mean and standard deviation of each
                proposed point.

        returns:
            res.x: solution of the optimisation
            res.fun: values of the objective function
            ref_dir: weights used.
        """
        def obj(X):
            # return -function(X, models, ref_dir, ideal_point, max_point, min_val, cache)
            return -function(X, models, ref_dir, scalar_func, min_val, cache)

        x = list(zip(self.lower, self.upper))
        res = differential_evolution(obj, x)
        return res.x, res.fun, ref_dir


    def _get_proposed_EHVI(self, function, models, ideal_point, max_point, pf, cache):
        """
        Function to retrieve the next sample point using EHVI.
        This is a seperate function to _get_proposed_scalarisation as the parameters are different.

        Params:
            function: acqusition function to optimise.
            models: array of the models trained on each objective.
            ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
                in the objective space.
            max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
            pf: pareto front approximation of the current population
            cache: the premade samples that will transformed over the mean and standard deviation of each
                proposed point.
        
        returns:
            res.x: solution of the optimisation
            res.fun: values of the objective function 

        """
        def obj(X):
            return -function(X, models, ideal_point, max_point, pf, cache)

        # x = [(bounds[0], bounds[1])] * n_var
        # x = list(zip(self.lower, self.upper))
        x = list(zip(self.test_problem.xl, self.test_problem.xu))

        res = differential_evolution(obj, x)
        return res.x, res.fun


    
    def solve(self, n_iterations=100, display_pareto_front=False, n_init_samples=5, sample_exponent=5, acquisition_func=None):
        """
        This fcontains the main algorithm to solve the optimisation problem.

        Params:
            n_iterations: the number of iterations 
            
            display_pareto_front: bool. When set to true, a matplotlib plot will show the pareto front 
                approximation discovered by the optimiser.
            
            n_init_samples: the number of initial samples evaluated before optimisation occurs.

            acquisition_func: the acquisition function used to select the next sample points. 
                            If left default it calls Expected Hypervolume improvement
                            EHVI. Scalarisation functions can be used also;
                            options include: ExponentialWeightedCriterion, IPBI, PBI, Tchebicheff, 
                            WeightedNorm, WeightedPower, WeightedProduct, AugmentedTchebicheff, ModifiedTchebicheff
        

        Returns: Four lists in a tuple.
            pf_approx, solution vectors on the pareto front approximation found by the optimiser.
            pf_inputs, corresponding inputs to the values in pf_approx.
            ysample, output objective vectors of all evaluated samples.
            Xsample, all samples that were evaluated.
        """
        # variables/constants
        problem = self.test_problem
        # We can only solve 2 objective problems at the moment. This is caused by the fact EHVI can only handle 2 objectives.
        assert(problem.n_obj == 2)

        # Initial samples.
        sampler = qmc.LatinHypercube(d=problem.n_var)
        Xsample = sampler.random(n=n_init_samples)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])

        # Create cached samples, this is to speed up computation in calculation of the acquisition functions.
        sampler = qmc.Sobol(d=2, scramble=True)
        sample = sampler.random_base2(m=sample_exponent)
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
            ref_dir = np.asarray(ref_dirs[np.random.randint(0,len(ref_dirs))])

            # Retrieve the next sample point.
            X_next = None
            if acquisition_func is None:
                pf = util_functions.calc_pf(ysample)
                X_next, _, = self._get_proposed_EHVI(util_functions.EHVI, models, self.ideal_point, self.max_point, pf, cached_samples)
            else:
                min_scalar =  np.min([acquisition_func(y, ref_dir) for y in ysample])
                X_next, _, _ = self._get_proposed_scalarisation(util_functions.expected_decomposition, models, min_scalar, acquisition_func, ref_dir, cached_samples)


            # expected_decomposition([1,1], models, ref_dir, acquisition_func, min_scalar, cached_samples)

            # Evaluate the next input.
            y_next = self._objective_function(problem, X_next)

            # Add the new sample.
            ysample = np.vstack((ysample, y_next))

            # Update archive.
            Xsample = np.vstack((Xsample, X_next))

        # Get hypervolume metric.
        # ref_point = self.max_point
        # HV_ind = HV(ref_point=ref_point)
        # hv = HV_ind(ysample)

        pf_approx = util_functions.calc_pf(ysample)

        if display_pareto_front:
            plt.scatter(ysample[5:,0], ysample[5:,1], color="red", label="Samples.")
            plt.scatter(ysample[0:n_init_samples,0], ysample[0:n_init_samples,1], color="blue", label="Initial samples.")
            plt.scatter(pf_approx[:,0], pf_approx[:,1], color="green", label="PF approximation.")
            plt.scatter(ysample[-1:-5:-1,0], ysample[-1:-5:-1,1], color="black", label="Last 5 samples.")
            plt.xlabel(r"$f_1(x)$")
            plt.ylabel(r"$f_2(x)$")
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

    Param:
        test_problem: problem to be solved. Defined via pymoo.
    ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
            in the objective space.
    max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
    """
    def __init__(self, test_problem, ideal_point, max_point):
        self.test_problem = test_problem
        # self.aggregation_func = aggregation_func
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.upper = test_problem.xu
        self.lower = test_problem.xl


    def _objective_function(self, problem, x):
        return problem.evaluate(x)

    def _expected_improvement(self, X, model, opt_value, kappa=0.01):
        """
        EI, single objective acquisition function.

        Returns:
            EI: The Expected improvement of X over the opt_value given the information
                from the model.
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

        Params:
            function: The acquisition function to be optimised.
            models: The model trained on the aggregated values.
            current_best: Best/most optimal solution found thus far.

        Returns:
            res.x: solution of the optimsiation.
            res.fun: function value of the optimisation.
        """

        def obj(X):
            # print("obj called")
            return -function(X, models, current_best)

        x = list(zip(self.test_problem.xl, self.test_problem.xu))

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

            aggregation_func: the aggregation function used to aggregate the objective vectors in a single scalar value. 
            Scalarisations are used. Options Include: ExponentialWeightedCriterion, IPBI, PBI, Tchebicheff, WeightedNorm, WeightedPower, WeightedProduct, AugmentedTchebicheff, ModifiedTchebicheff
            Imported from scalarisations.py

        
        Returns: Four lists in a tuple.
            pf_approx, solution vectors on the pareto front approximation found by the optimiser.
            pf_inputs, corresponding inputs to the values in pf_approx.
            ysample, output objective vectors of all evaluated samples.
            Xsample, all samples that were evaluated.
        """
        
        # ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        problem = self.test_problem
        assert(problem.n_obj==2)

        problem.n_obj = 2

        # 1/n_obj * 
        # initial weights are all the same
        weights = np.asarray([1/problem.n_obj]*problem.n_obj)

        # get the initial samples used to build first model
        # use latin hypercube sampling
        sampler = qmc.LatinHypercube(d=problem.n_var)
        Xsample = sampler.random(n=n_init_samples)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])

        aggregated_samples = np.asarray([aggregation_func(i, weights) for i in ysample]).flatten()

        model = GaussianProcessRegressor()

        # Fit initial model.
        model.fit(Xsample, aggregated_samples)

        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=100)

        for i in range(n_iterations):

            # Identify the current best sample, used for search.
            current_best = aggregated_samples[np.argmin(aggregated_samples)]

            model.fit(Xsample, aggregated_samples)

            # use the model, current best to get the next x value to evaluate.
            next_X, _ = self._get_proposed(self._expected_improvement, model, current_best)

            # Evaluate that point to get its objective values.
            next_y = self._objective_function(problem, next_X)

            # add the new sample
            ysample = np.vstack((ysample, next_y))

            ref_dir = ref_dirs[np.random.randint(0,len(ref_dirs))]
            # print("Selected weight: "+str(ref_dir))

            agg = aggregation_func(next_y, ref_dir)

            aggregated_samples = np.append(aggregated_samples, agg)

            # Add the variables into the archives.
            Xsample = np.vstack((Xsample, next_X))
        
        pf_approx = util_functions.calc_pf(ysample)

        if display_pareto_front:
            plt.scatter(ysample[5:,0], ysample[5:,1], color="red", label="Samples.")
            plt.scatter(ysample[0:n_init_samples,0], ysample[0:n_init_samples,1], color="blue", label="Initial samples.")
            plt.scatter(pf_approx[:,0], pf_approx[:,1], color="green", label="PF approximation.")
            plt.scatter(ysample[-1:-5:-1,0], ysample[-1:-5:-1,1], color="black", label="Last 5 samples.", zorder=10)
            plt.xlabel(r"$f_1(x)$")
            plt.ylabel(r"$f_2(x)$")
            plt.legend()
            plt.show()

        # Find the inputs that correspond to the pareto front.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        return pf_approx, pf_inputs, ysample, Xsample


