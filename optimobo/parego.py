import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import differential_evolution
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
# from pymoo.core.problem import ElementwiseProblem


# from util_functions import EHVI, calc_pf, expected_decomposition
# from . import util_functions
import util_functions
import result
import GPy

class ParEGO():

    def __init__(self, test_problem, ideal_point, max_point):
        self.test_problem = test_problem
        # self.aggregation_func = aggregation_func
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.n_obj = test_problem.n_obj
        self.upper = test_problem.xu
        self.lower = test_problem.xl

    
    def mutate(self, solution_vector, mutation_rate, mutation_strength):
        mutant = solution_vector.copy()
        for i in range(len(mutant)):
            if np.random.rand() < mutation_rate:
                mutant[i] += np.random.normal(0, mutation_strength)
        return mutant

    def generate_random_solution(self, vector_length):
        return np.random.uniform(low=self.lower, high=self.upper, size=vector_length)

    def _objective_function(self, problem, x):
        """
        Wrapper for the objective function, makes my code clearer.
        Returns objective values.

        Params:
            problem: Problem object
            x: input 
        """
        return problem.evaluate(x)

    def _expected_improvement(self, X, model, opt_value, kappa=0.01):
        """
        EI, single objective acquisition function.

        Returns:
            EI: The Expected improvement of X over the opt_value given the information
                from the model.
        """

        # get the mean and s.d. of the proposed point
        X_aux = X.reshape(1, -1)
        mu_x, sigma_x = model.predict(X_aux)
        # is the variance therefore we need the square root it
        sigma_x = np.sqrt(sigma_x)
        mu_x = mu_x[0]
        sigma_x = sigma_x[0]

        gamma_x = (opt_value - mu_x) / (sigma_x + 1e-10)
        ei = sigma_x * (gamma_x * norm.cdf(gamma_x) + norm.pdf(gamma_x))
        return ei.flatten() 


    def solve(self, n_iterations=100, n_init_samples=5, aggregation_func=None):


        problem = self.test_problem

        # 1/n_obj * 
        # initial weights are all the same
        weights = np.asarray([1/problem.n_obj]*problem.n_obj)

        # get the initial samples used to build first model
        # use latin hypercube sampling
        variable_ranges = list(zip(self.test_problem.xl, self.test_problem.xu))
        Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples*20, variable_ranges)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])
        aggregated_samples = np.asarray([aggregation_func(i, weights) for i in ysample]).flatten()
        ys= np.reshape(aggregated_samples, (-1,1))


        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=100)
        hypervolume_convergence = []

        for i in range(n_iterations):

            # Hypervolume performance.
            ref_point = self.max_point
            HV_ind = HV(ref_point=ref_point)
            hv = HV_ind(ysample)
            hypervolume_convergence.append(hv)
            
            
            # select a radnom weight vector
            ref_dir = ref_dirs[np.random.randint(0,len(ref_dirs))]


            # DACE procedure
            # compute aggregation function:
            agg = aggregation_func(next_y, ref_dir)
            
            # build model using scalarisations, mono-surrogate
            model = GPy.models.GPRegression(Xsample, np.reshape(aggregated_samples, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
            model.Gaussian_noise.variance.fix(0)
            model.optimize(messages=False,max_f_eval=1000)

            # now before we use EI we use evolutionary operators, EI is used in the evoalg
            # EVO_ALG(model, xpop[])
            # this gives us a newly proposed 

            # initialise a temporary population of solution vectors.
            mutant_population = [self.mutate(solution_vector, mutation_rate, mutation_strength) for solution_vector in Xsample]
            num_random_solutions = 10
            random_population = [self.generate_random_solution(len(Xsample[0])) for _ in range(num_random_solutions)]

            temporary_population = mutant_population + random_population

            n_remutations = 10
            # best soluton
            best_solution_found = None
            best_EI = 0
            for i in range(n_remutations):


                current_best = aggregated_samples[np.argmin(aggregated_samples)]
                # find EI of the all points in the population, get the best one
                EIs = [self._expected_improvement(i, model, current_best ) for i in temporary_population]
                best_EI_in_pop = np.max(EIs)
                best_in_current_pop = temporary_population[np.argmax(EIs)]
                if best_EI_in_pop > best_EI:
                    best_EI = best_EI_in_pop
                    best_solution_found = best_in_current_pop

                # 
                # @TODO
                # select, recombine, mutate to form new pop
                # 
            





                temporary_population = new_pop

            next_X = best_solution_found
            # Evaluate that point to get its objective values.
            next_y = self._objective_function(problem, next_X)

            # add the new sample to archive.
            ysample = np.vstack((ysample, next_y))

            ref_dir = ref_dirs[np.random.randint(0,len(ref_dirs))]

            # Aggregate new sample
            agg = aggregation_func(next_y, ref_dir)

            # Update archive.
            aggregated_samples = np.append(aggregated_samples, agg)          
            Xsample = np.vstack((Xsample, next_X))
        
        pf_approx = util_functions.calc_pf(ysample)

        # Find the inputs that correspond to the pareto front.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

        return res





        

