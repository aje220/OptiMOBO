import random
import numpy as np
from scipy.stats import norm
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from scipy.optimize import differential_evolution
import GPy


# from pymoo.core.problem import ElementwiseProblem


# from util_functions import EHVI, calc_pf, expected_decomposition
# from . import util_functions
import util_functions
import result

class cParEGO():
    """
    Proposed by J Knowles in 2006. Its a mono-surrogate algorithm that uses evolutionary operators to select the next sample point
    at each iteration. DOI:10.1109/TEVC.2005.851274
    
    """

    def __init__(self, test_problem, ideal_point, max_point):
        self.test_problem = test_problem
        # self.aggregation_func = aggregation_func
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.n_obj = test_problem.n_obj
        self.upper = test_problem.xu
        self.lower = test_problem.xl


    def mutate(self, solution_vector, mutation_rate):
        """
        Mutate a solution. This is not the same method of mutation discussed in the original paper, and forgive my ignorance,
        I dont know what they are suggesting. But this method works nonetheless. 
        """

        mutant = solution_vector.copy()

        for i, value in enumerate(mutant):
            if np.random.rand() < mutation_rate:
                # mu = np.random.uniform(0.0001, 1.000)
                # delta = 1/(100+mu)
                if np.random.uniform() > 0.5:
                    mutant[i] = mutant[i]*1.05
                else:
                    mutant[i] = mutant[i]*0.95
        
        # Prevent a mutation from exceeding the bounds of the decision variables.
        mutant = np.clip(mutant, self.lower, self.upper)
        return mutant

    def simulated_binary_crossover(self, parent1, parent2, eta=1, crossover_prob=0.2):
        
        if np.random.rand() > crossover_prob:
            return parent1.copy()
        
        u = np.random.rand(parent1.shape[0])
        beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta + 1)), (1.0 / (2 - 2 * u)) ** (1.0 / (eta + 1)))
        
        offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

        # Prevent mating from exceeding the bounds of the decision variables.
        offspring1 = np.clip(offspring1, self.lower, self.upper)
        # offspring2 = np.clip(offspring2, self.lower, self.upper)

        # ParEGO algorithm specifies that only one offspring is used after crossover.
        return offspring1

    
    def parego_binary_tournament_selection_without_replacment(self, population, model, opt_value):
        """
        Binary tournament function selection modified to fit to ParEGO specifications.
        It only runs the tournament twice, for two parents.
        """
    
        pop = population
        parents_pair = [0,0]
        parent1_index = None

        # We only need two parents.
        for i in range(2):
            # import pdb; pdb.set_trace()
            idx = random.sample(range(1, len(pop)), 2)
            ind1 = idx[0]
            ind2 = idx[1]
            selected = None
            ind1_fitness = self._expected_improvement(pop[ind1], model, opt_value)
            ind2_fitness = self._expected_improvement(pop[ind2], model, opt_value)


            if ind1_fitness > ind2_fitness:
                selected = ind1
            else:
                selected = ind2
            
            # We need to keep the index of the selected solution from the population.
            # This is so we can easily compare the offspring later.
            if i == 0:
                parent1_index = selected

            # import pdb; pdb.set_trace()
            parents_pair[i] = pop[selected]
            pop = np.delete(pop, selected, 0)
        return parents_pair, parent1_index

    
    def _objective_function(self, problem, x):
        """
        Wrapper for the objective function, makes my code clearer.
        Returns objective values.

        Params:
            problem: Problem object
            x: input 
        """
        return problem.evaluate(x)

    
    def _constraint_function(self, problem, x):
        """
        Wrapper for the constraint function, makes my code clearer.
        Returns objective values.

        Params:
            problem: Problem object
            x: input 
        """
        return problem.evaluate_constraints(x)

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


    def solve(self, n_iterations=100, n_init_samples=5, aggregation_func=None):
        """
        Main flow for the algorithm. Call this to solve the specified problem.
        """

        problem = self.test_problem

        # Initial Latin Hypercube samples.
        # The initialisation of samples used here isnt quite the same as the paper, but im sure its fine.
        variable_ranges = list(zip(self.test_problem.xl, self.test_problem.xu))
        Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples, variable_ranges)

        # Evaluate initial samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])

        # Evaluate initial constraints
        gsample = np.asarray([self._constraint_function(problem, x) for x in Xsample])

        
        # Weights that will be used in the aggregation of objective values.
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=10)

        # This will be filled at each iteration
        hypervolume_convergence = []

        for i in range(n_iterations):


            # Hypervolume performance.
            ref_point = self.max_point
            HV_ind = HV(ref_point=ref_point)
            hv = HV_ind(ysample)
            hypervolume_convergence.append(hv)

            # shuffle ref_dirs
            np.random.shuffle(ref_dirs)



            for ref_dir in ref_dirs:

                # update the lower and upper bounds
                # these are used as the ideal and max points
                upper = np.zeros(len(self.n_obj))
                lower = np.zeros(len(self.n_obj))
                for i in range(self.n_obj):
                    upper[i] = max(ysample[:,i])
                    lower[i] = min(ysample[:,i])

                # change the bounds of the scalarisation object
                aggregation_func.set_bounds(lower, upper)

                # calculate scalar fitness score
                aggregated_samples = np.asarray([aggregation_func(i, ref_dir) for i in ysample]).flatten()
                ys= np.reshape(aggregated_samples, (-1,1))


                # penalise infeasible solutions according to their degree of constraint violation,
                # this uses scalarised values
                


                # select a subset of solutions from Xsample with maximum size Nmax




                # fit a model using the penalised aggragated samples and the subset of inputs.
                model = GPy.models.GPRegression(Xsample, np.reshape(aggregated_samples, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                model.Gaussian_noise.variance.fix(0)
                model.optimize(messages=False,max_f_eval=1000)

                

                current_best = np.min(aggregated_samples)
                next_X = self._get_proposed(self._expected_improvement, model, current_best)


                # Evaluate that point to get its objective valuess
                next_y = self._objective_function(problem, next_X)

                # add the new sample to archive.
                ysample = np.vstack((ysample, next_y))

                # Aggregate new sample
                agg = aggregation_func(next_y, ref_dir)

                # Update archive.
                aggregated_samples = np.append(aggregated_samples, agg)          
                Xsample = np.vstack((Xsample, next_X))

                # add constraints to constraint archive
                next_g = self._constraint_function(problem, next_X)
                gsample = np.vstack((gsample, next_g))

                
        
        pf_approx = util_functions.calc_pf(ysample)

        # Find the inputs that correspond to the pareto front.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

        return res
