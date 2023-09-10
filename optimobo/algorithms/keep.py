import numpy as np
from scipy.stats import norm
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
import random
# from pymoo.core.problem import ElementwiseProblem


# from util_functions import EHVI, calc_pf, expected_decomposition
# from . import util_functions
import util_functions
import result
import GPy

class KEEP():
    """
    Presented in 2017 by Davins-Valldaura et al.. This algorithm extends ParEGO, and is therefore very similar. 
    It adds a second kriging model that predicts the probability a solution is a member of the pareto set and uses this to enhance
    convergence.
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

        mutant = solution_vector.copy()

        for i, value in enumerate(mutant):
            if np.random.rand() < mutation_rate:
                # mu = np.random.uniform(0.0001, 1.000)
                # delta = 1/(100+mu)
                if np.random.uniform() > 0.5:
                    mutant[i] = mutant[i]*1.05
                else:
                    mutant[i] = mutant[i]*0.95
        
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

    
    def KEEP_binary_tournament_selection_without_replacment(self, population, pareto_model, scalar_model, opt_value):
        """
        As we use the same evolutionary algorithm as ParEGO, only a small modification is needed for this selection
        function. The change is the fitness function.
        """
    
        pop = population
        parents_pair = [0,0]

        parent1_index = None

        for i in range(2):
            # import pdb; pdb.set_trace()
            idx = random.sample(range(1, len(pop)), 2)
            ind1 = idx[0]
            ind2 = idx[1]
            selected = None
            ind1_fitness = self.pareto_expected_improvement(pop[ind1], pareto_model, scalar_model, opt_value )
            ind2_fitness = self.pareto_expected_improvement(pop[ind2], pareto_model, scalar_model, opt_value )


            if ind1_fitness > ind2_fitness:
                selected = ind1
            else:
                selected = ind2
            
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


    


    def pareto_expected_improvement(self, X, pareto_model, scalarised_model, opt_value):
        """
        Performance function specific to KEEP. This takes into acount the Expected Improvement
        of a point and the probability it is a member of the Pareto set to evaluate fitness.
        As such, it requires two models.
        """
        EI = self._expected_improvement(X, scalarised_model, opt_value)
        pareto_pred = pareto_model.predict(np.asarray([X]))
        # import pdb; pdb.set_trace()
        # return np.dot(pareto_pred, EI)
        return pareto_pred[0][0]*EI


    def solve(self, n_iterations=100, n_init_samples=5, aggregation_func=None):
        """
        Main flow for the algorithm. Call this to solve the specified problem.
        """

        problem = self.test_problem
        # Initial Latin Hypercube samples.
        # The initialisation of samples used here isnt quite the same as the paper, but im sure its fine.
        variable_ranges = list(zip(self.test_problem.xl, self.test_problem.xu))
        Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples, variable_ranges)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])

        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=100)
        hypervolume_convergence = []

        for i in range(n_iterations):

            # Hypervolume performance.
            ref_point = self.max_point
            HV_ind = HV(ref_point=ref_point)
            hv = HV_ind(ysample)
            hypervolume_convergence.append(hv)
            
            
            # select a radnom weight vector, randomness promotes diversity.
            ref_dir = ref_dirs[np.random.randint(0,len(ref_dirs))]
            aggregated_samples = np.asarray([aggregation_func(i, ref_dir) for i in ysample]).flatten()
            ys= np.reshape(aggregated_samples, (-1,1))

            
            # build model using scalarisations, mono-surrogate
            scalar_model = GPy.models.GPRegression(Xsample, np.reshape(aggregated_samples, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
            scalar_model.Gaussian_noise.variance.fix(0)
            scalar_model.optimize(messages=False,max_f_eval=1000)

            
            # Now identify the probabilities of indivuals belonging to the pareto approximation.abs
            pareto_set = util_functions.calc_pf(ysample)
            probs = np.zeros(len(Xsample))
            for i, value in enumerate(Xsample):
                if ysample[i] in pareto_set:
                    probs[i] = 1
                else:
                    pass
            
            # Now train a new kriging based on the probabilities
            pareto_model = GPy.models.GPRegression(Xsample, np.reshape(probs, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
            pareto_model.Gaussian_noise.variance.fix(0)
            pareto_model.optimize(messages=False,max_f_eval=1000)

            # now before we use EI we use evolutionary operators, EI is used in the evoalg
            # EVO_ALG(model, xpop[]) 
            population_size = len(Xsample)            
            mutation_rate = 1/self.n_vars
            eta = 2.0

            # initialise a temporary population of solution vectors. This process is left over from ParEGO
            idx = random.sample(range(0,len(Xsample)), 10)
            # This temporary pop is partly made from mutants of the current population.
            mutant_population = [self.mutate(solution_vector, mutation_rate) for solution_vector in Xsample[idx]]
            # and also random solutions. This temp pop is used in the evolutionary algorithm.
            random_population = util_functions.generate_latin_hypercube_samples(10, list(zip(self.test_problem.xl, self.test_problem.xu)))

            temporary_population = np.vstack((mutant_population,  random_population))

            # Get the best currently found scalarised value. This is to calculate expected improvement.
            current_best = aggregated_samples[np.argmin(aggregated_samples)]

            n_remutations = 1000 # because the chance of something changing is low, lots of iterations are needed.
            best_solution_found = self.lower
            best_PEI = 0
            for i in range(n_remutations):

                # Find EI of the all points in the population, get the best one. Were gonna be trying to improve the best
                # via evolutionary operators.
                # EIs = [self._expected_improvement(i, model, current_best ) for i in temporary_population]
                PEIs = [self.pareto_expected_improvement(i, pareto_model, scalar_model, current_best ) for i in temporary_population]

                best_PEI_in_pop = np.max(PEIs)
                best_in_current_pop = temporary_population[np.argmax(PEIs)]
                if best_PEI_in_pop > best_PEI:
                    best_PEI = best_PEI_in_pop
                    best_solution_found = best_in_current_pop


                # Get parents for reombination, we need the index of parent1 to check if it needs to be replaced.
                # selected, parent1_idx = self.parego_binary_tournament_selection_without_replacment(temporary_population, model, current_best)
                selected, parent1_idx = self.KEEP_binary_tournament_selection_without_replacment(temporary_population, pareto_model, scalar_model, current_best)
                parent1 = selected[0]
                parent2 = selected[1]

                # Mating
                offspring = self.simulated_binary_crossover(parent1, parent2, eta, crossover_prob=0.2)

                # Now apply mutation
                mutated_offspring = self.mutate(offspring, mutation_rate)

                # We need to check is the child outperforms parent1.
                fitness_parent1 = self.pareto_expected_improvement(parent1, pareto_model, scalar_model, current_best)
                fitness_offspring = self.pareto_expected_improvement(mutated_offspring, pareto_model, scalar_model, current_best)

                # Check if the new one is better than the parent, if so replace it in the temporary population.
                final = parent1 if fitness_parent1 > fitness_offspring else mutated_offspring
                temporary_population[parent1_idx] = final


            next_X = best_solution_found
            # Evaluate that point to get its objective valuess
            next_y = self._objective_function(problem, next_X)

            # add the new sample to archive.
            ysample = np.vstack((ysample, next_y))

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
