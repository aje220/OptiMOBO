import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import qmc
from scipy.stats import norm
from scipy.optimize import differential_evolution
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
import random
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


    # def binary_tournament_selection_without_replacment(self, population, num_selections, model, opt_value):

    #     pop = population
    #     all_parents = []
    #     for i in range(num_selections):
    #         # print(len(pop))
    #         parents_pair = [0,0]
    #         for i in range(2):
    #             if len(pop) < 3:
    #                 all_parents.append(pop)
    #                 return all_parents
    #             # import pdb; pdb.set_trace()
    #             idx = random.sample(range(1, len(pop)), 2)
    #             ind1 = idx[0]
    #             ind2 = idx[1]
    #             selected = None
    #             ind1_fitness = self._expected_improvement(pop[ind1], model, opt_value)
    #             ind2_fitness = self._expected_improvement(pop[ind2], model, opt_value)

    #             if ind1_fitness > ind2_fitness:
    #                 selected = ind1
    #             else:
    #                 selected = ind2

    #             parent1 = pop[selected]
    #             parents_pair[i] = parent1
    #             pop = np.delete(pop, ind1, 0)
    #             if len(pop) == 1 :
    #                 return all_parents
    #                 # import pdb; pdb.set_trace()
    #         all_parents.append(parents_pair)
    #     return all_parents

    def simulated_binary_crossover(self, parent1, parent2, eta=1, crossover_prob=0.2):
        
        if np.random.rand() > crossover_prob:
            return parent1.copy()
        
        u = np.random.rand(parent1.shape[0])
        beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta + 1)), (1.0 / (2 - 2 * u)) ** (1.0 / (eta + 1)))
        
        offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

        offspring1 = np.clip(offspring1, self.lower, self.upper)
        offspring2 = np.clip(offspring2, self.lower, self.upper)

        # import pdb; pdb.set_trace()

        # ParEGO algorithm specifies that only one offspring is used after crossover.
        return offspring1
    
    def parego_binary_tournament_selection_without_replacment(self, population, model, opt_value):
    
        pop = population
        parents_pair = [0,0]

        parent1_index = None

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
            
            if i == 0:
                parent1_index = selected

            # import pdb; pdb.set_trace()
            parents_pair[i] = pop[selected]
            pop = np.delete(pop, selected, 0)
        return parents_pair, parent1_index

    
    def binary_tournament_selection(self, population, num_selections, model, opt_value):
        selected_indices = []
    
        for _ in range(num_selections):
            candidate_indices = random.sample(range(len(population)), 2)
            selected_index = candidate_indices[0] if self._expected_improvement(population[candidate_indices[0]], model, opt_value) > self._expected_improvement(population[candidate_indices[1]], model, opt_value) else candidate_indices[1]
            selected_indices.append(selected_index)
        
        selected_individuals = [population[index] for index in selected_indices]
        import pdb; pdb.set_trace()
        return selected_individuals

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
        # weights = np.asarray([1/problem.n_obj]*problem.n_obj)

        # get the initial samples used to build first model
        # use latin hypercube sampling
        variable_ranges = list(zip(self.test_problem.xl, self.test_problem.xu))
        Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples, variable_ranges)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])
        
        print(self.upper)
        print(self.lower)

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
            aggregated_samples = np.asarray([aggregation_func(i, ref_dir) for i in ysample]).flatten()
            ys= np.reshape(aggregated_samples, (-1,1))

            # DACE procedure
            # compute aggregation function:
            # agg = aggregation_func(next_y, ref_dir)
            
            # build model using scalarisations, mono-surrogate
            model = GPy.models.GPRegression(Xsample, np.reshape(aggregated_samples, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
            model.Gaussian_noise.variance.fix(0)
            model.optimize(messages=False,max_f_eval=1000)

            # now before we use EI we use evolutionary operators, EI is used in the evoalg
            # EVO_ALG(model, xpop[])
            # this gives us a newly proposed 
            population_size = len(Xsample)            
            mutation_rate = 1/self.n_vars
            eta = 2.0


            # initialise a temporary population of solution vectors.
            idx = random.sample(range(0,len(Xsample)), 10)
            # import pdb; pdb.set_trace()
            mutant_population = [self.mutate(solution_vector, mutation_rate) for solution_vector in Xsample[idx]]
            # num_random_solutions = population_size
            # random_population = [self.generate_random_solution(len(Xsample[0])) for _ in range(num_random_solutions*2)]
            random_population = util_functions.generate_latin_hypercube_samples(10, list(zip(self.test_problem.xl, self.test_problem.xu)))

            temporary_population = np.vstack((mutant_population,  random_population))


            # get the best currently found scalarised value. This is to calculate expected improvement.
            current_best = aggregated_samples[np.argmin(aggregated_samples)]


            n_remutations = 1000
            best_solution_found = self.lower
            best_EI = 0
            for i in range(n_remutations):


                # find EI of the all points in the population, get the best one
                EIs = [self._expected_improvement(i, model, current_best ) for i in temporary_population]
                best_EI_in_pop = np.max(EIs)
                best_in_current_pop = temporary_population[np.argmax(EIs)]
                if best_EI_in_pop > best_EI:
                    best_EI = best_EI_in_pop
                    best_solution_found = best_in_current_pop



                # Get parents for reombination, we need the index of parent1 to check if it needs to be replaced.
                selected, parent1_idx = self.parego_binary_tournament_selection_without_replacment(temporary_population, model, current_best)

                parent1 = selected[0]
                parent2 = selected[1]


                offspring = self.simulated_binary_crossover(parent1, parent2, eta, crossover_prob=0.2)

                # now apply mutation
                mutated_offspring = self.mutate(offspring, mutation_rate)

                # to check if mutant is better than parent1
                fitness_parent1 = self._expected_improvement(parent1, model, current_best)
                fitness_offspring = self._expected_improvement(mutated_offspring, model, current_best)

                # check if the new one is better than the parent
                final = parent1 if fitness_parent1 > fitness_offspring else mutated_offspring


                temporary_population[parent1_idx] = final


            print("NEXT_X FOUND")
            next_X = best_solution_found
            # Evaluate that point to get its objective valuess
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





        

