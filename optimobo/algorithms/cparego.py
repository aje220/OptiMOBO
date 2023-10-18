import random
import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
import GPy

import optimobo.util_functions as util_functions
import optimobo.result as result

class ParEGO_C1():
    """
    J A. Duro et al. 2022. https://doi.org/10.1016/j.ejor.2022.08.032
    This algorithm is ParEGO-C1    
    """

    def __init__(self, test_problem, ideal_point, max_point):
        self.test_problem = test_problem
        self.aggregation_func = None
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.n_obj = test_problem.n_obj
        self.upper = test_problem.xu
        self.lower = test_problem.xl

    
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


    def select_subset(self, X_feasible, X_infeasible, ref_dir, N_max):
        """
        returns subset X_prime of size N_max. We do this to minimise the cost of reconstructing the
        Gaussian process at each iteration.
        """

        def xi(x):
            """
            Get the infeasibility score of a potential solution
            TODO: change the constrain function call the information we have calculated already.
            """
            constr = self._constraint_function(self.test_problem, x) # really, we shouldnt call this, need to change this.
            xi = sum([max(i, 0) for i in constr])
            return xi

        def best_performing(X, N, ref_dir):
            X_sorted = X[X[:,-1].argsort()]
            # fitness_scores = self.aggregation_func(X[:,:-1], ref_dir)
            # fitness_scores.sort()
            X_prime = X_sorted[0:(N//2)]
            X_prime_squared = X_sorted[(N//2):]


            # compute the euclidean norm of the remaining solutions in the objective space
            deltas = [np.linalg.norm((x[self.n_vars:-1]-ref_dir)) for x in X_prime_squared]

            # append the deltas to the end of each solution
            aux = np.hstack((X_prime_squared, np.reshape(deltas, (-1, 1))))

            # sort by the distances
            deltas_sorted = aux[aux[:,-1].argsort()]

            # select the best (smallest) distances
            X_prime_cubed = deltas_sorted[0:(N-N//2)]

            X_prime = np.vstack((X_prime, X_prime_cubed[:,:-1]))
            return X_prime

        scores = [xi(x[:-1-self.n_obj]) for x in X_infeasible]
        X_infease_score = np.hstack((X_infeasible, np.reshape(scores, (-1,1))))
        X_infease_scores_sorted = X_infease_score[X_infease_score[:,-1].argsort()]


        H = N_max//2

        if len(X_feasible) + len(X_infeasible) < N_max: # all solutions are selected
            return np.vstack((X_feasible, X_infeasible))

        elif len(X_infeasible) == 0:
            return best_performing(X_feasible, N_max, ref_dir)

        elif len(X_feasible) == 0:
            X_prime = X_infease_scores_sorted[0:H]
            X_prime = X_prime[:,:-1]
            
            rows1 = [tuple(row) for row in X_prime]

            rows2 = [tuple(row) for row in X_infeasible]

            # Find the set difference of rows
            difference = list(set(rows2) - set(rows1))
            difference = np.array(difference)

            best_leftover = best_performing(difference, N_max-len(X_prime), ref_dir)
            X_prime = np.vstack((X_prime ,best_leftover ))
            return X_prime

        elif len(X_infeasible) >= H and len(X_feasible) >= H:
            X_prime = best_performing(X_feasible, H, ref_dir)
            X_prime_squared = X_infease_scores_sorted[0:(N_max - len(X_prime))]
            X_prime_squared = X_prime_squared[:,:-1] # as it has feasibility scores on the end we take it off

            # if there are no feasible solutions the stack fails so ive included an if statement
            if len(X_prime) == 0:
                return X_prime_squared
            else:
                return np.vstack((X_prime, X_prime_squared))

        elif len(X_infeasible) < H and len(X_feasible) >= H:
            X_prime = X_infeasible
            the_other = best_performing(X_feasible, N_max - len(X_prime), ref_dir)
            return(np.vstack((X_prime, the_other)))

        elif len(X_infeasible) >= H and len(X_feasible) < H:
            X_prime = X_feasible
            X_prime_squared = X_infease_scores_sorted[0:(N_max - len(X_prime))]
            X_prime_squared = X_prime_squared[:,:-1]
            return np.vstack((X_prime, X_prime_squared))
        
        else:
            # should never reach this point
            return np.vstack((X_infeasible, X_feasible))



    def solve(self, aggregation_func, budget=50, n_init_samples=5, N_max=100):
        """
        Main flow for the algorithm. Call this to solve the specified problem.
        For each iteration, 10 sample points are discovered and evaluated, this means that less iterations are
        needed for convergence.

        budget: The termination condition for the algorithm.
        n_init_samples: The number of initial samples.
        aggregation_func: The aggregation function used to scalarise the multiple objective values.
        N_max: the maximum number of data points used to construct the gaussian process model.
            By limiting the reconstruction of the model to only the best N_max points we can improve
            the speed of the algorithm. Useless datapoints dont have to be included.

        """
        self.aggregation_func = aggregation_func
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

        n_iters = budget//10

        for i in range(n_iters):


            # Hypervolume performance.
            ref_point = self.max_point
            HV_ind = HV(ref_point=ref_point)
            hv = HV_ind(ysample)
            hypervolume_convergence.append(hv)

            # shuffle ref_dirs, for diversity's sake
            np.random.shuffle(ref_dirs)


            for ref_dir in ref_dirs:
                print("Iteration with reference direction: "+str(ref_dir))

                # update the lower and upper bounds
                # these are used as the ideal and max points
                upper = np.zeros(self.n_obj)
                lower = np.zeros(self.n_obj)
                for i in range(self.n_obj):
                    upper[i] = max(ysample[:,i])
                    lower[i] = min(ysample[:,i])

                # change the bounds of the scalarisation object
                aggregation_func.set_bounds(lower, upper)

                # calculate scalar fitness score
                aggregated_samples = np.asarray([aggregation_func(i, ref_dir) for i in ysample]).flatten()
                best_index = np.argmin(aggregated_samples)

                current_best = aggregated_samples[best_index]
                current_best_X = Xsample[best_index]

                # identify X_feasible and X_infeasible
                infease = gsample > 0
                feasible_mask = np.zeros(len(infease), dtype=bool)
                for i, value in enumerate(infease):
                    if np.any(value):
                        feasible_mask[i] = True 
                X_feasible = Xsample[~feasible_mask]
                X_infeasible = Xsample[feasible_mask]

                # So we need scalarisation scores for the penalisation, so we should join them together
                S_feasible = aggregated_samples[~feasible_mask]
                S_infeasible = aggregated_samples[feasible_mask]
                y_feasible = ysample[~feasible_mask]
                y_infeasible = ysample[feasible_mask]

                # stack information toegther so they can be found together
                feasible_pairs = np.hstack((X_feasible, y_feasible, np.reshape(S_feasible, (-1,1))))
                infeasible_pairs = np.hstack((X_infeasible, y_infeasible, np.reshape(S_infeasible, (-1,1))))
               
                # Penalise the infeasible solutions, if they exist.
                if len(infeasible_pairs) > 0:

                    v_max = [max(idx) for idx in zip(*gsample)]

                    # these next four functions are written as they are shown in the paper
                    def xi_single(J, v_max):
                        """
                        J is a consraint vector, one entry from gsample
                        """
                        calc = sum([max(value, 0)/v_max[count] for count, value in enumerate(J)])
                        return calc/len(J)

                    def s_dot(x, x_star):
                        """
                        this function changes the scalarised fitness values of the infeasible solutions, it returns a new scalarised fitness value
                        so this updates the aggregated_samples?
                        """
                        s_fitness = x[-1]
                        s_star_fitness = x_star[-1]

                        if s_fitness < s_star_fitness:
                            return s_star_fitness
                        else:
                            return s_fitness
                    
                    def s_bar(x, upper, lower, ref_dir):
                        return (x[-1] - min(aggregated_samples))/(max(aggregated_samples)-min(aggregated_samples))

                    def xi_bar(x, infeasibility_scores):
                        # it fails if there is only one infeasible solution, therefore this if else statement to fix it.
                        if len(infeasibility_scores) == 1:
                            return (xi_single(x, v_max) - min(aggregated_samples))/(max(aggregated_samples)-min(aggregated_samples))
                        else:
                            return (xi_single(x, v_max) - min(infeasibility_scores)/(max(infeasibility_scores) - min(infeasibility_scores)))

                    def s_double_dot(x, infeasibility_scores, x_star, g_vector):
                        """
                        main function for computing the new fitness score for the infeasible solutions
                        """
                        numerator = np.exp(2*(s_bar(x, upper,lower, ref_dir)+xi_bar(g_vector,infeasibility_scores))-1)
                        denom = np.exp(2) -1
                        return s_dot(x, x_star) + (numerator/denom)

                    # if there are infeasible solutions, calculate the infeasibility scores
                    # we need the max violaion of each constraint for XI
                    # so this is find because you are passing contraint values to xi_single
                    workable = gsample[feasible_mask]
                    infeasibility_scores = [xi_single(x, v_max) for x in workable]
                    
                    # get x_star, 
                    x_star = None
                    if np.any(X_feasible): # if there is a feasible solution
                        x_star = feasible_pairs[np.argmin(feasible_pairs[:,-1])]
                    else:
                        # if there is no feasible solutions, all the solutions are infeasible (obviously)
                        # so you just pick the "best" feasibility score
                        x_star = infeasible_pairs[np.argmin(infeasibility_scores)]

                    penalised = np.asarray([s_double_dot(value, infeasibility_scores, x_star, workable[count]) for count, value in enumerate(infeasible_pairs)])

                    aggregated_samples[feasible_mask] = penalised.flatten()
                    
                # penalisation is over
                # now select a subset of solutions from X with maximum size N_max, which is a subset of all the solutions
                # you are picking a smaller set of the best solutions to reconstruct the model, this should improve performance

                # Stack everything together so when we are selecting a subset the input output pairs are kept together.
                S_feasible = aggregated_samples[~feasible_mask]
                S_infeasible = aggregated_samples[feasible_mask]
                y_feasible = ysample[~feasible_mask]
                y_infeasible = ysample[feasible_mask]

                feasible_pairs = np.hstack((X_feasible, y_feasible, np.reshape(S_feasible, (-1,1))))
                infeasible_pairs = np.hstack((X_infeasible, y_infeasible, np.reshape(S_infeasible, (-1,1))))

            
                X_prime = self.select_subset(feasible_pairs, infeasible_pairs, ref_dir, N_max)
            
            
                model_input = X_prime[:,:self.n_vars]
                model_output = X_prime[:,-1]
                # fit a model using the penalised aggragated samples and the subset of inputs.
                model = GPy.models.GPRegression(model_input, np.reshape(model_output, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                model.Gaussian_noise.variance.fix(0)
                model.optimize(messages=False,max_f_eval=1000)

                next_X, _ = self._get_proposed(self._expected_improvement, model, current_best)

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

                
        # the pf_approx is the pf of the feasible solutions
        pf_approx = util_functions.calc_pf(y_feasible)
        # Find the inputs that correspond to the pareto front.
        indicies = []
        for i, item in enumerate(y_feasible):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = y_feasible[indicies]

        res = result.Constrained_Res(y_infeasible, y_feasible, X_infeasible, X_feasible, pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

        return res


class ParEGO_C2():
    """
    J A. Duro et al. 2022. https://doi.org/10.1016/j.ejor.2022.08.032
    This algorithm is ParEGO-C2
    """

    def __init__(self, test_problem, ideal_point, max_point):
        self.test_problem = test_problem
        self.aggregation_func = None
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.n_eq_constr = test_problem.n_eq_constr
        self.n_ieq_constr = test_problem.n_ieq_constr
        self.n_obj = test_problem.n_obj
        self.upper = test_problem.xu
        self.lower = test_problem.xl

    
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
        # is the variance therefore we need the square root it to get STD
        sigma_x = np.sqrt(sigma_x)
        mu_x = mu_x[0]
        sigma_x = sigma_x[0]

        gamma_x = (opt_value - mu_x) / (sigma_x + 1e-10)
        ei = sigma_x * (gamma_x * norm.cdf(gamma_x) + norm.pdf(gamma_x))
        return ei.flatten()
    
    def probability_of_feasibility(self, X, model):
        """
        probability of a decision vector being feasible
        X: decision vector/input variable
        model: the contraint model for a specific constraint.
        
        """
        X_aux = X.reshape(1, -1)
        mu_x, sigma_x = model.predict(X_aux)
        sigma_x = np.sqrt(sigma_x+1e-5) # this noise may need to be changed
        fraction = (0-mu_x)/sigma_x
        pof = norm.cdf(fraction)

        return pof

    def consraint_ei(self, X, aggregate_model, constraint_models, current_best):
        """
        This is the acquisition function used for finding the next X input into the expenive function.
        X: decision vector
        aggregate_model: the regression model trained on the currently explored inputs and the scalarised objective values.
        constraint_models: an array of the models trained on the contraint functions.
        current_best
        """
        ei = self._expected_improvement(X, aggregate_model, current_best)
        pof = np.prod([self.probability_of_feasibility(X, model) for model in constraint_models])
        return (ei*pof).flatten()

    def select_current_best(self, X_feasible, X_infeasible):
        """
        Returns the best scalarisation value. This function is written as to improve the 
        exploration of the acquisition function. 
        """
        if len(X_feasible) == 0: # if there are no feasible solutions
            # get the infeasibility scores of all the solutions
            scores = [self.xi(x[:self.n_vars]) for x in X_infeasible]
            X_infease_score = np.hstack((X_infeasible, np.reshape(scores, (-1,1))))
            idx = np.argmax(X_infease_score[:,-1])
            return X_infeasible[idx][-1] # minus one because we want the scalar value.
            
        else:
            idx = np.argmax(X_feasible[:,-1])
            return X_feasible[idx][-1]
    
    def xi(self, x):
        """
        Get the infeasibility score of a potential solution
        TODO: change the constrain function call the information we have calculated already.
        """
        constr = self._constraint_function(self.test_problem, x) # really, we shouldnt call this, need to change this.
        xi = sum([max(i, 0) for i in constr])
        return xi


    def _get_proposed(self, function, models, constraint_models, current_best):
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
            return -function(X, models, constraint_models, current_best)

        x = list(zip(self.test_problem.xl, self.test_problem.xu))

        res = differential_evolution(obj, x)
        return res.x, res.fun


    def select_subset(self, X_feasible, X_infeasible, ref_dir, N_max):
        """
        returns subset X_prime of size N_max. We do this to minimise the cost of reconstructing the
        Gaussian process at each iteration.
        """

        def xi(x):
            """
            Get the infeasibility score of a potential solution
            TODO: change the constrain function call the information we have calculated already.
            """
            constr = self._constraint_function(self.test_problem, x) # really, we shouldnt call this, need to change this.
            xi = sum([max(i, 0) for i in constr])
            return xi

        def best_performing(X, N, ref_dir):
            """
            Subroutine, defined in the paper. This selects the best solutions given their scalarisation values
            """
            X_sorted = X[X[:,-1].argsort()]
            X_prime = X_sorted[0:(N//2)]
            X_prime_squared = X_sorted[(N//2):]

            # compute the euclidean norm of the remaining solutions in the objective space
            n_constr = self.n_ieq_constr+self.n_eq_constr
            deltas = [np.linalg.norm((x[self.n_vars:-1-n_constr]-ref_dir)) for x in X_prime_squared]
            
            # append the deltas to the end of each solution
            aux = np.hstack((X_prime_squared, np.reshape(deltas, (-1, 1))))

            # sort by the distances
            deltas_sorted = aux[aux[:,-1].argsort()]

            # select the best (smallest) distances
            X_prime_cubed = deltas_sorted[0:(N-N//2)]

            X_prime = np.vstack((X_prime, X_prime_cubed[:,:-1]))
            return X_prime

        scores = [xi(x[:self.n_vars]) for x in X_infeasible]
        X_infease_score = np.hstack((X_infeasible, np.reshape(scores, (-1,1))))
        X_infease_scores_sorted = X_infease_score[X_infease_score[:,-1].argsort()]



        H = N_max//2

        if len(X_feasible) + len(X_infeasible) < N_max: # all solutions are selected
            return np.vstack((X_feasible, X_infeasible))

        elif len(X_infeasible) == 0:
            return best_performing(X_feasible, N_max, ref_dir)

        elif len(X_feasible) == 0:
            X_prime = X_infease_scores_sorted[0:H]
            X_prime = X_prime[:,:-1]
            # difference = np.setdiff1d(X_prime, X_infeasible)
            
            rows1 = [tuple(row) for row in X_prime]

            rows2 = [tuple(row) for row in X_infeasible]

            # Find the set difference of rows
            difference = list(set(rows2) - set(rows1))
            difference = np.array(difference)

            best_leftover = best_performing(difference, N_max-len(X_prime), ref_dir)
            X_prime = np.vstack((X_prime ,best_leftover ))
            return X_prime

        elif len(X_infeasible) >= H and len(X_feasible) >= H:
            X_prime = best_performing(X_feasible, H, ref_dir)
            X_prime_squared = X_infease_scores_sorted[0:(N_max - len(X_prime))]
            X_prime_squared = X_prime_squared[:,:-1] # as it has feasibility scores on the end we take it off

            # if there are no feasible solutions the stack fails so ive included an if statement

            if len(X_prime) == 0:
                return X_prime_squared
            else:
                return np.vstack((X_prime, X_prime_squared))

        elif len(X_infeasible) < H and len(X_feasible) >= H:
            X_prime = X_infeasible
            the_other = best_performing(X_feasible, N_max - len(X_prime), ref_dir)
            return(np.vstack((X_prime, the_other)))

        elif len(X_infeasible) >= H and len(X_feasible) < H:
            X_prime = X_feasible
            X_prime_squared = X_infease_scores_sorted[0:(N_max - len(X_prime))]
            X_prime_squared = X_prime_squared[:,:-1]
            return np.vstack((X_prime, X_prime_squared))
        
        else:
            # should never reach this point
            print("Reached the unreachable")
            return np.vstack((X_infeasible, X_feasible))



    def solve(self, aggregation_func, budget=10, n_init_samples=5, N_max=100):
        """
        Main flow for the algorithm. Call this to solve the specified problem.
        For each iteration, 10 sample points are discovered and evaluated, this means that less iterations are
        needed for convergence.

        budget: The termination condition for the algorithm.
        n_init_samples: The number of initial samples.
        aggregation_func: The aggregation function used to scalarise the multiple objective values.
        N_max: the maximum number of data points used to construct the gaussian process model.
            By limiting the reconstruction of the model to only the best N_max points we can improve
            the speed of the algorithm. Useless datapoints dont have to be included.

        """
        self.aggregation_func = aggregation_func
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

        n_iters = budget//10

        for i in range(n_iters):


            # Hypervolume performance.
            ref_point = self.max_point
            HV_ind = HV(ref_point=ref_point)
            hv = HV_ind(ysample)
            hypervolume_convergence.append(hv)

            # shuffle ref_dirs, for diversity's sake
            np.random.shuffle(ref_dirs)


            for ref_dir in ref_dirs:
                print("Iteration with reference direction: "+str(ref_dir))

                # update the lower and upper bounds
                # these are used as the ideal and max points
                upper = np.zeros(self.n_obj)
                lower = np.zeros(self.n_obj)
                for i in range(self.n_obj):
                    upper[i] = max(ysample[:,i])
                    lower[i] = min(ysample[:,i])

                # change the bounds of the scalarisation object
                aggregation_func.set_bounds(lower, upper)

                # calculate scalar fitness score
                aggregated_samples = np.asarray([aggregation_func(i, ref_dir) for i in ysample]).flatten()
   
                # identify X_feasible and X_infeasible
                infease = gsample > 0
                feasible_mask = np.zeros(len(infease), dtype=bool)
                for i, value in enumerate(infease):
                    if np.any(value):
                        feasible_mask[i] = True 
                X_feasible = Xsample[~feasible_mask]
                X_infeasible = Xsample[feasible_mask]

                # So we need scalarisation scores for the penalisation, so we should join them together
                S_feasible = aggregated_samples[~feasible_mask]
                S_infeasible = aggregated_samples[feasible_mask]
                y_feasible = ysample[~feasible_mask]
                y_infeasible = ysample[feasible_mask]

                # stack information toegther so they can be found together
                feasible_pairs = np.hstack((X_feasible, y_feasible, np.reshape(S_feasible, (-1,1))))
                infeasible_pairs = np.hstack((X_infeasible, y_infeasible, np.reshape(S_infeasible, (-1,1))))
               
                # Penalise the infeasible solutions, if they exist.
                if len(infeasible_pairs) > 0:

                    v_max = [max(idx) for idx in zip(*gsample)]

                    # these next four functions are written as they are shown in the paper
                    def xi_single(J, v_max):
                        """
                        J is a consraint vector, one entry from gsample
                        """
                        calc = sum([max(value, 0)/v_max[count] for count, value in enumerate(J)])
                        return calc/len(J)

                    def s_dot(x, x_star):
                        """
                        this function changes the scalarised fitness values of the infeasible solutions, it returns a new scalarised fitness value
                        so this updates the aggregated_samples?
                        """
                        s_fitness = x[-1]
                        s_star_fitness = x_star[-1]

                        if s_fitness < s_star_fitness:
                            return s_star_fitness
                        else:
                            return s_fitness
                    
                    def s_bar(x, upper, lower, ref_dir):
                        return (x[-1] - min(aggregated_samples))/(max(aggregated_samples)-min(aggregated_samples))

                    def xi_bar(x, infeasibility_scores):
                        # it fails if there is only one infeasible solution, therefore this if else statement to fix it.
                        if len(infeasibility_scores) == 1:
                            return (xi_single(x, v_max) - min(aggregated_samples))/(max(aggregated_samples)-min(aggregated_samples))
                        else:
                            return (xi_single(x, v_max) - min(infeasibility_scores)/(max(infeasibility_scores) - min(infeasibility_scores)))

                    def s_double_dot(x, infeasibility_scores, x_star, g_vector):
                        """
                        main function for computing the new fitness score for the infeasible solutions
                        """
                        numerator = np.exp(2*(s_bar(x, upper,lower, ref_dir)+xi_bar(g_vector,infeasibility_scores))-1)
                        denom = np.exp(2) -1
                        return s_dot(x, x_star) + (numerator/denom)

                    # if there are infeasible solutions, calculate the infeasibility scores
                    # we need the max violaion of each constraint for XI
     

                    # so this is find because you are passing contraint values to xi_single
                    workable = gsample[feasible_mask]
                    infeasibility_scores = [xi_single(x, v_max) for x in workable]
                    
                    # get x_star, 
                    x_star = None
                    if np.any(X_feasible): # if there is a feasible solution
                        x_star = feasible_pairs[np.argmin(feasible_pairs[:,-1])]
                    else:
                        # if there is no feasible solutions, all the solutions are infeasible (obviously)
                        # so you just pick the "best" feasibility score
                        x_star = infeasible_pairs[np.argmin(infeasibility_scores)]

                    penalised = np.asarray([s_double_dot(value, infeasibility_scores, x_star, workable[count]) for count, value in enumerate(infeasible_pairs)])

                    aggregated_samples[feasible_mask] = penalised.flatten()
                    
                # penalisation is over
                # now select a subset of solutions from X with maximum size N_max, which is a subset of all the solutions
                # you are picking a smaller set of the best solutions to reconstruct the model, this should improve performance

                # Stack everything together so when we are selecting a subset the input output pairs are kept together.
                S_feasible = aggregated_samples[~feasible_mask]
                S_infeasible = aggregated_samples[feasible_mask]
                y_feasible = ysample[~feasible_mask]
                y_infeasible = ysample[feasible_mask]
                g_feasible = gsample[~feasible_mask]
                g_infeasible = gsample[feasible_mask]

                feasible_pairs = np.hstack((X_feasible, y_feasible, g_feasible, np.reshape(S_feasible, (-1,1))))
                infeasible_pairs = np.hstack((X_infeasible, y_infeasible, g_infeasible, np.reshape(S_infeasible, (-1,1))))


            
                X_prime = self.select_subset(feasible_pairs, infeasible_pairs, ref_dir, N_max)
                
                
                model_input = X_prime[:,:self.n_vars]
                model_output = X_prime[:,-1]
                # fit a model using the penalised aggragated samples and the subset of inputs.


                agg_model = GPy.models.GPRegression(model_input, np.reshape(model_output, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                agg_model.Gaussian_noise.variance.fix(0)
                agg_model.optimize(messages=False,max_f_eval=1000)

                # train constraint functions
                constraints = X_prime[:,self.n_vars+self.n_obj:-1]
                constraint_models = []
                n_constr = self.n_eq_constr+self.n_ieq_constr
                for i in range(n_constr):
                    model = GPy.models.GPRegression(model_input, np.reshape(constraints[:,i], (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                    model.Gaussian_noise.variance.fix(0)
                    model.optimize(messages=False,max_f_eval=1000)
                    constraint_models.append(model)
                
                
                current_best = self.select_current_best(feasible_pairs, infeasible_pairs)
                next_X, _ = self._get_proposed(self.consraint_ei, agg_model, constraint_models, current_best)

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

                
        # the pf_approx is the pf of the feasible solutions
        pf_approx = util_functions.calc_pf(y_feasible)
        # Find the inputs that correspond to the pareto front.
        indicies = []
        for i, item in enumerate(y_feasible):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = y_feasible[indicies]

        res = result.Constrained_Res(y_infeasible, y_feasible, X_infeasible, X_feasible, pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

        return res
