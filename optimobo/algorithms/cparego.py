import random
import numpy as np
from scipy.stats import norm
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from scipy.optimize import differential_evolution
import GPy

import matplotlib.pyplot as plt
GPy.plotting.change_plotting_library('matplotlib')
from matplotlib.gridspec import GridSpec
from Bayesian_optimisation_util import plot_acquisition


# from pymoo.core.problem import ElementwiseProblem


# from util_functions import EHVI, calc_pf, expected_decomposition
# from . import util_functions
import util_functions
import result

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
            # X_prime_squared = np.setdiff1d(X[:,:-1], X_prime)
            # import pdb; pdb.set_trace()

            # compute the euclidean norm of the remaining solutions in the objective space
            deltas = [np.linalg.norm((x[self.n_vars:-1]-ref_dir)) for x in X_prime_squared]
            
            # import pdb; pdb.set_trace()

            # append the deltas to the end of each solution
            aux = np.hstack((X_prime_squared, np.reshape(deltas, (-1, 1))))

            # sort by the distances
            deltas_sorted = aux[aux[:,-1].argsort()]

            # select the best (smallest) distances
            X_prime_cubed = deltas_sorted[0:(N-N//2)]

            X_prime = np.vstack((X_prime, X_prime_cubed[:,:-1]))
            # import pdb; pdb.set_trace()
            return X_prime

        scores = [xi(x[:-1-self.n_obj]) for x in X_infeasible]
        X_infease_score = np.hstack((X_infeasible, np.reshape(scores, (-1,1))))
        X_infease_scores_sorted = X_infease_score[X_infease_score[:,-1].argsort()]
        # scores.sort()
        # import pdb; pdb.set_trace()


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

            # import pdb; pdb.set_trace()
            best_leftover = best_performing(difference, N_max-len(X_prime), ref_dir)
            X_prime = np.vstack((X_prime ,best_leftover ))
            return X_prime

        elif len(X_infeasible) >= H and len(X_feasible) >= H:
            X_prime = best_performing(X_feasible, H, ref_dir)
            X_prime_squared = X_infease_scores_sorted[0:(N_max - len(X_prime))]
            X_prime_squared = X_prime_squared[:,:-1] # as it has feasibility scores on the end we take it off
            # import pdb; pdb.set_trace()
            # if there are no feasible solutions the stack fails so ive included an if statement
            # import pdb; pdb.set_trace()
            if len(X_prime) == 0:
                return X_prime_squared
            else:
                return np.vstack((X_prime, X_prime_squared))

        elif len(X_infeasible) < H and len(X_feasible) >= H:
            X_prime = X_infeasible
            the_other = best_performing(X_feasible, N_max - len(X_prime), ref_dir)
            # import pdb; pdb.set_trace()
            return(np.vstack((X_prime, the_other)))

        elif len(X_infeasible) >= H and len(X_feasible) < H:
            X_prime = X_feasible
            X_prime_squared = X_infease_scores_sorted[0:(N_max - len(X_prime))]
            X_prime_squared = X_prime_squared[:,:-1]
            return np.vstack((X_prime, X_prime_squared))
        
        else:
            # should never reach this point
            return np.vstack((X_infeasible, X_feasible))



    def solve(self, n_iterations=100, n_init_samples=5, aggregation_func=None, N_max=100):
        """
        Main flow for the algorithm. Call this to solve the specified problem.
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

        for i in range(n_iterations):


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

                # upper = self.max_point
                # lower = self.ideal_point
                # import pdb; pdb.set_trace()

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
                # import pdb; pdb.set_trace()

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
                        # import pdb; pdb.set_trace()
                        return s_dot(x, x_star) + (numerator/denom)

                    # if there are infeasible solutions, calculate the infeasibility scores
                    # we need the max violaion of each constraint for XI
                    
                    # import pdb; pdb.set_trace()
                    # so this is find because you are passing contraint values to xi_single
                    workable = gsample[feasible_mask]
                    infeasibility_scores = [xi_single(x, v_max) for x in workable]
                    
                    # get x_star, 
                    x_star = None

                    if np.any(X_feasible): # if there is a feasible solution
                        x_star = feasible_pairs[np.argmin(feasible_pairs[:,-1])]
                        # x_star = current_best_X
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
                # print(len(X_prime))
                # import pdb; pdb.set_trace
            
                
                
                model_input = X_prime[:,:self.n_vars]
                model_output = X_prime[:,-1]
                # fit a model using the penalised aggragated samples and the subset of inputs.
                # import pdb; pdb.set_trace()
                model = GPy.models.GPRegression(model_input, np.reshape(model_output, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                model.Gaussian_noise.variance.fix(0)
                model.optimize(messages=False,max_f_eval=1000)

                

                

                next_X, _ = self._get_proposed(self._expected_improvement, model, current_best)

                next_y = self._objective_function(problem, next_X)
                # print(next_X)
                # print(next_y)
                # print(X_prime[0])

                # add the new sample to archive.
                ysample = np.vstack((ysample, next_y))
                # print(len(ysample))

                # Aggregate new sample
                agg = aggregation_func(next_y, ref_dir)

                # Update archive.
                aggregated_samples = np.append(aggregated_samples, agg)          
                Xsample = np.vstack((Xsample, next_X))

                # add constraints to constraint archive
                next_g = self._constraint_function(problem, next_X)
                gsample = np.vstack((gsample, next_g))

                # import pdb; pdb.set_trace()

                # ###################################################
                # X = np.asarray(np.arange(0, 8, 0.01))
                # X = np.asarray([[x] for x in X])
                # # Get the aggregation function outputs and objective values for plotting.
                # Y = np.asarray([self._objective_function(problem, x) for x in X])
                # fig = plt.figure(figsize=(11, 6))
                # gs = GridSpec(nrows=2, ncols=2)

                # ax0 = fig.add_subplot(gs[0, 0])
                # ax0.grid(alpha=0.3)
                # # import pdb; pdb.set_trace()
                # model.plot(ax=ax0, plot_limits=[0.0,8.0])
                # line = ax0.lines[0]
                # line.set_color("green")
                # fill = ax0.collections[0]
                # fill.set_color("green")  # Set confidence bound color to light blue
                # fill.set_alpha(0.3)
                # scatter1 = ax0.plot(model.X, model.Y, '+', color="b", markersize=12, markerfacecolor='black', markeredgecolor='black',label="_nolegend_", zorder=30)
                
                # for collection in ax0.collections:
                #     collection.set_color("green")
                # aggre = np.asarray([aggregation_func(y, ref_dir) for y in Y])

                # lab = None
                # ylimits = None
                # position = None
                # cols = None
                # line_label = None
                
                # lab = "TCH"
                # ylimits = (-0.5, 0.7)
                # position = "lower left"
                # cols=1
                # line_label = r"True $g_{TCH}(x)$"

                # true = ax0.plot(X, aggre, linestyle="--", color="green", label=line_label)
                

                # leg = ax0.legend(labels=["Data", r"$\mu$", r"$2*\sigma$", line_label, line_label])
                # leg.legendHandles[0].set_color('black')
                # leg.legendHandles[0].set_alpha(1.0)
                # # leg.legendHandles[0].set_marker('+')

                # handles, labels = ax0.get_legend_handles_labels()
                # # Create a new handle with the desired marker style
                # new_handle = plt.Line2D([], [], marker='+', linestyle='None', color='black')

                # # Replace the original handle with the new handle
                # handles[0] = new_handle

                # # Create a new legend with the modified handles and labels
                # ax0.legend(handles, labels, ncol=cols, loc=position)

                # # Display the plot
                # ax0.set_ylim(ylimits)
                # # ax0.set_xlim(0.0,8.0)
                # ax0.set_xlim(-0.1,8.2)

                # ###########
                # ax1 = fig.add_subplot(gs[1, 0])
                # plot_acquisition(X, [self._expected_improvement(x, model, current_best) for x in X], next_X)
                # ax1.set_ylabel(lab)

                # # ax1.set_xlim(0.0,8.0)
                # ax1.set_xlim(-0.1,8.2)

                # ax1.set_xlabel(r"$x$")
                # ax1.legend()
                    
                #     # plot_acquisition(X, [expected_decomposition(acquisition_func, models, ref_dir, [-1.7,-1.9], [3,3], PBI_min) for x in X], X_next)
               
                # plt.ylabel("EI over Tchebicheff")
                        
                # plt.legend()
                # plt.xlabel("x")

                # plt.show()

                
        # the pf_approx is the pf of the feasible solutions
        pf_approx = util_functions.calc_pf(feasible_pairs[:,self.n_vars:-1])
        # import pdb; pdb.set_trace()
        # Find the inputs that correspond to the pareto front.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

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
        # import pdb; pdb.set_trace()
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
        # if np.isnan(pof):
        #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
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
        # print(ei*pof)
        if np.isnan(ei*pof):
            import pdb; pdb.set_trace()
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
            # fitness_scores = self.aggregation_func(X[:,:-1], ref_dir)
            # fitness_scores.sort()
            X_prime = X_sorted[0:(N//2)]
            X_prime_squared = X_sorted[(N//2):]
            # X_prime_squared = np.setdiff1d(X[:,:-1], X_prime)
            # import pdb; pdb.set_trace()

            # compute the euclidean norm of the remaining solutions in the objective space
            # import pdb; pdb.set_trace()
            n_constr = self.n_ieq_constr+self.n_eq_constr
            deltas = [np.linalg.norm((x[self.n_vars:-1-n_constr]-ref_dir)) for x in X_prime_squared]
            
            # import pdb; pdb.set_trace()

            # append the deltas to the end of each solution
            aux = np.hstack((X_prime_squared, np.reshape(deltas, (-1, 1))))

            # sort by the distances
            deltas_sorted = aux[aux[:,-1].argsort()]

            # select the best (smallest) distances
            X_prime_cubed = deltas_sorted[0:(N-N//2)]

            X_prime = np.vstack((X_prime, X_prime_cubed[:,:-1]))
            # import pdb; pdb.set_trace()
            return X_prime

        # import pdb; pdb.set_trace()
        scores = [xi(x[:self.n_vars]) for x in X_infeasible]
        X_infease_score = np.hstack((X_infeasible, np.reshape(scores, (-1,1))))
        X_infease_scores_sorted = X_infease_score[X_infease_score[:,-1].argsort()]
        # scores.sort()
        # import pdb; pdb.set_trace()


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

            # import pdb; pdb.set_trace()
            best_leftover = best_performing(difference, N_max-len(X_prime), ref_dir)
            X_prime = np.vstack((X_prime ,best_leftover ))
            return X_prime

        elif len(X_infeasible) >= H and len(X_feasible) >= H:
            X_prime = best_performing(X_feasible, H, ref_dir)
            X_prime_squared = X_infease_scores_sorted[0:(N_max - len(X_prime))]
            X_prime_squared = X_prime_squared[:,:-1] # as it has feasibility scores on the end we take it off
            # import pdb; pdb.set_trace()
            # if there are no feasible solutions the stack fails so ive included an if statement
            # import pdb; pdb.set_trace()
            if len(X_prime) == 0:
                return X_prime_squared
            else:
                return np.vstack((X_prime, X_prime_squared))

        elif len(X_infeasible) < H and len(X_feasible) >= H:
            X_prime = X_infeasible
            the_other = best_performing(X_feasible, N_max - len(X_prime), ref_dir)
            # import pdb; pdb.set_trace()
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



    def solve(self, n_iterations=100, n_init_samples=5, aggregation_func=None, N_max=100):
        """
        Main flow for the algorithm. Call this to solve the specified problem.
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

        for i in range(n_iterations):


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

                # upper = self.max_point
                # lower = self.ideal_point
                # import pdb; pdb.set_trace()

                # change the bounds of the scalarisation object
                aggregation_func.set_bounds(lower, upper)

                # calculate scalar fitness score
                aggregated_samples = np.asarray([aggregation_func(i, ref_dir) for i in ysample]).flatten()
                # best_index = np.argmin(aggregated_samples)

                # current_best = aggregated_samples[best_index]
                # current_best_X = Xsample[best_index]

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
                # import pdb; pdb.set_trace()

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
                        # import pdb; pdb.set_trace()
                        return s_dot(x, x_star) + (numerator/denom)

                    # if there are infeasible solutions, calculate the infeasibility scores
                    # we need the max violaion of each constraint for XI
                    
                    # import pdb; pdb.set_trace()
                    # so this is find because you are passing contraint values to xi_single
                    workable = gsample[feasible_mask]
                    infeasibility_scores = [xi_single(x, v_max) for x in workable]
                    
                    # get x_star, 
                    x_star = None

                    if np.any(X_feasible): # if there is a feasible solution
                        x_star = feasible_pairs[np.argmin(feasible_pairs[:,-1])]
                        # x_star = current_best_X
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
                # import pdb; pdb.set_trace()

            
                X_prime = self.select_subset(feasible_pairs, infeasible_pairs, ref_dir, N_max)
                
                
                model_input = X_prime[:,:self.n_vars]
                model_output = X_prime[:,-1]
                # fit a model using the penalised aggragated samples and the subset of inputs.
                # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()

                agg_model = GPy.models.GPRegression(model_input, np.reshape(model_output, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                agg_model.Gaussian_noise.variance.fix(0)
                agg_model.optimize(messages=False,max_f_eval=1000)

                constraints = X_prime[:,self.n_vars+self.n_obj:-1]
                constraint_models = []
                n_constr = self.n_eq_constr+self.n_ieq_constr
                for i in range(n_constr):
                    # import pdb; pdb.set_trace()
                    model = GPy.models.GPRegression(model_input, np.reshape(constraints[:,i], (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                    model.Gaussian_noise.variance.fix(0)
                    model.optimize(messages=False,max_f_eval=1000)
                    constraint_models.append(model)

                # import pdb; pdb.set_trace()
                
                # okay where are we
                # write the functions 
                
                
                current_best = self.select_current_best(feasible_pairs, infeasible_pairs)
                next_X, _ = self._get_proposed(self.consraint_ei, agg_model, constraint_models, current_best)

                next_y = self._objective_function(problem, next_X)
                # print(next_X)
                # print(next_y)
                # print(X_prime[0])

                # add the new sample to archive.
                ysample = np.vstack((ysample, next_y))
                # print(len(ysample))

                # Aggregate new sample
                agg = aggregation_func(next_y, ref_dir)

                # Update archive.
                aggregated_samples = np.append(aggregated_samples, agg)          
                Xsample = np.vstack((Xsample, next_X))

                # add constraints to constraint archive
                next_g = self._constraint_function(problem, next_X)
                gsample = np.vstack((gsample, next_g))

                # import pdb; pdb.set_trace()

                ##################################################
                # X = np.asarray(np.arange(0, 8, 0.01))
                # X = np.asarray([[x] for x in X])
                # # Get the aggregation function outputs and objective values for plotting.
                # Y = np.asarray([self._objective_function(problem, x) for x in X])
                # fig = plt.figure(figsize=(11, 6))
                # gs = GridSpec(nrows=2, ncols=2)

                # ax0 = fig.add_subplot(gs[0, 0])
                # ax0.grid(alpha=0.3)
                # # import pdb; pdb.set_trace()
                # agg_model.plot(ax=ax0, plot_limits=[0.0,8.0])
                # line = ax0.lines[0]
                # line.set_color("green")
                # fill = ax0.collections[0]
                # fill.set_color("green")  # Set confidence bound color to light blue
                # fill.set_alpha(0.3)
                # scatter1 = ax0.plot(agg_model.X, agg_model.Y, '+', color="b", markersize=12, markerfacecolor='black', markeredgecolor='black',label="_nolegend_", zorder=30)
                
                # for collection in ax0.collections:
                #     collection.set_color("green")
                # aggre = np.asarray([aggregation_func(y, ref_dir) for y in Y])

                # lab = None
                # ylimits = None
                # position = None
                # cols = None
                # line_label = None
                
                # lab = "TCH"
                # ylimits = (-0.5, 0.7)
                # position = "lower left"
                # cols=1
                # line_label = r"True $g_{TCH}(x)$"

                # true = ax0.plot(X, aggre, linestyle="--", color="green", label=line_label)
                

                # leg = ax0.legend(labels=["Data", r"$\mu$", r"$2*\sigma$", line_label, line_label])
                # leg.legendHandles[0].set_color('black')
                # leg.legendHandles[0].set_alpha(1.0)
                # # leg.legendHandles[0].set_marker('+')

                # handles, labels = ax0.get_legend_handles_labels()
                # # Create a new handle with the desired marker style
                # new_handle = plt.Line2D([], [], marker='+', linestyle='None', color='black')

                # # Replace the original handle with the new handle
                # handles[0] = new_handle

                # # Create a new legend with the modified handles and labels
                # ax0.legend(handles, labels, ncol=cols, loc=position)

                # # Display the plot
                # ax0.set_ylim(ylimits)
                # # ax0.set_xlim(0.0,8.0)
                # ax0.set_xlim(-0.1,8.2)

                # ###########
                # ax1 = fig.add_subplot(gs[1, 0])
                # plot_acquisition(X, [self.consraint_ei(x, agg_model, constraint_models, current_best) for x in X], next_X)
                # ax1.set_ylabel(lab)

                # # ax1.set_xlim(0.0,8.0)
                # ax1.set_xlim(-0.1,8.2)

                # ax1.set_xlabel(r"$x$")
                # ax1.legend()
                    
                #     # plot_acquisition(X, [expected_decomposition(acquisition_func, models, ref_dir, [-1.7,-1.9], [3,3], PBI_min) for x in X], X_next)
               
                # plt.ylabel("EI over Tchebicheff")
                        
                # plt.legend()
                # plt.xlabel("x")

                # plt.show()

                
        # the pf_approx is the pf of the feasible solutions
        pf_approx = util_functions.calc_pf(feasible_pairs[:,self.n_vars:-1])
        # import pdb; pdb.set_trace()
        # Find the inputs that correspond to the pareto front.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

        return res
