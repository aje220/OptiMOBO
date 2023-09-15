import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from pymoo.indicators.hv import HV
import GPy
# from util_functions import EHVI, calc_pf, expected_decomposition
# from . import util_functions
import optimobo.util_functions as util_functions
import optimobo.result as result





class EMO:
    """
    Couckuyt, I., Deschrijver, D. & Dhaene, T. 
    Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization. 
    J Glob Optim 60, 575-594 (2014). https://doi.org/10.1007/s10898-013-0118-2

    EMO algorithm. Using Hypervolume based Probability of Improvement.
    Only works for 2D so far.
    
    """
    
    def __init__(self, test_problem, ideal_point, max_point):
        self.test_problem = test_problem
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


    def decompose_into_cells(self, data_points):
        """
        This decomoposes the non-dominated space into cells.
        It returns an array of sets of coordinates, each having information
        on the upper and lower bound of each cell.
        """

        def inclhv(p, ref_point):
            """
            Hypervolume of a single objective vector.
            p, objective vector
            ref_point: reference point from which to measure hypervolume.
            """
            return np.array([p, ref_point])


        def limitset(pl, k):
            result = []
            for j in range(len(pl)-k-1):
                aux = []
                for (p, q) in zip(pl[k], pl[j+k+1]):
                    res = max(p,q)
                    aux.append(res)
                result.append(aux)
            return result

        def iterative(table):
            """
            Iterative WFG algorithm that produces bounds to create the cells that represent the 
            non-dominated objective space.
            """
            stack = [] # tuples: front, index, inclusive HV, list of exclusive HVs
            front = table
            index = 0
            excl= []
            depth = 1
            hv = 0
            limi = []

            while depth > 0:
                if index >= len(front):
                    hv = sum(excl)
                    depth -= 1
                    if depth > 0:
                        front, index, incl, excl = stack.pop()

                        # this bit needs to be sorted to fix the algorithm for higher dimensions
                        lower = np.maximum(np.rot90(incl)[-1], hv[-1])
                        # lower = np.maximum(np.rot90(incl)[1], hv[1])
                        hv[-1] = lower
                        excl.append(hv)

                        index += 1
                else:
                    point = front[index]
                    incl = inclhv(point, self.ideal_point)
                    limset = util_functions.calc_pf(limitset(front, index))
                    if len(limset) == 0:
                        excl.append(incl)
                        index += 1
                    else:
                        limi.append(limset)
                        stack.append((front, index, incl, excl))
                        front = limset
                        index = 0
                        excl = []
                        depth += 1
            # import pdb; pdb.set_trace()
            return excl

        # Sorting the coords makes understanding whats going on easier,
        # also makes it less efficient, work for the future to fix
        sorted_coordinates = sorted(data_points, key=lambda coord: coord[0])
        
        # Add a final coordinate at the end of sorted coords. this ensures the final cell is correct.
        final_upper = np.zeros(self.n_obj)
        final_upper[-1] = sorted_coordinates[-1][-1]
        for i in range(0, len(sorted_coordinates[-1])-1):
            final_upper[i] = max(sorted_coordinates[-1][i], self.max_point[i])
        sorted_coordinates = np.vstack((sorted_coordinates, final_upper))

        # Get bounds of the points.
        ss = np.asarray(iterative(sorted_coordinates))



        # So the algorithm doesnt produce the bounds of the first cell but we can just construct it
        # as we have the information available 
        upper = np.zeros(self.n_obj)
        upper[0] = sorted_coordinates[0][0]
        for i in range(1, len(sorted_coordinates[0])):
            upper[i] = max(sorted_coordinates[0][i], self.max_point[i])
        
        first = np.vstack((upper, self.ideal_point))
        first = np.reshape(first, (1,2,2))

        # Return the first cell stacked with the old cell, we remove the failed final cell. A 
        # side effect of the modified algorithm.
        return np.vstack((first, ss))[:-1]


    def hypervolume_improvement(self, query_point, P, ref_point):
        """
        query_point: objective vector to query:
        P pareto set
        Returns the improvment in hypervolume from the inclusion of the query point
        into the set of objective values.
        """
        # this works
        before = util_functions.wfg(P, ref_point)
        # import pdb; pdb.set_trace()

        aggregated = np.vstack((P,query_point))

        after = util_functions.wfg(aggregated, ref_point)
        improvement = after - before 
        exc = util_functions.exclhv(aggregated, len(P)-1, ref_point)
        if improvement > 0:
            return improvement
        else:
            # print("CALL")
            return 0

    def vol5(self, mu, lower, upper):
        """
        Given mu and some upper and lower bounds (of a cell), this function computes how much volume 
        some coordinate (mu) takes up in the cell (if it does at all). Its caluclates the hypervolume
        improvement when you sum this function over all cells.
        """
        valid = []
        mask = upper > mu
        if mask.all():
            for j in range(self.n_obj):
                valid.append(upper[j]-max(lower[j],mu[j]))
        else:
            return 0
        return(np.prod(valid))


    def hypervolume_based_PoI(self,X, models, P, cells):
        """
        Hypervolume based Probability of Improvement, as defined in the paper.
        X, some input to the optimisation problem we are trying to solve
        models, the models, each trained on a different objective
        P, the set of solutions in the objective space, aka ysample.
        cells, the sets of coordinates specifing the cells in the non-dominated space
        """
        

        predictions = []
        for i in models:
            output = i.predict(np.asarray([X]))
            predictions.append(output)
        mu = np.asarray([predictions[0][0][0][0], predictions[1][0][0][0]])
        # mu2 = np.asarray([predictions[0][0][0][0], predictions[1][0][0][0]])
        # mu = [mu1, mu2]
        var = np.asarray([predictions[0][1][0][0], predictions[1][1][0][0]])
        std = np.sqrt(var+1e-5) # add some noise, otherwise it can throw lots of NaNs and makes optimisation very slow.

        # compute the probability of improvement
        all_of_them = []
        for j, value in enumerate(cells):
            ppp = []
            for i in range(self.n_obj):
                # xxx = norm.cdf(cells[j][0][i]) - norm.cdf(cells[j][1][i])
                xxx = norm.cdf(cells[j][0][i], loc=mu[i], scale=std[i]) - norm.cdf(cells[j][1][i], loc=mu[i], scale=std[i])
                ppp.append(xxx)
            all_of_them.append(np.product(ppp))
        poi = sum(all_of_them)
    
        # Get the hypervolume improvement.
        # The reason we use cells to compute improvement is because it means we only call the WFG algorithm once.
        # WFG is very computationally expensive therefore we can improve performance.
        improvement = sum([self.vol5(mu, cells[i][1], cells[i][0]) for i, value in enumerate(cells)])
    
        return poi * improvement


    def get_proposed(self, function, P, cells, models):
        """
        Function that implements the optimisation of the acquisition function.
        For EMO, this is Hypervolume based probability of improvement.
        """
        def obj(X):
            return -function(X, models, P, cells)

        x = list(zip(self.lower, self.upper))
        res = differential_evolution(obj, x)
        return res.x, res.fun


    def solve(self, n_iterations=100, n_init_samples=5):
        """
        This fcontains the main algorithm to solve the optimisation problem.
        """
        problem = self.test_problem

        # Initial samples.
        variable_ranges = list(zip(self.test_problem.xl, self.test_problem.xu))
        Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples, variable_ranges)

        # Evaluate inital samples.
        ysample = np.asarray([self._objective_function(problem, x) for x in Xsample])

        hypervolume_convergence = []

        for i in range(n_iterations):
            print("Iteration")

            # Get hypervolume metric.
            ref_point = self.max_point
            HV_ind = HV(ref_point=ref_point)
            # import pdb; pdb.set_trace()
            hv = HV_ind(ysample)
            hypervolume_convergence.append(hv)

            # Create models for each objective.
            models = []
            for i in range(problem.n_obj):
                ys = np.reshape(ysample[:,i], (-1,1))
                model = GPy.models.GPRegression(Xsample,ys, GPy.kern.Matern52(self.n_vars,ARD=True))


                model.Gaussian_noise.variance.fix(0)
                model.optimize(messages=False,max_f_eval=1000)
                models.append(model)

            # Decompose the non-dominated space into cells and retrieve the bounds.
            cells = self.decompose_into_cells(util_functions.calc_pf(ysample))
            X_next, _ = self.get_proposed(self.hypervolume_based_PoI, ysample, cells, models)
            

            # Evaluate the next input.
            y_next = self._objective_function(problem, X_next)

            # Add the new sample.
            ysample = np.vstack((ysample, y_next))

            # Update archive.
            Xsample = np.vstack((Xsample, X_next))


        pf_approx = util_functions.calc_pf(ysample)
        # Identify the inputs that correspond to the pareto front solutions.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

        return res


    


