import numpy as np
from scipy.optimize import differential_evolution
from pymoo.indicators.hv import HV
# from util_functions import EHVI, calc_pf, expected_decomposition
# from . import util_functions
import util_functions
import result
import GPy




class EMO:
    """
    Couckuyt, I., Deschrijver, D. & Dhaene, T. 
    Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization. 
    J Glob Optim 60, 575-594 (2014). https://doi.org/10.1007/s10898-013-0118-2

    EMO algorithm, WARNING: this takes ages to run.
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


    def decompose_into_cells(self, data_points, ref_point):
        """
        This decomoposes the non-dominated space into cells.
        It returns an array of sets of coordinates, each having information
        on the upper and lower bound of each cell.
        """
        if len(data_points) == 1:
            data_points = np.vstack((data_points, ref_point))
        def wfg(pl, ref_point):
            """
            L. While et al. 
            10.1109/TEVC.2010.2077298
            Algorithm for calculating the hypervolume of a set of points. Assumes minimisation.
            
            Params:
                pl: set of points.
                ref_point: the coordinate from which to measure hypervolume, the reference point.
            
            """

            # return sum([exclhv(pl, k, ref_point) for k in range(len(pl))])
            return [exclhv(pl, k, ref_point) for k in range(len(pl))]


        def exclhv(pl, k, ref_point):

            limit_set = limitset(pl, k)
            ls_hv = wfg(util_functions.calc_pf(limit_set), ref_point)
            return limit_set
            
        def limitset(pl, k):
            result = []
            for j in range(len(pl)-k-1):
                aux = []
                for (p, q) in zip(pl[k], pl[j+k+1]):
                    res = min(p,q)
                    aux.append(res)
                result.append(aux)
            # result = [[max(p,q) for (p,q) in zip(pl[k], pl[j+k+1])] for j in range(len(pl)-k-1)]
            return result

        def inclhv(p, ref_point):

            return np.product([np.abs(p[j] - ref_point[j]) for j in range(self.n_obj)])

        # Sorting the coords makes understanding whats going on easier
        sorted_coordinates = sorted(data_points, key=lambda coord: coord[0])

        # Get the limitsets of the pareto set of the points
        ss = np.asarray(wfg(sorted_coordinates, ref_point))
        # import pdb; pdb.set_trace()

        # The final limitset needs to be fixed to include the correct point, this is due to a limitation of the modified
        # wfg algorithm
        # I dont think i need to call calc_pf here???
        ss[-1] = [[ref_point[0], util_functions.calc_pf(sorted_coordinates)[-1][0]]]

        # We get the upper bounds of each cell
        upperlower = [[sorted_coordinates[i], ss[i][0]] for i, _ in enumerate(ss)]

        # Now in this loop we include the other coordinates for each cell
        bbbb = []
        asdff = []
        for i in upperlower:
            asdf = []
            for j in i:
                asdf.append([1,j[1]])
                bbbb.append([1,j[1]])
            asdff.append(asdf)


        # stack the coordinates together
        final = np.hstack((upperlower, asdff))
        # We need to fix the final cell again.
        final[-1][-1] = ref_point
        return final#

    
    
    def hypervolume_improvement(self, query_point, P, ref_point):
        """
        query_point: objective vector to query:
        P pareto set
        Returns the improvment in hypervolume from the inclusion of the query point
        into the set of objective values.
        """
        before = util_functions.wfg(P, ref_point)
        # import pdb; pdb.set_trace()

        aggregated = np.vstack((P,query_point))
        after = util_functions.wfg(aggregated, ref_point)
        improvement = after - before 
        if improvement > 0:
            return improvement
        else:
            return 0

    def hypervolume_based_PoI(self,X, models, P, cells):
        """
        Hypervolume based Probability of Improvement
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
        # As spoken about in the paper, the variance of the predicted models
        # is not used, only the mean.

        def vol2(mu, lower, upper):
            counter = 0
            for j in range(self.n_obj):
                if upper[j] > mu[j]:
                    counter = counter + 1
            if counter == self.n_obj:
                return np.product([upper[j]-max(lower[j],mu[j]) for j in range(self.n_obj)])
            else:
                return 0

        sum_total = sum([vol2(mu, cells[i][3], cells[i][0]) for i, value in enumerate(cells)])

        hvi = self.hypervolume_improvement(mu, P, self.max_point)
        # print(sum_total*hvi)
        return sum_total * hvi


    def get_proposed(self, function, P, cells, models):
        """
        Function used to optimise the evaluation criterion.
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
            cells = self.decompose_into_cells(util_functions.calc_pf(ysample), self.ideal_point)

            # optimse the criterion
            X_next, _ = self.get_proposed(self.hypervolume_based_PoI, ysample, cells, models)
            
            print("XFOUND")

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


    


