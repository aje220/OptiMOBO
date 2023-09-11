import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from pymoo.indicators.hv import HV
# from util_functions import EHVI, calc_pf, expected_decomposition
# from . import util_functions
import util_functions
import result
import GPy



import matplotlib.pyplot as plt
GPy.plotting.change_plotting_library('matplotlib')
from matplotlib.gridspec import GridSpec
from Bayesian_optimisation_util import plot_acquisition



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
        # improvement = before - after 

        if improvement > 0:
            return improvement
        else:
            return 0

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
        std = np.sqrt(var+1e-5)


        all_of_them = []
        for j, value in enumerate(cells):
            ppp = []
            for i in range(self.n_obj):
                # import pdb; pdb.set_trace()

                # xxx = norm.cdf(cells[j][3][i], loc=mu[i], scale=np.sqrt(var[i]+1e-5)) - norm.cdf(cells[j][0][i], loc=mu[i], scale=np.sqrt(var[i]+1e-5))
                xxx = norm.cdf(cells[j][3][i], loc=mu[i], scale=std[i]) - norm.cdf(cells[j][0][i], loc=mu[i], scale=std[i])

                # xxx = norm.cdf(cells[j][3][i]) - norm.cdf(cells[j][0][i])

                ppp.append(xxx)
            all_of_them.append(np.product(ppp))
        poi = sum(all_of_them)

        def vol4(mu, lower, upper):
            valid = []
            for j in range(self.n_obj):
                if upper[j] > mu[j]:
                    valid.append(upper[j]-max(lower[j],mu[j]))
                else:
                    continue
            if len(valid) == 0:
                # print("CALLED")
                return 0
            else:
                return np.prod(valid)


        improvement = sum([vol4(mu, cells[i][3], cells[i][0]) for i, value in enumerate(cells)])
        print(improvement)
        print(poi)
        print(poi*improvement)
        return poi * improvement


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
        # Xsample = np.asarray([[0.8333625 ],[5.96804276],[1.51721302],[2.24009326],[4.94240056]])
        
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

            ###################################################
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
            # # aggre = np.asarray([acquisition_func(y, ref_dir) for y in Y])

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

            ###########
            # ax1 = fig.add_subplot(gs[1, 0])
            # plot_acquisition(X, [self.hypervolume_based_PoI(x, models, ysample, cells) for x in X], X_next)
            # ax1.set_ylabel(lab)

            # # ax1.set_xlim(0.0,8.0)
            # ax1.set_xlim(-0.1,8.2)

            # ax1.set_xlabel(r"$x$")
            # ax1.legend()
                
            #     # plot_acquisition(X, [expected_decomposition(acquisition_func, models, ref_dir, [-1.7,-1.9], [3,3], PBI_min) for x in X], X_next)
            
            # plt.ylabel("EI over Tchebicheff")
                    
            # plt.legend()
            # plt.xlabel("x")

            # pf = util_functions.calc_pf(Y)
            # # plt.subplot(1,2,2)
            # # plt.scatter(ysample[5:,0], ysample[5:,1], color="red", marker="s", zorder=10, label="Solution")
            # # plt.scatter(ysample[0:4,0], ysample[0:4,1], color="black", marker="x", zorder=5, label="Initial samples")
            # # plt.plot(Y[:,0], Y[:,1], color="grey", alpha=0.7, zorder=-5, label="Objective space")
            # # plt.plot(pf[:,0], pf[:,1], color="black", zorder=0, linewidth=2, label="Pareto front")
            # # plt.xlabel(r"$f_1(x)$")
            # # plt.ylabel(r"$f_2(x)$")
            # # plt.grid(alpha=0.3)
            # # plt.legend()

            # # plt.show()
            # ax2 = fig.add_subplot(gs[:, 1])
            # pf = util_functions.calc_pf(Y)
            # ax2.scatter(ysample[5:,0], ysample[5:,1], color="red", marker="s", zorder=10, label="Solution")
            # ax2.scatter(ysample[0:4,0], ysample[0:4,1], color="black", marker="x", zorder=5, label="Initial samples")
            # ax2.plot(Y[:,0], Y[:,1], color="grey", alpha=0.7, zorder=-5, label="Objective space")
            # ax2.plot(pf[:,0], pf[:,1], color="black", zorder=0, linewidth=2, label="Pareto front")
            # ax2.set_xlabel(r"$f_1(x)$")
            # ax2.set_ylabel(r"$f_2(x)$")
            # ax2.grid(alpha=0.3)
            # ax2.legend()
            # # plt.show()

            # plt.show()



        pf_approx = util_functions.calc_pf(ysample)


        # Identify the inputs that correspond to the pareto front solutions.
        indicies = []
        for i, item in enumerate(ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, ysample, Xsample, hypervolume_convergence, problem.n_obj, n_init_samples)

        return res


    


