import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from scipy.stats import qmc
from pymoo.indicators.hv import HV
import GPy
from pymoo.util.ref_dirs import get_reference_directions
from copy import deepcopy

import optimobo.util_functions as util_functions
import optimobo.result as result
import math
import matplotlib.pyplot as plt



class TuRBO_1():

    def __init__(self, test_problem, ideal_point, max_point, batch_size):
        self.test_problem = test_problem
        self.max_point = max_point
        self.ideal_point = ideal_point
        self.n_vars = test_problem.n_var
        self.n_obj = test_problem.n_obj
        self.upper = test_problem.xu
        self.lower = test_problem.xl
        self.n_evals = 0
        self.max_evals = 100
        self.Xsample = np.zeros((0, self.n_vars))
        self.ysample = np.zeros((0, self.n_obj))
        self.aggregated_samples = np.zeros((0, 1))
        self.batch_size = batch_size
        self.n_cand = min(100*self.n_vars, 5000)
        # self.n_cand = 100


        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.4
        self.length = self.length_init
        self.ref_dirs = get_reference_directions("das-dennis", self.n_obj, n_partitions=10)

        self.failtol = np.ceil(np.max([4.0 / batch_size, self.n_vars / batch_size]))
        self.succtol = 3
        
    def _objective_function(self, problem, x):
        """
        Wrapper for the objective function, makes my code clearer.
        Returns objective values.
        Params:
            problem: Problem object
            x: input 
        """
        return problem.evaluate(x)

    def normalise(self, X):
        # its normalising the landscape, not the objective space
        return (np.asarray(X) - np.asarray(self.lower)) / (np.asarray(self.upper) - np.asarray(self.lower))

    def denormalise(self, X):
         return (X * (self.upper - self.lower)) + self.lower

    def create_candidates(self, Xsample, ysample, GP, length):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert Xsample.min() >= 0.0 and Xsample.max() <= 1.0
        
        
        x_center = Xsample[ysample.argmin().item(), :][None, :]
        weights = GP.kern.lengthscale
        weights = weights / weights.mean()
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # plt.axvline(self.denormalise(lb), color="black")
        # plt.axvline(self.denormalise(ub), color="black")
        # plt.axvline(self.denormalise(x_center[0]), color="red")

        # import pdb; pdb.set_trace()
        # Draw a Sobolev sequence in [lb, ub]
        # qmc.Sobol(d=self.n_vars, scramble=True)
        sampler = qmc.Sobol(d=self.n_vars, scramble=False)
        sample = sampler.random(n=self.n_cand)
        sample = qmc.scale(sample, lb[0], ub[0])


        # Create a perturbation mask, comes into action with problems greater than 20 input dimensions.
        prob_perturb = min(20.0 / self.n_vars, 1.0)
        mask = np.random.rand(self.n_cand, self.n_vars) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.n_vars - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.n_vars))
        shape = np.shape(X_cand)
        X_cand[mask] = sample[mask]
        X_cand = np.reshape(X_cand, shape)

        # y_cand = GP.posterior_samples(np.reshape(X, (-1,1)), size=n_samples)
        y_cand = GP.posterior_samples(X_cand, size=self.batch_size)

        # import pdb; pdb.set_trace()
        return X_cand, y_cand


    def _restart(self):
        self._Xsample = []
        self._ysample = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._aggregated_samples) - 1e-3 * math.fabs(np.min(self._aggregated_samples)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def select_candidates(self, X_cand, y_cand):
        """Select candidates.
        Get the optimum of each sampled function"""
        X_next = np.ones((self.batch_size, self.n_vars))
        for i in range(self.batch_size):

            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:,0,i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        # import pdb; pdb.set_trace()
        return X_next

    def get_random_weight(self):
        return self.ref_dirs[np.random.randint(0,len(self.ref_dirs))]

    def solve(self, aggregation_func, max_evals=100, n_init_samples=5):
        """
        Main optimiser flow. 
        Aggregation func: scalarisation function used to scalarise the objective values
            This is a mono surrogate method.
        Max_evals: the maximum number of expensive function evaluations, the budget limit.
        n_init_samples: the number of initial samples.
        
        """

        self.max_evals = max_evals
        hypervolume_convergence = []

        while self.n_evals < self.max_evals:

            ref_point = self.max_point
            HV_ind = HV(ref_point=ref_point)
            hv = HV_ind(self.ysample)
            hypervolume_convergence.append(hv)

            # Initialize parameters
            self._restart()

            # Init samples
            variable_ranges = list(zip(self.test_problem.xl, self.test_problem.xu))
            Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples, variable_ranges)

            # Evaluate initial samples.
            ysample = np.asarray([self._objective_function(self.test_problem, x) for x in Xsample])

            # ref_dir = self.ref_dirs[np.random.randint(0,len(self.ref_dirs))]
            # How the reference vectors are changed needs to be thought about
            # ref_dir = np.array([0.5,0.5])

            # Aggregate Initial samples
            aggregated_samples = np.asarray([aggregation_func(i, [0.5,0.5]) for i in ysample]).flatten()

            self.n_evals = self.n_evals + n_init_samples
   
            self._Xsample = deepcopy(Xsample)
            self._ysample = deepcopy(ysample)
            self._aggregated_samples = np.reshape(aggregated_samples, (-1,1))

            # Append new init to the global history
            self.Xsample = np.vstack((self.Xsample, deepcopy(Xsample)))
            self.ysample = np.vstack((self.ysample, deepcopy(ysample)))

            self.aggregated_samples = np.vstack((self.aggregated_samples, np.reshape(deepcopy(aggregated_samples), (-1,1))))

            
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Normalise inputs, this is needed as the candidate selection assumes normalisation.
                Xsample_normed = self.normalise(self._Xsample)

                # Get local history
                ysam = self._ysample
                aggre = self._aggregated_samples

                # Train model on the aggregated samples
                GP = GPy.models.GPRegression(Xsample_normed, np.reshape(aggre, (-1,1)), GPy.kern.Matern52(self.n_vars,ARD=True))
                GP.Gaussian_noise.variance.fix(0)
                GP.optimize(messages=False,max_f_eval=1000)

                # Returns n=batch_size posterior samples (functions) from the GP, bounded by the trust region.
                # y_cand are scalarised/aggregated values, not objective vectors.
                X_cand, y_cand = self.create_candidates(
                    Xsample_normed, aggre, GP, length=self.length
                )
               
                # Get the best (smallest) value from each of the posterior samples (functions)
                X_next = self.select_candidates(X_cand, y_cand)

                # Undo the normalisation.
                X_next = self.denormalise(X_next)

                # Evaluate batch.
                # This is what is aggregated.
                ref_dir = self.get_random_weight()
                print(ref_dir)
                y_next = np.array([self._objective_function(self.test_problem, x) for x in X_next])
                aggregated_next = np.array([aggregation_func(y, ref_dir) for y in y_next])

                # Update trust region.
                self._adjust_length(aggregated_next)

                self.n_evals += self.batch_size

                self._Xsample = np.vstack((self._Xsample, X_next))
                self._ysample = np.vstack((self._ysample, y_next))
                self._aggregated_samples = np.vstack((self._aggregated_samples, aggregated_next))
                # self.agg_samples = np.vstack((self.ysample, y_next))


                # if self.verbose and fX_next.min() < self.fX.min():
                #     n_evals, fbest = self.n_evals, fX_next.min()
                #     print(f"{n_evals}) New best: {fbest:.4}")
                #     sys.stdout.flush()

                # Append data to the global history
                self.Xsample = np.vstack((self.Xsample, deepcopy(X_next)))
                self.ysample = np.vstack((self.ysample, deepcopy(y_next)))
                # import pdb; pdb.set_trace()
                # self.aggregated_samples = np.vstack((self.aggregated_samples, deepcopy(aggregated_next)))
                self.aggregated_samples = np.vstack((self.aggregated_samples, np.reshape(deepcopy(aggregated_samples), (-1,1))))


                ###################################################################
                # import pdb; pdb.set_trace()
            #     X = np.linspace(0,8)
            #     X = np.asarray([[x] for x in X])
            # # Y = np.asarray([self._objective_function(problem, x) for x in X])
            #     # plt.figure(figsize=(10, 6))
            #     for i in range(self.batch_size):
            #         # plt.plot(X, y_cand[:,0,i], label=f'Sample {i+1}')
            #         plt.scatter(self.denormalise(X_cand), y_cand[:,0,i], label=f'Sample {i+1}')
            #     # for i in range(3):
            #     #     # plt.plot(X, y_cand[:,0,i], label=f'Sample {i+1}')
            #     #     plt.plot(X, y_cand[:,0,i])
            #     for i in X_next:
            #         plt.axvline(i)


            #     # plot the actual func

            #     Y = np.array([self._objective_function(self.test_problem, x) for x in X])
            #     aggre = np.array([aggregation_func(y, ref_dir) for y in Y])
            #     plt.plot(X, aggre)
    
            #     plt.scatter(X_next, aggregated_next)
            #     plt.title('Sample Functions from Gaussian Process')
            #     plt.xlabel('X')
            #     plt.ylabel('Sampled Values')
            #     plt.legend()
            #     plt.show()


        pf_approx = util_functions.calc_pf(self.ysample)
        # Identify the inputs that correspond to the pareto front solutions.
        indicies = []
        for i, item in enumerate(self.ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = self.Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, self.ysample, self.Xsample, hypervolume_convergence, self.n_obj, n_init_samples)

        return res


class TuRBO_M(TuRBO_1):
    """The TuRBO-m algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points *FOR EACH TRUST REGION* (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    n_trust_regions : Number of trust regions
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo5 = TurboM(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, n_trust_regions=5)
        turbo5.optimize()  # Run optimization
        X, fX = turbo5.X, turbo5.fX  # Evaluated points
    """

    def __init__(
        self,
        test_problem,
        ideal_point,
        max_point,
        batch_size,
        n_trust_regions
        
    ):
        self.n_trust_regions = n_trust_regions
        super().__init__(
            test_problem=test_problem,
            ideal_point=ideal_point,
            max_point=max_point,
            batch_size=batch_size,
        )

        self.succtol = 3
        self.failtol = max(5, self.n_vars)

        # Very basic input checks
        

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart()

    def _restart(self):
        print("RESTART CALLED")
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)

    def _adjust_length(self, fX_next, i):
        assert i >= 0 and i <= self.n_trust_regions - 1

        fX_min = self.ysample[self._idx[:, 0] == i, 0].min()  # Target value
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        # assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.n_vars)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.n_vars))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (self.n_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next, idx_next

    def solve(self, aggregation_func, max_evals, n_init_samples):
        hypervolume_convergence = []

        self.max_evals = max_evals
        assert self.n_trust_regions > 1 and isinstance(max_evals, int)
        assert self.max_evals > self.n_trust_regions * n_init_samples, "Not enough trust regions to do initial evaluations"
        assert max_evals > self.batch_size, "Not enough evaluations to do a single batch"
        """Run the full optimization process."""
        # Create initial points for each TR
        for i in range(self.n_trust_regions):

            # Get init samples
            variable_ranges = list(zip(self.test_problem.xl, self.test_problem.xu))
            Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples, variable_ranges)
            # Evaluate initial samples.
            ysample = np.asarray([self._objective_function(self.test_problem, x) for x in Xsample])
            aggregated_samples = np.asarray([aggregation_func(i, [0.5,0.5]) for i in ysample]).flatten()

            # Update budget and set as initial data for this TR
            self.n_evals = self.n_evals + n_init_samples
            self._idx = np.vstack((self._idx, i * np.ones((n_init_samples, 1), dtype=int)))
   
            # self.Xsample = deepcopy(Xsample)
            # self.ysample = deepcopy(ysample)
            # self.aggregated_samples = np.reshape(aggregated_samples, (-1,1))

            # Append new init to the global history
            self.Xsample = np.vstack((self.Xsample, deepcopy(Xsample)))
            self.ysample = np.vstack((self.ysample, deepcopy(ysample)))
            self.aggregated_samples = np.vstack((self.aggregated_samples, np.reshape(deepcopy(aggregated_samples), (-1,1))))

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals:

            # Generate candidates from each TR
            # import pdb; pdb.set_trace()
            X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.n_vars))
            y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))
            for i in range(self.n_trust_regions):
                idx = np.where(self._idx == i)[0]  # Extract all "active" indices

                Xsample_normed = self.normalise(self.Xsample[idx,:])

                # Get local history
                ysam = self.ysample[idx,:]
                aggre = self.aggregated_samples[idx,:]
                # import pdb; pdb.set_trace()

                # Train model on the aggregated samples
                print(len(Xsample_normed))
                print(len(aggre))
                GP = GPy.models.GPRegression(Xsample_normed, aggre, GPy.kern.Matern52(self.n_vars,ARD=True))
                GP.Gaussian_noise.variance.fix(0)
                GP.optimize(messages=False,max_f_eval=1000)

                # Returns n=batch_size posterior samples (functions) from the GP, bounded by the trust region.
                # y_cand are scalarised/aggregated values, not objective vectors.
                # X_cand[i, :, :], y_cand[i, :, :] = self.create_candidates(
                #     Xsample_normed, aggre, GP, length=self.length[i]
                # )
                # import pdb; pdb.set_trace()

                X_cand_i, y_cand_i = self.create_candidates(
                    Xsample_normed, aggre, GP, length=self.length[i]
                )
                X_cand[i, :, :] = X_cand_i
                y_cand[i, :, :] = np.reshape(y_cand_i, (self.n_cand, self.batch_size))
                # import pdb; pdb.set_trace()
                print("HELLO")
                


            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            # Undo the warping
            X_next = self.denormalise(X_next)
            # X_next = from_unit_cube(X_next, self.lb, self.ub)

            ref_dir = self.get_random_weight()
            print(ref_dir)
            y_next = np.array([self._objective_function(self.test_problem, x) for x in X_next])

            aggregated_next = np.array([aggregation_func(y, ref_dir) for y in y_next])
            # import pdb; pdb.set_trace()


            # Update trust regions
            for i in range(self.n_trust_regions):
                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0:
                    self.hypers[i] = {}  # Remove model hypers
                    fX_i = aggregated_next[idx_i]

                    # if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                    #     n_evals, fbest = self.n_evals, fX_i.min()
                    #     print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                    #     sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # Update budget and append data
            self.n_evals += self.batch_size
            self.Xsample = np.vstack((self.Xsample, deepcopy(X_next)))
            self.ysample = np.vstack((self.ysample, deepcopy(y_next)))
            self.aggregated_samples = np.vstack((self.aggregated_samples, np.reshape(deepcopy(aggregated_next), (-1,1))))
            self._idx = np.vstack((self._idx, deepcopy(idx_next)))

            # Check if any TR needs to be restarted
            for i in range(self.n_trust_regions):
                if self.length[i] < self.length_min:  # Restart trust region if converged
                    print("UPDA")

                    idx_i = self._idx[:, 0] == i

                    # if self.verbose:
                    #     n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                    #     print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                    #     sys.stdout.flush()

                    # Reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1  # Remove points from trust region
                    self.hypers[i] = {}  # Remove model hypers

                    # Create a new initial design
                    Xsample = util_functions.generate_latin_hypercube_samples(n_init_samples, variable_ranges)
                    # Evaluate initial samples.
                    ysample = np.asarray([self._objective_function(self.test_problem, x) for x in Xsample])
                    aggregated_samples = np.asarray([aggregation_func(i, [0.5,0.5]) for i in ysample]).flatten()


                    # Print progress
                    # if self.verbose:
                    #     n_evals, fbest = self.n_evals, fX_init.min()
                    #     print(f"{n_evals}) TR-{i} is restarting from: : {fbest:.4}")
                    #     sys.stdout.flush()

                    # Append data to local history
                    self.Xsample = np.vstack((self.Xsample, Xsample))
                    self.ysample = np.vstack((self.ysample, ysample))
                    self.aggregated_samples = np.vstack((self.aggregated_samples, np.reshape(aggregated_samples, (-1,1))))
                    self._idx = np.vstack((self._idx, i * np.ones((n_init_samples, 1), dtype=int)))
                    self.n_evals += n_init_samples
        ##################################################################
        #     import pdb; pdb.set_trace()
        #     X = np.linspace(0,8)
        #     X = np.asarray([[x] for x in X])
        # # Y = np.asarray([self._objective_function(problem, x) for x in X])
        #     # plt.figure(figsize=(10, 6))
        #     for i in range(self.n_trust_regions):

        #         for j in range(self.batch_size):
        #         # plt.plot(X, y_cand[:,0,i], label=f'Sample {i+1}')
        #             plt.scatter(self.denormalise(X_cand)[i], y_cand[i][:,j], label=f'Sample {i+1}')
        #     # for i in range(3):
        #     #     # plt.plot(X, y_cand[:,0,i], label=f'Sample {i+1}')
        #     #     plt.plot(X, y_cand[:,0,i])
        #     for i in X_next:
        #         plt.axvline(i)


        #     # plot the actual func

        #     Y = np.array([self._objective_function(self.test_problem, x) for x in X])
        #     aggre = np.array([aggregation_func(y, ref_dir) for y in Y])
        #     plt.plot(X, aggre)

        #     plt.scatter(X_next, aggregated_next)
        #     plt.title('Sample Functions from Gaussian Process')
        #     plt.xlabel('X')
        #     plt.ylabel('Sampled Values')
        #     plt.legend()
        #     plt.show()


        pf_approx = util_functions.calc_pf(self.ysample)
        # Identify the inputs that correspond to the pareto front solutions.
        indicies = []
        for i, item in enumerate(self.ysample):
            if item in pf_approx:
                indicies.append(i)
        pf_inputs = self.Xsample[indicies]

        res = result.Res(pf_approx, pf_inputs, self.ysample, self.Xsample, hypervolume_convergence, self.n_obj, n_init_samples)

        return res