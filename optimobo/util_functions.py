
import numpy as np
from scipy import stats
from pygmo import fast_non_dominated_sorting, hypervolume
import GPy


# def expected(X, models, agg_func, cache, weights):
#     """
#     I cant remember why i wrote this function, i should probably delete it.
 
#     """

#     predicitions = []
#     for i in models:
#         # import pdb; pdb.set_trace()
#         output = i.predict(np.asarray([X]))
#         predicitions.append(output)

#     dimensions = len(models)

#     # Translate the samples to the correct position
#     sample_values = change(predicitions, cache, dimensions)

#     n_samples = len(sample_values)

#     # PBI values of all the points sampled from the distribution.
#     aggregated_samples = np.asarray([agg_func(x, weights) for x in sample_values])

#     total = np.mean(aggregated_samples)
#     std = np.std(aggregated_samples)

#     # Calculate twice the standard deviation
#     twice_std = 2 * std

#     upper_bound = total + twice_std
#     lower_bound = total - twice_std

#     return total, upper_bound, lower_bound


def generate_latin_hypercube_samples(num_samples: int, variable_ranges: list[float]):
    """
    Function that generates latin hypercube samples.
    Params:
        num_samples : int
    """
    num_vars = len(variable_ranges)
    samples = np.empty((num_samples, num_vars))

    for i, (min_val, max_val) in enumerate(variable_ranges):
        intervals = np.linspace(min_val, max_val, num_samples + 1)
        points = np.random.rand(num_samples) + np.arange(num_samples)
        points /= num_samples
        samples[:, i] = np.random.permutation(intervals[:-1] + (intervals[1:] - intervals[:-1]) * points)

    return samples


def calc_pf(Y: list[float]):
    """
    For a bi-objective minimisation problem, this function computes the pareto front via non-dominated sorting.
    Y:  a set of points within an objective space. numpy array in the form: (n_points x n_dimensions) 
        e.g. np.shape(y) = (800,2) 800 points in a 2 dimensional objective space.

    returns: Pareto front, in the same form as y, numpy array in the shape: (n_points x n_dimensions)
    """
    ndf, _, _, _ = fast_non_dominated_sorting(Y)
    return np.asarray([list(Y[i]) for i in ndf[0]])



def EHVI_2D_aux(PF,r,mu,sigma):
    """
    doi: 10.5281/zenodo.6406360 

    Inputs:

    PF: Pareto front approximation
    r: reference point
    mu: mean of a multivariate gaussian distribution 
    sigma : variance of a multivariate gaussian distribution

    """
    n = PF.shape[0]
    S1 = np.array([r[0],-np.inf])
    S1 = S1.reshape(1,-1)
    Send = np.array([-np.inf,r[1]])
    Send = Send.reshape(1,-1)
    index = np.argsort(PF[:,1])

    # get the locations of the pareto front approximation so we can define the stripes
    S = PF[index,:]

    # Array of all the stripes
    # Send is "end of S"
    S = np.concatenate((S1,S,Send),axis = 0)

    # get some weird vectors y1 and y2, they relate to the stripes somehow, read the definition again its complicated
    y1 = S[:,0] 
    y2 = S[:,1]

    y1 = y1.reshape(-1,1)
    y2 = y2.reshape(-1,1)

    mu = np.reshape(mu, (1,-1))
    sigma = np.reshape(sigma, (1,-1))
        
    sum_total1 = 0
    sum_total2 = 0

    for i in range(1,n+1):
        t = (y1[i] - mu[0][0])/sigma[0][0]
        # import pdb; pdb.set_trace()
        sum_total1 = sum_total1 + (y1[i-1] - y1[i])*stats.norm.cdf(t[0])*psi_cal(y2[i],y2[i],mu[0][1],sigma[0][1])
        sum_total2 = sum_total2 + (psi_cal(y1[i-1],y1[i-1],mu[0][0],sigma[0][0]) \
                                    - psi_cal(y1[i-1],y1[i],mu[0][0],sigma[0][0]))*psi_cal(y2[i],y2[i],mu[0][1],sigma[0][1])
        
    EHVI = sum_total1 + sum_total2
    return EHVI

def psi_cal(a,b,m,s):
    t = (b - m)/s
    # import pdb; pdb.set_trace()
    return s*stats.norm.pdf(t[0]) + (a - m)*stats.norm.cdf(t[0])


def EHVI(X, models, max_point, PF, cache):
    """
    Calculate Expected hypervolume improvement of a point given the current solutions.
    This is for 2d objective spaces.
    
    Params:
        X, solution vector, numpy array.
        models, array of sklearn gaussian processes built on each objective.
        max_point, coordinates of a maximum point, a reference point.
        PF, current pareto front approximation, array of coordinates.
        cache, array of samples that are translated and then evaluated.

    Return: 
        The expected hypervolume improvement of the objective vector X. Float.

    """
    # n_samples = 1024

    predictions = []
    for i in models:
        output = i.predict(np.asarray([X]))
        # output = i.predict(np.asarray([X]), return_std=True)
        predictions.append(output)

    sample_values = change(predictions, cache, 2)

    samples_vals = np.asarray(sample_values)
    cov = np.cov(samples_vals[:,0], samples_vals[:,1])
    r = max_point
    mu = np.asarray([predictions[0][0][0], predictions[1][0][0]])

    return(EHVI_2D_aux(PF, r, mu, cov))


def EHVI_3D(X, models, max_point, PF, cache):
    """
    Calculate EHVI in a 3d objective space.
    Calculate Expected hypervolume improvement of a point given the current solutions.
    
    Params:
        X, solution vector, numpy array.
        models, array of sklearn gaussian processes built on each objective.
        max_point, coordinates of a maximum point, a reference point.
        PF, current pareto front approximation, array of coordinates.
        cache, array of samples that are translated and then evaluated.

    Return: 
        The expected hypervolume improvement of the objective vector X. Float.
    """

    predictions = []
    for i in models:
        output = i.predict(np.asarray([X]))
        # output = i.predict(np.asarray([X]), return_std=True)
        predictions.append(output)

    dimensions = len(models)
    sample_values = change(predictions, cache, dimensions)

    samples_vals = np.asarray(sample_values)

    answer = 0
    hv = hypervolume(PF)
    Sminus = hv.compute(max_point)
    # import pdb; pdb.set_trace()

    for i in samples_vals:

        hvol = (max_point[0] - i[0]) * (max_point[1] - i[1]) * (max_point[2] - i[2])
        hv_one = hypervolume([i])
        hvol = hv_one.compute(max_point)

        hvol -= Sminus

        if hvol > 0:
            answer += hvol
        
    # print(answer)
    return answer / len(samples_vals)


def change(predicitions, samples, dimensions):
    """
    This function takes the predictions from the models, along with some pre generated uniform random samples.
    It returns normally distributed samples with the mean in predictions and a covariance matrix given in predicitions.
    
    Params:
        predictions, array containing means and standard deviations from multiple gaussian processes.
        samples, cached sobol samples that will be translated to match the mean and variance of the predictions.
        dimensions, the number of dimensions the samples are in, the number of predictions.
        

    returns:
        array of coordinates, normally distributed samples with a mean and variance taken from the predictions.
    """

    mus = [predicitions[i][0][0] for i in range(dimensions)]
    sigmas = [predicitions[0][1][0] for i in range(dimensions)]
    
    scaled_samples = [samples[:,i] * np.sqrt(sigmas[i]) + mus[i] for i in range(dimensions)]

    return np.asarray(scaled_samples).T

    # return list(zip(scaled_samples1, scaled_samples2))


################################################################################

def ei_over_decomposition(X, models, weights, agg_func, minimum_current_val, n_samples):
    """
    Generic function to compute the expected improvement of a point over the current best found point in the objective space.

    Params:
        X: objective vector to evaluate
        models: a list containing sklearn gaussian processes trained on the values of each objective.
        weights: weights for each objective.
        agg_func: the aggregation/scalarisation function to be used as a measure of convergence.
        minimum_current_val: best sample point according to the agg_func and the weights. 
        n_samples: the number of samples to be taken from the combined distribution to calculate 
                   the EI value. 

    Returns:
        float, the expected improvement of X over the minimum_current_val according to some agg_func 
        used to measure convergence.
    """

    predicitions = []
    for i in models:
        # import pdb; pdb.set_trace()
        output = i.predict(np.asarray([X]), return_std=True)
        predicitions.append(output)

    mu = np.asarray([predicitions[0][0][0], predicitions[1][0][0]])
    sigma = np.asarray([predicitions[0][1][0], predicitions[0][1][0]])

    # n_samples = len(sample_values)
    y1_samples = np.random.normal(mu[0], sigma[0], size=n_samples)
    y2_samples = np.random.normal(mu[1], sigma[1], size=n_samples)


    # sample_values = np.asarray(list(zip(y1_samples,y2_samples)))
    sample_values = np.stack((y1_samples,y2_samples), axis = 1)
    aggregated_samples = np.asarray([agg_func(x, weights) for x in sample_values])

    total = np.mean(np.maximum(np.zeros((n_samples,1)), minimum_current_val - aggregated_samples ))
    # print(total)

    return total

def expected_decomposition(X, models, weights, agg_func, agg_function_min, cache):
    """
    Function to calculate the expected improvement of an objective vector. It is calculated in respect to a 
    scalarisation/decomposition function used to measure convergence.

    Params:
        X, objective vector
        models, list of sklearn gaussian processes trained of each of the objectives
        weights, weights for each objective
        agg_func, the decomposition function/aggregation function that is used as a performance 
        measure.
        agg_function_min, the best converged objective vector according to the weights and the 
        aggregation function.
        cache, the cached samples.

    Returns:
        The expected improvement of objective vector X over the current best objective vector respect 
        to the agg_func and the weights.
    """

    # get some point and find its mean and std for both models
    predicitions = []
    for i in models:
        # import pdb; pdb.set_trace()
        output = i.predict(np.asarray([X]))
        predicitions.append(output)


    dimensions = len(models)
    # Translate the samples to the correct position
    sample_values = change(predicitions, cache, dimensions)

    n_samples = len(sample_values)

    # PBI values of all the points sampled from the distribution.
    # aggregated_samples = np.asarray([agg_func(x, weights) for x in sample_values])
    aggregated_samples = agg_func(sample_values, weights)


    total = np.mean(np.maximum(np.zeros((n_samples,1)), agg_function_min - aggregated_samples ))
    # print(total)

    return total



