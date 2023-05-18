
import numpy as np
from scipy import stats
from pygmo import fast_non_dominated_sorting

# from pymoo.util.ref_dirs import get_reference_directions


def calc_pf(Y):
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

def EHVI(X, models, ideal_point, max_point, PF, cache):
    """
    Calculate Expected hypervolume improvement of a point given the current solutions.
    
    Params:
        X, solution vector
        models, list of sklearn gaussian processes built on each objective.
        ysample, current set of solutions
        cache, array of samples that are translated and then evaluated.

    Return: 
        The expected hypervolume improvement of the objective vector X.

    """
    # n_samples = 1024

    predicitions = []
    for i in models:
        output = i.predict(np.asarray([X]), return_std=True)
        predicitions.append(output)


    sample_values = change(predicitions, cache)

    samples_vals = np.asarray(sample_values)
    cov = np.cov(samples_vals[:,0], samples_vals[:,1])

    # PF = calc_pf(ysample)
    r = max_point
    mu = np.asarray([predicitions[0][0][0], predicitions[1][0][0]])
    # sigma = np.asarray([[predicitions[0][1][0], 0],[0, predicitions[0][1][0]]])
    # sigma = predicitions[0][1][0]
    return(EHVI_2D_aux(PF, r, mu, cov))

def change(predicitions, samples):
    """
    This function takes the predictions from both models, along with some pre generated uniform random samples.
    It returns normally distributed samples with the mean in predictions and a covariance matrix given in predicitions.
    mu; vector of means

    sigma; covariance matrix

    samples; uniform samples
    """
    
    # get the means and standard deviations
    mu1 = predicitions[0][0][0]
    mu2 = predicitions[1][0][0]

    sigma1 = predicitions[0][1][0]
    sigma2 = predicitions[0][1][0]
    
    # import pdb; pdb.set_trace()

    # transform the 
    # Scale and shift the normal distribution to match the desired mean and standard deviation
    scaled_samples1 = samples[:,0] * sigma1 + mu1
    scaled_samples2 = samples[:,1] * sigma2 + mu2
    # return list(map(np.asarray, zip(scaled_samples1, scaled_samples2)))
    return np.stack((scaled_samples1,scaled_samples2), axis = 1)

    # return list(zip(scaled_samples1, scaled_samples2))


################################################################################
"""
Scalarisation functions.
"""

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
        output = i.predict(np.asarray([X]), return_std=True)
        predicitions.append(output)

    # mu = np.asarray([predicitions[0][0][0], predicitions[1][0][0]])
    # sigma = np.asarray([predicitions[0][1][0], predicitions[0][1][0]])
    # print(mu)
    # print(sigma)

    # Translate the samples to the correct position
    sample_values = change(predicitions, cache)

    n_samples = len(sample_values)

    # PBI values of all the points sampled from the distribution.
    aggregated_samples = np.asarray([agg_func(x, weights) for x in sample_values])

    # import pdb; pdb.set_trace()

    total = np.mean(np.maximum(np.zeros((n_samples,1)), agg_function_min - aggregated_samples ))
    # print(total)

    return total


def chebyshev(f, W, ideal_point, max_point):
    
    nobjs = 2

    objs = [(f[i]-ideal_point[i])/(max_point[i]-ideal_point[i]) for i in range(nobjs)]

    return max([W[i]*(objs[i]) for i in range(nobjs)])


def ei_cheb_aux(sample_values, weights, ideal_point, max_point, TCH_min):
    """
    This helper function handles the creation of the samples around each point. It computes EI over Tchebycheff
    mu: mean vector
    sigma: vector of variances
    """
    n_samples = len(sample_values)

    TCHs = np.asarray([chebyshev(x, weights, ideal_point, max_point) for x in np.asarray(sample_values)])

    total = np.mean(np.maximum(np.zeros((n_samples,1)), TCH_min - TCHs ))
    return total

def EITCH(X, models, weights, ideal_point, max_point, min_tch, cache):
    """
    Example input ETCH(objective_value, models, [0.5,0.5], [-1.7,-1.9], [3,3])

    X represents some singular multi-objective function value. This function computes the 
    expected Tchebycheff at that point.
    """

    # get some point and find its mean and std for both models
    predicitions = []
    for i in models:
        output = i.predict(np.asarray([X]), return_std=True)
        predicitions.append(output)

    mu = np.asarray([predicitions[0][0][0], predicitions[1][0][0]])
    sigma = np.asarray([predicitions[0][1][0], predicitions[0][1][0]])

    sample_values = change(predicitions, cache)
    return ei_cheb_aux(sample_values, weights, ideal_point, max_point, min_tch)



def PBI(f, W, ideal_point, max_point):
    # import pdb; pdb.set_trace()
    objs = [(f[i]-np.asarray(ideal_point)[i])/(np.asarray(max_point)[i]-np.asarray(ideal_point)[i]) for i in range(2)]
    
    # trans_f = f - f_ideal # translated objective values 
    # print(W)
    # W = [0.5,0.5]
    # import pdb; pdb.set_trace()

    W = np.reshape(W,(1,-1))
    normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
    normW = normW.reshape(-1,1)
    # import pdb; pdb.set_trace()
    d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
    d_1 = d_1.reshape(-1,1)
    
    # import pdb; pdb.set_trace()

    d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
    d_1 = d_1.reshape(-1) 
    PBI = d_1 + 5*d_2 # PBI with theta = 5    
    PBI = PBI.reshape(-1,1)
    return PBI[0]




def EIPBI_aux(sample_values, weights, ideal_point, max_point, PBI_min):
    """
    mu: mean vector
    sigma: vector of variances
    """
    n_samples = len(sample_values)

    # PBI values of all the points sampled from the distribution.
    PBIs = np.asarray([PBI(x, weights, ideal_point, max_point) for x in sample_values])

    # import pdb; pdb.set_trace()

    total = np.mean(np.maximum(np.zeros((n_samples,1)), PBI_min - PBIs ))
    # print(total)

    return total

def EIPBI(X, models, weights, ideal_point, max_point, PBI_min, cache):
    """
    """

    # get some point and find its mean and std for both models
    predicitions = []
    for i in models:
        # import pdb; pdb.set_trace()
        output = i.predict(np.asarray([X]), return_std=True)
        predicitions.append(output)

    mu = np.asarray([predicitions[0][0][0], predicitions[1][0][0]])
    sigma = np.asarray([predicitions[0][1][0], predicitions[0][1][0]])
    # print(mu)
    # print(sigma)

    sample_values = change(predicitions, cache)


    return EIPBI_aux(sample_values, weights, ideal_point, max_point, PBI_min)
# print(expected_decomposition(X, models, weights, agg_func, agg_function_min, cache))
