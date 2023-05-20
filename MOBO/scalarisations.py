
import numpy as np
import matplotlib.pyplot as plt

class Scalarisation:
    """
    Parent class for all.
    This exists to collect the common arguments between the all scalarisation functions, the vector and weights.
    It also enables me to implement a __call__ function making scalariations easier to use.

    It makes implementing the scalaristion functions easier too. When writing them I need to implement an __init__
    that only concerns the parameters used in that particular function.
    """
    def __init__(self):
        return

    def __call__(self, *args, **kwargs):
        return self.do(*args, **kwargs)

    def do(self, F, weights, **args):
        """
        Params:
            F: array. Objective row vector.
            Weights: weights row vector. Corresponding to each component of the objective vector.
        """
        D = self._do(F, weights, **args).flatten()
        return D

class WeightedSum(Scalarisation):
    
    def _do(self, F, weights):
        # print(F)
        # print(weights)
        return np.sum(F*weights)



class Tchebicheff(Scalarisation):
    """
    Tchebicheff takes two extra arguments when instantiated. 

    Params:
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
        in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.

    """
    def __init__(self, ideal_point, max_point) -> None:
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point

    def _do(self, F, weights):
        F_prime = [(F[i]-self.ideal_point[i])/(self.max_point[i]-self.ideal_point[i]) for i in range(len(F))]
        return max([weights[i]*(F_prime[i]) for i in range(len(F))])


class AugmentedTchebicheff(Scalarisation):
    """
    Augemented Tchebicheff takes one extra argument when instantiated. 

    Params:
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
        in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
        alpha: determines the power of the additional augmented term, this helps prevent
        the addition of weakly Pareto optimal solutions. 
    """
    
    def __init__(self, ideal_point, max_point, alpha=0.0001) -> None:
        super().__init__()
        self.alpha = alpha
        self.ideal_point = ideal_point
        self.max_point = max_point

    def _do(self, F, weights):
        v = np.abs(F - self.ideal_point) * weights
        # import pdb; pdb.set_trace()
        tchebi = v.max(axis=0) # add augemnted part to this
        aug = np.sum(np.abs(F - self.ideal_point), axis=0)
        return tchebi + (self.alpha*aug)
    
class ModifiedTchebicheff(Scalarisation):
    """
    Like Augmented Tchebycheff we have the alpha parameter. 
    This differs from Augmented tchebicheff in that the slope that determines inclusion of weakly
    Pareto optimal solutions is different. 

    Params:
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
        in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
        alpha: influences inclusion of weakly Pareto optimal solutions.
    """
    
    def __init__(self, ideal_point, max_point, alpha=0.0001) -> None:
        super().__init__()
        self.alpha = alpha
        self.ideal_point = ideal_point
        self.max_point = max_point

    def _do(self, F, weights):
        left = np.abs(F - self.ideal_point)
        right = self.alpha*(np.sum(np.abs(F - self.ideal_point)))

        total = (left + np.asarray(right))*weights
        tchebi = total.max(axis=0)
        return tchebi

class ExponentialWeightedCriterion(Scalarisation):
    """
    Improves on WeightedSum by enabling discovery of all solutions in non-convex problems.

    Params:
        p: can influence performance.
    """

    def __init__(self, p=100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p = p

    def _do(self, F, weights, **kwargs):
        return np.sum(np.exp(self.p*weights - 1)*(np.exp(self.p*F)), axis=0)

class WeightedNorm(Scalarisation):
    """
    Generalised form of weighted sum.

    Params:
        p, infuences performance
    """

    def __init__(self, p=3) -> None:
        super().__init__()
        self.p = p

    def _do(self, F, weights):
        # import pdb; pdb.set_trace()
        return np.sum([np.abs(F[i])**self.p * weights[i] for i in range(len(F))])**(1/self.p)

class WeightedPower(Scalarisation):
    """
    Can find solutions in non-convex problems.
    Params:
        p: exponent, influences performance.
    """

    def __init__(self, p=3) -> None:
        super().__init__()
        self.p = p

    def _do(self, F, weights):
        return np.sum((F**self.p) * weights, axis=0)

class WeightedProduct(Scalarisation):
    """
    Can find solutions in non-convex problems.
    """
    def _do(self, F, weights):
        # this needs to be fixed
        return np.prod((F+100000)**weights, axis=0)

class PBI(Scalarisation):
    """
    First used as a measure of convergence in the evolutionary algorithm MOEA/D.
    Params:
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
        in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
        theta: multiplier that effects performance.
    """

    def __init__(self, ideal_point, max_point, theta=5) -> None:
        super().__init__()
        self.theta = theta
        self.ideal_point = ideal_point
        self.max_point = max_point

    
    def _do(self, f, weights):

        objs = [(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(2)]
        
        W = np.reshape(weights,(1,-1))
        normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
        normW = normW.reshape(-1,1)

        d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
        d_1 = d_1.reshape(-1,1)

        d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
        d_1 = d_1.reshape(-1) 
        PBI = d_1 = self.theta*d_2 # PBI with theta = 5    
        PBI = PBI.reshape(-1,1)

        return PBI[0]


class IPBI(Scalarisation):
    """
    Similar to PBI but inverts the final calculation. This is to improve diversity of solutions.
    Params:
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
        in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
        theta: multiplier that effects performance.
    """

    def __init__(self, ideal_point, max_point, theta=5) -> None:
        super().__init__()
        self.theta = theta
        self.ideal_point = ideal_point
        self.max_point = max_point

    
    def _do(self, f, weights):

        objs = [(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(2)]

        W = np.reshape(weights,(1,-1))
        normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
        normW = normW.reshape(-1,1)

        d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
        d_1 = d_1.reshape(-1,1)
        
        d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
        d_1 = d_1.reshape(-1) 
        PBI = self.theta*d_2 - d_1 # PBI with theta = 5    
        PBI = PBI.reshape(-1,1)

        return PBI[0]



