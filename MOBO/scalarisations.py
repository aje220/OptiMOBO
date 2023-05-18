
import numpy as np
import matplotlib.pyplot as plt

class Scalarisation:
    """
    Parent class for all.
    """
    def __init__(self):
        return

    def __call__(self, *args, **kwargs):
        return self.do(*args, **kwargs)

    def do(self, F, weights, **args):
        D = self._do(F, weights, **args).flatten()
        return D


class AugmentedTchebicheff(Scalarisation):
    
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
    

class ExponentialWeightedCriterion(Scalarisation):
    """
    Exponential Weighted Criterion
    """
    def __init__(self, p=100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p = p


    def _do(self, F, weights, **kwargs):
        return np.sum(np.exp(self.p*weights - 1)*(np.exp(self.p*F)), axis=0)

class ModifiedTchebicheff(Scalarisation):
    

    def __init__(self, ideal_point, max_point, alpha=0.0001) -> None:
        super().__init__()
        self.alpha = alpha
        self.ideal_point = ideal_point
        self.max_point = max_point

    def _do(self, F, weights):

        left = np.abs(F - self.ideal_point)
        right = self.alpha*(np.sum(np.abs(F - self.ideal_point)))

        sum = (left + np.asarray(right))*weights
        tchebi = sum.max(axis=0)
        return tchebi

class WeightedNorm(Scalarisation):
    """
    Weighted Norm
    """

    def __init__(self, p=3) -> None:
        super().__init__()
        self.p = p

    def _do(self, F, weights):
        # import pdb; pdb.set_trace()
        return np.sum([np.abs(F[i])**self.p * weights[i] for i in range(len(F))])**(1/self.p)

class WeightedPower(Scalarisation):
    """
    Weighted Power
    """

    def __init__(self, p=3) -> None:
        super().__init__()
        self.p = p

    def _do(self, F, weights):

        # p = 3
        return np.sum((F**self.p) * weights, axis=0)

class WeightedProduct(Scalarisation):
    """
    Weighted product.
    """

    def _do(self, F, weights):
        # this needs to be fixed
        return np.prod((F+100000)**weights, axis=0)

class IPBI(Scalarisation):

    def __init__(self, ideal_point, max_point, theta=5) -> None:
        super().__init__()
        self.theta = theta
        self.ideal_point = ideal_point
        self.max_point = max_point

    
    def _do(self, f, weights):
        # import pdb; pdb.set_trace()
        objs = [(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(2)]
        
        # trans_f = f - f_ideal # translated objective values 
        # print(W)
        # W = [0.5,0.5]
        # import pdb; pdb.set_trace()

        W = np.reshape(weights,(1,-1))
        normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
        normW = normW.reshape(-1,1)
        # import pdb; pdb.set_trace()
        d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
        d_1 = d_1.reshape(-1,1)
        
        # import pdb; pdb.set_trace()

        d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
        d_1 = d_1.reshape(-1) 
        PBI = self.theta*d_2 - d_1 # PBI with theta = 5    
        PBI = PBI.reshape(-1,1)
        # import pdb; pdb.set_trace()
        return PBI[0]

class PBI(Scalarisation):
    
    def __init__(self, ideal_point, max_point, theta=5) -> None:
        super().__init__()
        self.theta = theta
        self.ideal_point = ideal_point
        self.max_point = max_point

    
    def _do(self, f, weights):
        # import pdb; pdb.set_trace()
        objs = [(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(2)]
        
        # trans_f = f - f_ideal # translated objective values 
        # print(W)
        # W = [0.5,0.5]
        # import pdb; pdb.set_trace()

        W = np.reshape(weights,(1,-1))
        normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
        normW = normW.reshape(-1,1)
        # import pdb; pdb.set_trace()
        d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
        d_1 = d_1.reshape(-1,1)
        
        # import pdb; pdb.set_trace()

        d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
        d_1 = d_1.reshape(-1) 
        PBI = d_1 = self.theta*d_2 # PBI with theta = 5    
        PBI = PBI.reshape(-1,1)
        # import pdb; pdb.set_trace()
        return PBI[0]


class Tchebicheff(Scalarisation):

    def __init__(self, ideal_point, max_point) -> None:
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point

    def _do(self, F, weights):
        F_prime = [(F[i]-self.ideal_point[i])/(self.max_point[i]-self.ideal_point[i]) for i in range(len(F))]
        return max([weights[i]*(F_prime[i]) for i in range(len(F))])
