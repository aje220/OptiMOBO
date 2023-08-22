
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
    
    def __init__(self, ideal_point, max_point):
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point

    def _do(self, F, weights):
        



        obj = (np.asarray(F) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))

        if np.ndim(F) == 2:
            return np.sum(obj*weights, axis=1)
        else:
            return np.sum(obj*weights)

# def _do(self, F, weights):
#         if np.ndim(F) == 2:
#             aggre = np.zeros(len(F))
#             for count, value in enumerate(F):
#                 x = np.prod((value+100000)**weights, axis=0)
#                 aggre[count] = x
#             return aggre
#         else:
#             return np.prod((F+100000)**weights, axis=0)


class Tchebicheff(Scalarisation):
    """
    Tchebicheff takes two extra arguments when instantiated. 

    Params:
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
        in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.

    """
    def __init__(self, ideal_point, max_point):
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point

    # def _do(self, F, weights):
    #     if np.ndim(F) == 2:
    #         F_prime2 = (np.asarray(F) - np.asarray(self.ideal_point))/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
    #         hhh = np.max(weights*F_prime2, axis=1)
    #         return hhh
    #     else:
    #         F_prime = [(F[i]-self.ideal_point[i])/(self.max_point[i]-self.ideal_point[i]) for i in range(len(F))]
    #         return max([weights[i]*(F_prime[i]) for i in range(len(F))])
    def _do(self, F, weights):
        F_prime = (np.asarray(F) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))
        if np.ndim(F) == 2:
            hhh = np.max(weights * F_prime, axis=1)
            return hhh
        else:
            return np.max(weights * F_prime)

    
        


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

        # obj = F
        obj = (np.asarray(F) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))

        if np.ndim(F) == 2:
            v = np.abs(obj - self.ideal_point) * weights
            tchebi = v.max(axis=1) # add augemnted part to this
            aug = np.sum(np.abs(obj - self.ideal_point), axis=1)
            return tchebi + (self.alpha*aug)
        else:
            v = np.abs(obj - self.ideal_point) * weights
            tchebi = v.max(axis=0) # add augemnted part to this
            aug = np.sum(np.abs(obj - self.ideal_point), axis=0)
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
    
    def __init__(self, ideal_point, max_point, alpha=0.0001):
        super().__init__()
        self.alpha = alpha
        self.ideal_point = ideal_point
        self.max_point = max_point

    def _do(self, F, weights):

        # obj = F
        obj = (np.asarray(F) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))


        if np.ndim(F) == 2:
            left = np.abs(obj - self.ideal_point)
            # left = np.abs(obj - self.ideal_point / (np.asarray(self.max_point) - np.asarray(self.ideal_point)))

            right = self.alpha*(np.sum(np.abs(obj - self.ideal_point), axis=1))
            # right = self.alpha*(np.sum(np.abs((obj - self.ideal_point) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))), axis=1))
            # right = self.alpha*(np.sum(np.abs((obj - self.ideal_point) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))), axis=1))


            # total = (left + right)*weights
            total = (left + np.reshape(right, (-1,1)))*weights

            tchebi = total.max(axis=1)
            # import pdb; pdb.set_trace()
            return tchebi
        else:
            left = np.abs(obj - self.ideal_point)
            right = self.alpha*(np.sum(np.abs(obj - self.ideal_point)))
            total = (left + np.asarray(right))*weights
            tchebi = total.max(axis=0)
            return tchebi

class ExponentialWeightedCriterion(Scalarisation):
    """
    Improves on WeightedSum by enabling discovery of all solutions in non-convex problems.

    Params:
        p: can influence performance.
    """

    def __init__(self, ideal_point, max_point, p=100, **kwargs):
        super().__init__(**kwargs)
        self.ideal_point = ideal_point
        self.max_point = max_point
        self.p = p

    # def _do(self, F, weights):
        #     if np.ndim(F) == 2:
    #         aggre = np.zeros(len(F))
    #         for count, value in enumerate(F):
    #             x = np.sum(np.exp(self.p*weights - 1)*(np.exp(self.p*value)), axis=0)
    #             aggre[count] = x
    #         return aggre
    #     else:
    #         return np.sum(np.exp(self.p*weights - 1)*(np.exp(self.p*F)), axis=0)

    def _do(self, F, weights):

        if np.ndim(F) == 2:
            objs = (F - self.ideal_point)/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
        else:
            objs = [(F[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(len(F))]
        
        if np.ndim(F) == 2:
            return np.sum(np.exp(self.p*weights - 1)*(np.exp(self.p*objs)), axis=1)
        else:
            return np.sum(np.exp(self.p*weights - 1)*(np.exp(self.p*objs)))

        

class WeightedNorm(Scalarisation):
    """
    Generalised form of weighted sum.

    Params:
        p, infuences performance
    """

    def __init__(self, ideal_point, max_point, p=3) -> None:
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point
        self.p = p

    def _do(self, F, weights):

        if np.ndim(F) == 2:
            objs = (F - self.ideal_point)/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
        else:
            objs = [(F[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(len(F))]
        

        if np.ndim(F) == 2:
            return np.power(np.sum(np.power(np.abs(objs), self.p) * weights, axis=1), 1/self.p)
        else:
            return np.power(np.sum(np.power(np.abs(objs), self.p) * weights), 1/self.p)

        


class WeightedPower(Scalarisation):
    """
    Can find solutions in non-convex problems.
    Params:
        p: exponent, influences performance.
    """

    def __init__(self, ideal_point, max_point, p=3):
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point
        self.p = p

    def _do(self, F, weights):

        if np.ndim(F) == 2:
            objs = (F - self.ideal_point)/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
        else:
            objs = np.asarray([(F[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(len(F))])
        
        
        if np.ndim(F) == 2:
            return np.sum((objs**self.p) * weights, axis=1)
        else:
            return np.sum((objs**self.p) * weights)


class WeightedProduct(Scalarisation):
    """
    Can find solutions in non-convex problems.
    """

    def __init__(self, ideal_point, max_point):
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point
    

    def _do(self, F, weights):

        if np.ndim(F) == 2:
            objs = (F - self.ideal_point)/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
        else:
            objs = np.asarray([(F[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(len(F))])
        

        objs = objs + 100000  # Add 100000 to F outside the loop
        if np.ndim(F) == 2:
            return np.prod(objs ** weights, axis=1)
        else:
            return np.prod(objs ** weights)
        # this needs to be fixed
        

class PBI(Scalarisation):
    """
    First used as a measure of convergence in the evolutionary algorithm MOEA/D.
    Params:
        ideal_point: also known as the utopian point. is the smallest possible value of an objective vector
        in the objective space.
        max_point: the upper boundary of the objective space. The upper boundary for an objective vector.
        theta: multiplier that effects performance.
    """

    def __init__(self, ideal_point, max_point, theta=5):
        super().__init__()
        self.theta = theta
        self.ideal_point = ideal_point
        self.max_point = max_point

    
    def _do(self, f, weights):
    
        # if np.ndim(f) == 2:
        #     objs = (f - self.ideal_point)/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
        # else:
        #     objs = [(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(len(f))]
        
        objs = (np.asarray(f) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))

        # objs = f

        W = np.reshape(weights,(1,-1))
        normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
        normW = normW.reshape(-1,1)

        d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
        d_1 = d_1.reshape(-1,1)

        d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
        d_1 = d_1.reshape(-1) 
        PBI = d_1 + self.theta*d_2 # PBI with theta = 5    
        PBI = PBI.reshape(-1,1)

        return PBI


    # def _do(self, f, weights):
    #     objs = (np.asarray(f) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))

    #     W = np.asarray(weights)
    #     normW = np.linalg.norm(W, axis=1, keepdims=True)

    #     d_1 = np.sum(objs * (W / normW), axis=1, keepdims=True)
    #     d_2 = np.linalg.norm(objs - d_1 * (W / normW), axis=1, keepdims=True)
    #     PBI = d_1 + self.theta * d_2

    #     return PBI


    




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

        if np.ndim(f) == 2:
            objs = (f - self.ideal_point)/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
        else:
            objs = [(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(len(f))]
        
        W = np.reshape(weights,(1,-1))
        normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
        normW = normW.reshape(-1,1)

        d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
        d_1 = d_1.reshape(-1,1)
        
        d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
        d_1 = d_1.reshape(-1) 
        PBI = self.theta*d_2 - d_1 # PBI with theta = 5    
        PBI = PBI.reshape(-1,1)

        return PBI


class QPBI(Scalarisation):

    # more testing required.

    def __init__(self, ideal_point, max_point, theta=5, alpha=5.0, H=5.0) -> None:
        super().__init__()
        self.theta = theta
        self.ideal_point = ideal_point
        self.max_point = max_point
        self.alpha = alpha
        self.H = H

    
    def _do(self, f, weights):

        k = None
        # print(str(np.ndim(f)) + " "+ str(d_2))

        if np.ndim(f) == 2:
            k  = len(f[0])
            objs = (np.asarray(f) - self.ideal_point)/(np.asarray(self.max_point) - np.asarray(self.ideal_point))
            # print(str(np.ndim(f)) + " "+ str(objs))
            # print(f[0][0])
            # print(objs[0][0])
        else:
            k = len(f)
            objs = np.asarray([[(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(k)]])
            # objs = [(f[i]-np.asarray(self.ideal_point)[i])/(np.asarray(self.max_point)[i]-np.asarray(self.ideal_point)[i]) for i in range(len(f))]
            # print(str(np.ndim(f)) + " "+ str(objs))
            # print(f[0])
            # print(objs[0][0])

        # print(str(np.ndim(f)) + " objs "+ str(k))
        
       
        # objs = f

        W = np.reshape(weights,(1,-1))
        normW = np.linalg.norm(W, axis=1) # norm of weight vectors    
        normW = normW.reshape(-1,1)

        # print(str(np.ndim(f)) + " "+ str(W))
        # print(str(np.ndim(f)) + " "+ str(normW))


        d_1 = np.sum(np.multiply(objs,np.divide(W,normW)),axis=1)
        # import pdb; pdb.set_trace()
        d_1 = d_1.reshape(-1,1)

        d_2 = np.linalg.norm(objs - d_1*np.divide(W,normW),axis=1)
        
        # print(str(np.ndim(f)) + " "+ str(d_1))
        # print(str(np.ndim(f)) + " "+ str(d_2))

        d_1 = d_1.reshape(-1) 


        # import pdb; pdb.set_trace()
        d_star = self.alpha*(np.reciprocal(float(self.H))*np.reciprocal(float(k))*np.sum(np.asarray(self.max_point) - np.asarray(self.ideal_point)))
        
        # print(str(np.ndim(f)) + " "+ str(d_star))
        # print(str(np.ndim(f)) + " "+ str(d_1))
        # PBI with theta = 5    
        ret = d_1 + self.theta*d_2*(d_2/d_star)
        ret = np.reshape(ret, (-1,1))
        return ret





class APD(Scalarisation):
    
    # def __init__(self, ideal_point, max_point, FE, FE_max, gamma):
    #     super().__init__()
    #     self.ideal_point = ideal_point
    #     self.max_point = max_point
    #     self.FE = FE
    #     self.FE_max = FE_max
    #     self.gamma = gamma
    
    def __init__(self, ideal_point, max_point, FE=1, FE_max=10, gamma=0.010304664101210016):
        super().__init__()
        self.ideal_point = ideal_point
        self.max_point = max_point
        self.FE = FE
        self.FE_max = FE_max
        self.gamma = gamma
    
    def _unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    # def _do(self, FE,FE_max,gamma, trans_f,norm_trans_f,w_vector,w_vector_index):
    def _do(self, f, w_vector):
        
        # obj = f
        # obj = (np.asarray(f) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point))


        if np.ndim(f) == 2:
            # trans_f = np.asarray(f) - np.asarray(self.ideal_point)
            trans_f = (np.asarray(f) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point)) # to be used in APD
        else:
            # trans_f = np.asarray([f]) - np.asarray(self.ideal_point) # to be used in APD
            trans_f = (np.asarray([f]) - np.asarray(self.ideal_point)) / (np.asarray(self.max_point) - np.asarray(self.ideal_point)) # to be used in APD

            
        # import pdb; pdb.set_trace()
        norm_trans_f = np.linalg.norm(trans_f,axis=1) # to be used in APD
        norm_trans_f = norm_trans_f.reshape(-1,1) # to be used in APD

        theta = np.zeros((trans_f.shape[0],1))
        for i  in range(0,trans_f.shape[0]):
        #for j in range(0,len(weight_vectors)):
            if np.all(trans_f[i,:]==0):
                trans_f[i,:] = np.tile(np.array([1e-5]),(1,trans_f.shape[1]))
            if np.all(w_vector==0):
                w_vector = np.tile(np.array([1e-5]),(1,trans_f.shape[1]))
            theta[i] = self._angle_between(trans_f[i,:],w_vector)
        ratio_angles = theta/self.gamma
        apd = (1 + trans_f.shape[1]*(self.FE/self.FE_max)*ratio_angles)*norm_trans_f
        return apd



