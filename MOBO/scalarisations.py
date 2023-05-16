
import numpy as np

def weighted_norm(F, weights, p=3):
    return (np.sum(np.abs((F**p)) * weights, axis=1))**(1/p)

def weighted_power(F, weights, p=3):
    return np.sum((F**p) * weights, axis=1)

def weighted_product(F, weights):
    return np.prod(F**weights, axis=1)

def modified_tchebicheff(F, weights, ideal_point, alpha=0.001):
    
    left = np.abs(F - ideal_point)
    right = alpha*(np.sum(np.abs(F - ideal_point)))

    total = (left + np.asarray(right))*weights
    tchebi = total.max(axis=1)
    return tchebi

def augmented_tchebicheff(F, weights, ideal_point, alpha=0.001):
    
    v = np.abs(F - ideal_point) * weights
    tchebi = v.max(axis=1) # add augemnted part to this
    aug = np.sum(np.abs(F - ideal_point), axis=1)
    return tchebi + (alpha*aug)

def tchebicheff(f, W, ideal_point, max_point):
    
    nobjs = 2

    objs = [(f[i]-ideal_point[i])/(max_point[i]-ideal_point[i]) for i in range(nobjs)]

    return max([W[i]*(objs[i]) for i in range(nobjs)])

def PBI(f, W, ideal_point, max_point, theta):
    """
    Penalty-Boundary Intersection.
    """
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
    PBI = d_1 + theta*d_2 # PBI with theta = 5    
    PBI = PBI.reshape(-1,1)
    return PBI[0]

def IPBI(f, W, ideal_point, max_point, theta):
    """
    Inverted Penalty-Boundary Intersection.
    """
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
    PBI = theta*d_2 - d_1
    PBI = PBI.reshape(-1,1)
    return PBI[0]


def exponential_weighted_criterion(self, F, weights, p=3):
    return np.sum(np.exp(p*weights - 1)*(np.exp(p*F)), axis=1)


