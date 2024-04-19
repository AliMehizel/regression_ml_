import numpy as np



def with_intercept_(X):
    
    """Add intercept if True.
    
    ------
    Params:
    ------
    X: numpy.ndarray (n,p)
    
    
    Returns:
      ndarray (n,p+1)
    
    """
    
    len_ = len(X)
    intercept_ = np.ones((len_,1),dtype=int)

    try:
        X = np.concatenate((intercept_, X),axis=1)
    except ValueError:
        X = np.vstack((intercept_.T, X)).T

    
    return X 


def check_size_(X,y):
    
    """Check the size of labeled and features data.
    
    ------
    Params:
    ------
    X: numpy.ndarray (n,p)
    y: numpy.ndarray (n,1)
    
    Returns:
      Boolean
    
    """
    
    len_x = len(X)
    len_y = len(y)
    
    if len_x == len_y:
        return True 
    else:
        return False 
    
    
def check_nan_(X,y):
    
    """Check missed value.
    
    ------
    Params:
    ------
    X: numpy.ndarray (n,p)
    y: numpy.ndarray (n,1)
    
    Returns:
      Boolean
    
    """
    x_ = np.isnan(X).sum()
    y_ = np.isnan(y).sum()
    
    if x_ == 0 and y_ == 0:
        return True
    else:
        return False 
    
    
"""Here we have some function to have a control on learning rate."""


def linear_dec(lambda_0, iter):
    
    """Decrease linearily."""
    
    return lambda_0 / (iter + 1)

def quadratic_dec(lambda_0, iter):
    
    """Decrease qaudraticaly."""
    
    return lambda_0 / (iter + 1)**2 

def exp_dec(lambda_0, iter, beta=1):
    
    """Exponentiel decrease."""
    
    return lambda_0 * np.exp(-beta*iter)