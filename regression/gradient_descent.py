import numpy as np 
import pandas as pd
from .utils import *
from .lineair_regression import *


""" 
Gradient descent is a generic optimization algorithm capable of finding optimal solutions
to a wide range of problems. The general idea of gradient descent is to tweak
parameters iteratively in order to minimize a cost function.
"""
class BGDRegressor(LineairRegression):
    
    """Stochastic gradient descent picks a random instance in the training set 
       at every step and computes the gradients based only on that single instance.
    
    ------
    Params (constructor):
    ------
    
    intercept: bool. 
    params: ndarray estimated coefficient.
    
    
    
    -This class inherit from Linear Regression class.
    """
    def __init__(self,intercept=True):
        super().__init__(intercept)


    def fit(self,X,y,beta_0=None,learning_rate=0.001,max_iter = 1000):
        
        """Fit the regression model.
        
        ------
        Params:
        ------
        X: numpy.ndarray (n,p)
        y: numpy.ndarray (n,1)
        
        
        -in this task we are using Batch gradient descent to reach the optimum params.
        """
        if beta_0 is None:
            raise Exception('you need to initialize params.')
        
        
       
        
       
        if isinstance(X, (pd.DataFrame,pd.Series)):
            X = X.to_numpy()
        
        
        if isinstance(X, pd.Series):
            y = y.to_numpy()
        

        if self.intercept:
            X = with_intercept_(X)
        
        size_ = check_size_(X, y)    
        if not size_:
            """try to customize exception in one class!!!!!!"""
            raise Exception('Two input data does not have the same size !!!')
    
        na_value = check_nan_(X,y)
        if not na_value:
            raise Exception('Can not fit the model, data contain some missed value.')
        
        

        learning_rate = learning_rate
        l_r_0 = learning_rate
        
     

        for i in range(max_iter):
            
            gradient = (2/len(y)) * X.reshape(3,1) @ (X @ beta_0 - y)
            beta_1 = beta_0 - (learning_rate * gradient)
            beta_0 = beta_1 
           
            learning_rate = quadratic_dec(l_r_0,i)
            
        self.params = np.round(beta_0,2)
        
      
        
        


class SGDRegressor(LineairRegression):
    
    """Stochastic gradient descent picks a random instance in the training set 
       at every step and computes the gradients based only on that single instance.
    
    ------
    Params (constructor):
    ------
    
    intercept: bool. 
    params: ndarray estimated coefficient.
    
    
    
    -This class inherit from Linear Regression class.
    """
    
    def __init__(self,intercept=True):
        super().__init__(intercept)


    def fit(self,X,y,beta_0=None,learning_rate=0.001,max_iter = 1000):

        if beta_0 is None:
            raise Exception('you need to initialize params.')
        
        
       
        
       
        # if isinstance(X, (pd.DataFrame,pd.Series)):
        #     X = X.to_numpy()
        
        
        # if isinstance(X, pd.Series):
        #     y = y.to_numpy()
        

        if self.intercept:
            X = with_intercept_(X)
        
    
        size_ = check_size_(X, y)    
        if not size_:
            """try to customize exception in one class!!!!!!"""
            raise Exception('Two input data does not have the same size !!!')
    
        na_value = check_nan_(X,y)

        if not na_value:
            raise Exception('Can not fit the model, data contain some missed value.')
        
        
        

        learning_rate = learning_rate
        l_r_0 = learning_rate
        
 

        for i in range(max_iter):
            i =  np.random.choice(len(y))
            gradient = (2/len(y)) * X[i:i+1].T @ (X[i:i+1] @ beta_0-y[i:i+1] )
       
            beta_1 = beta_0 - (learning_rate * gradient)
            
        
            beta_0 = beta_1
            
          
            learning_rate = quadratic_dec(l_r_0,i)
            
       
        self.params = np.round(beta_0,2)
        
    



class MBGDRegressor:
    """mini-batch GD compute the gradients on small random sets of instances called mini-batches."""
    """will be implemented"""
    pass
