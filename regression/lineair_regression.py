from .score_cum import *
from .utils import *
import numpy as pd
import numpy as np 





class LineairRegression(Score):
    
    """Linear regression task.
    
    ------
    Params (constructor):
    ------
    
    intercept: bool. 
    params: ndarray estimated coefficient.
    
    
    
    
    """
    def __init__(self,intercept=True):
        self.intercept = intercept
        self.params = None
        
    def fit(self, X, y):
        
        """Fit the regression model.
        
        ------
        Params:
        ------
        X: numpy.ndarray (n,p)
        y: numpy.ndarray (n,1)
        
        
        -in this task we are using standard OLS estimator.
        """
        
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
            
   
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        
        """Estimate or predict data points.
    
        ------
        Params:
        ------
        
        X: numpy.ndarray (n,p)

        
        Returns:
         int (insatance) (1,p)*(p,1) or ndarray (n,p)*(p,1)
        """
        
        return X @ self.params
        
    def get_score(self, X, y):
        
        """This function are inherted from Score class with some modification.
        
        ------
        Params:
        ------
        X: numpy.ndarray (n,p)
        y: numpy.ndarray (n,1)
        
        
        Returns:
         dict: all performance indicator for linear regression
        """
        y_estimated = self.predict(X)
        
        super().get_score(y,y_estimated)
        
        

        
    
        
        