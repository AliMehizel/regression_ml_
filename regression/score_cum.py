import numpy as np 



class Score:
    """Score data object.
    
    ------
    Params (constructor):
    ------

    
    """
    def __init__(self):
        self.score_ = {
            'RMSE': None,
            'MSE': None,
            'MAE': None,
            'R_squared': None
        }
        
    def get_score(self ,y , y_estimated):
        """Calculate performance measure.
        
        ------
        Params:
        ------
        y: numpy.ndarray (n,1)
        y_estimated: numpy.ndarray (n,1)
        
        
        Returns:
         dict: all performance indicator for linear regression
        """
        
        """use tabulate package to present data on terminal!!!!!"""
        y_diff = y-y_estimated
        m = len(y)
        self.score_['RMSE'] = np.sqrt((1/m)* np.sum(y_diff**2))
        
        self.score_['MAE'] = (1/m) * np.sum(np.abs(y_diff))
        
        self.score_['MSE'] = (1/m)* np.sum(y_diff**2)
        
        self.score_['R_squared'] = np.sum((y_estimated - np.mean(y_estimated))**2) / np.sum((y - np.mean(y))**2)


      
        