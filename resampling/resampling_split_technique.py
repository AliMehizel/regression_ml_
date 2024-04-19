import numpy as np
import random 



def random_split(X, y, test_prop=0.3, seed_num=None):
    """This will split our data into training set & test set .
    
    params:
    --------
    X: ndarray|dataframe object
      model features matrix with shape (n,p)
      
    y: ndarray|dataframe object
      vector container labled data (output) shape (n,1) 
    
    test_prop : float
      default is 0.3
      
    seed_num : int
      define number for seed function (default None)
      
    ---------
    
    return X_train, y_train, X_test, y_test
     
    """
    
    """
    1- check shape
    2- check data type 
    3- validate data if there is any error na value ?
    
    """

    data_size = len(y)
    test_size = int(test_prop * data_size)
    
    random.seed(seed_num)
    test_idx = np.random.choice(data_size,test_size, replace=False)
    
    X_,y_ = np.ones(data_size,dtype=bool), np.ones(data_size,dtype=bool)

    #used to select (subset un-indexed value)
    X_[test_idx] = False
    y_[test_idx] = False
    
    return X[X_],y[y_],X[test_idx],y[test_idx]

def stratified_split():
    """we are waiting here."""
    pass


def k_fold_cv():
    """here we need to build class that inherit fropm regression classes and fit each k fold & test it"""
    pass

