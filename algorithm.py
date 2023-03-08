# algorithm

# This module contains all the function in order to build the machine learning algorithm of the project

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

# Funtion that call linear regressor method and return it fitted to the data
def LinearRegression_Algorithm(X_train, y_train,  X_train_labels):

    # assert type(X_train) == np.ndarray # Check if data input is an acceptable format i.e {array-like, sparse matrix} of shape (n_samples, n_features)
    # assert type(y_train) == np.ndarray # Check if data input is an acceptable format i.e array-like of shape (n_samples,) or (n_samples, n_targets)
    # assert len(X_train) == len(y_train) # Check if data input have the same amount of samples

    # Switching to DataFrame so X train labels are stored in model
    df_X_train = pd.DataFrame(X_train, columns = X_train_labels)

    regressor = LinearRegression()
    regressor.fit(df_X_train, y_train)

    return regressor

# Funtion that call Decision Tree regressor method and return it fitted to the data
def DecisionTreeRegressor_Algorithm(X_train, y_train,  X_train_labels, max_depth = 2):

    # assert type(X_train) == np.ndarray # Check if data input is an acceptable format i.e {array-like, sparse matrix} of shape (n_samples, n_features)
    # assert type(y_train) == np.ndarray # Check if data input is an acceptable format i.e array-like of shape (n_samples,) or (n_samples, n_targets)
    # assert len(X_train) == len(y_train) # Check if data input have the same amount of samples

    # Switching to DataFrame so X train labels are stored in model
    df_X_train = pd.DataFrame(X_train, columns = X_train_labels)

    regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=0) # random_state = 0 to stick to the same random seed
    regressor.fit(df_X_train, y_train)

    return regressor