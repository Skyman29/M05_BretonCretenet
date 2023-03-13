# algorithm

# This module contains all the function in order to build the machine learning algorithm of the project

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor

def linear_regression_algorithm(X_train, y_train,  X_train_labels):
    """
    Fit a linear regression model to the training data.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training input data of shape (n_samples, n_features).
    y_train : numpy.ndarray
        Target values of shape (n_samples,).
    X_train_labels : list
        List of strings representing the feature names.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        A fitted linear regression model.
    """

    # assert type(X_train) == np.ndarray # Check if data input is an acceptable format i.e {array-like, sparse matrix} of shape (n_samples, n_features)
    # assert type(y_train) == np.ndarray # Check if data input is an acceptable format i.e array-like of shape (n_samples,) or (n_samples, n_targets)
    # assert len(X_train) == len(y_train) # Check if data input have the same amount of samples

    # Switching to DataFrame so X train labels are stored in model
    df_X_train = pd.DataFrame(X_train, columns = X_train_labels)

    regressor = LinearRegression()
    regressor.fit(df_X_train, y_train)

    return regressor

def decision_tree_regressor_algorithm(X_train, y_train,  X_train_labels, max_depth = 2):
    """
    Fit a decision tree regression model to the training data.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training input data of shape (n_samples, n_features).
    y_train : numpy.ndarray
        Target values of shape (n_samples,).
    X_train_labels : list
        List of strings representing the feature names.
    max_depth : int, optional (default=2)
        The maximum depth of the decision tree.

    Returns
    -------
    sklearn.tree.DecisionTreeRegressor
        A fitted decision tree regression model.
    """

    # assert type(X_train) == np.ndarray # Check if data input is an acceptable format i.e {array-like, sparse matrix} of shape (n_samples, n_features)
    # assert type(y_train) == np.ndarray # Check if data input is an acceptable format i.e array-like of shape (n_samples,) or (n_samples, n_targets)
    # assert len(X_train) == len(y_train) # Check if data input have the same amount of samples

    # Switching to DataFrame so X train labels are stored in model
    df_X_train = pd.DataFrame(X_train, columns = X_train_labels)

    regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=0) # random_state = 0 to stick to the same random seed
    regressor.fit(df_X_train, y_train)

    return regressor

def predict_from_regressor(model, X, X_labels):
    """
    Predict the target values for new input data using a given regression model.

    Parameters
    ----------
    model : sklearn estimator
        A fitted regression model.
    X : numpy.ndarray
        Input data of shape (n_samples, n_features).
    X_labels : list
        List of strings representing the feature names.

    Returns
    -------
    numpy.ndarray
        Predicted target values of shape (n_samples,).
    """

    # Check if Input Data correspond to model parameters, i.e features numbers, order
    if all(feature in model.feature_names_in_ for feature in X_labels):
        return model.predict(X)
    else:
        df_X_predict = pd.DataFrame(X, columns = X_labels)
        df_X_predict = df_X_predict[model.feature_names_in_]
        return model.predict(df_X_predict)
    
def lasso_regression_feature_selection(X_train, y_train, X_train_labels):
    """
    Apply Lasso regression feature selection to the training data.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training input data of shape (n_samples, n_features).
    y_train : numpy.ndarray
        Target values of shape (n_samples,).
    X_train_labels : list
        List of strings representing the feature names.

    Returns
    -------
    tuple
        A tuple containing the selected training input data of shape (n_samples, n_selected_features)
        and a list of strings representing the names of the selected features.
        If the number of training samples is less than or equal to 50, the function returns
        the original input data and feature names unchanged.
    """

    # assert type(X_train) == np.ndarray # Check if data input is an acceptable format i.e {array-like, sparse matrix} of shape (n_samples, n_features)
    # assert type(y_train) == np.ndarray # Check if data input is an acceptable format i.e array-like of shape (n_samples,) or (n_samples, n_targets)
    # assert len(X_train) == len(y_train) # Check if data input have the same amount of samples
    # assert len(X_train) == len(X_train_labels)  # Check if data input have the same amount of samples
    
    #Cannot apply cv on low amount of sample, just return input as output
    if(len(X_train)>50):
        cv = 5
    else:
        return X_train, X_train_labels

    cv = GridSearchCV(Lasso(),
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = cv, 
                      scoring="neg_mean_absolute_error",
                      verbose=3
                      )
    
    cv.fit(X_train, y_train)

    # print(cv.best_params_)
    coefficients = cv.best_estimator_.named_steps['model'].coef_
    X_train_labels_selected = np.array(X_train_labels)[np.abs(coefficients) > 0]
    X_train_selected = np.array(X_train)[np.abs(coefficients) > 0]

    return X_train_selected, X_train_labels_selected