# algorithm

# This module contains all the function in order to build the machine learning algorithm of the project

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def linear_regression_algorithm(X_train, y_train, X_train_labels, y_train_label):
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
    y_train_label : str
        string representing the feature name.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        A fitted linear regression model.
    """

    # assert type(X_train) == np.ndarray # Check if data input is an acceptable format i.e {array-like, sparse matrix} of shape (n_samples, n_features)
    # assert type(y_train) == np.ndarray # Check if data input is an acceptable format i.e array-like of shape (n_samples,) or (n_samples, n_targets)
    # assert len(X_train) == len(y_train) # Check if data input have the same amount of samples

    # Switching to DataFrame so X train, y train labels are stored in model
    df_X_train = pd.DataFrame(X_train, columns=X_train_labels)
    df_y_train = pd.DataFrame(y_train, columns=[y_train_label])

    regressor = LinearRegression()
    regressor.fit(df_X_train, df_y_train)

    return regressor


def decision_tree_regressor_algorithm(
    X_train, y_train, X_train_labels, y_train_label, max_depth=2
):
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
    y_train_label : str
        string representing the feature name.
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
    df_X_train = pd.DataFrame(X_train, columns=X_train_labels)
    df_y_train = pd.DataFrame(y_train, columns=[y_train_label])

    regressor = DecisionTreeRegressor(
        max_depth=max_depth, random_state=0
    )  # random_state = 0 to stick to the same random seed
    regressor.fit(df_X_train, df_y_train)

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
    df_X_predict = pd.DataFrame(X, columns=X_labels)
    df_X_predict = df_X_predict[model.feature_names_in_]
    return model.predict(df_X_predict)


def lasso_regression_feature_selection(
    X_train, y_train, X_train_labels, y_train_label, X_test
):
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
    y_train_label : str
        string representing the feature name.
    X_test : numpy.ndarray
        Training input data of shape (n_samples, n_features).

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

    # Cannot apply cv on low amount of sample, just return input as output
    if len(X_train) > 500:
        cv = 5
    else:
        return X_train, X_train_labels

    cv = GridSearchCV(
        Lasso(),
        {"alpha": [0.01, 0.1, 1.0, 5.0, 10.0]},
        cv=cv,
        scoring="neg_mean_absolute_error",
        verbose=3,
    )

    df_X_train = pd.DataFrame(X_train, columns=X_train_labels)
    df_y_train = pd.DataFrame(y_train, columns=[y_train_label])
    df_X_test = pd.DataFrame(X_test, columns=X_train_labels)

    cv.fit(df_X_train, df_y_train)

    # print(cv.best_params_)
    coefficients = cv.best_estimator_.coef_
    X_train_labels_selected = np.array(X_train_labels)[np.abs(coefficients) > 0]
    X_train_selected = df_X_train[X_train_labels_selected]
    X_test_selected = df_X_test[X_train_labels_selected]

    return X_train_selected.to_numpy(), X_train_labels_selected, X_test_selected


def score(y_true, y_predict):
    """Calculate the mean absolute error (MAE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Correct target values.
    y_predict : np.ndarray
        Estimated target values.

    Returns
    -------
    float
        Mean absolute error between `y_true` and `y_predict`.

    Examples
    --------
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_predict = np.array([2.5, 0.0, 2, 8])
    >>> score(y_true, y_predict)
    0.5
    """
    return mean_absolute_error(y_true, y_predict)
