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

# Function that predict y with given x input and model
def PredictFromRegressor(model, X, X_labels):

    # Check if Input Data correspond to model parameters, i.e features numbers, order
    if all(feature in model.feature_names_in_ for feature in X_labels):
        return model.predict(X)
    else:
        df_X_predict = pd.DataFrame(X, columns = X_labels)
        df_X_predict = df_X_predict[model.feature_names_in_]
        return model.predict(df_X_predict)
    
def LassoRegression_FeatureSelection(X_train, y_train, X_train_labels):

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