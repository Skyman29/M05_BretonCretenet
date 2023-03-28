from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, RobustScaler


def preprocess(X_train, X_test, method='standardize', degree=2):
    """
    Creates a training and a test set from the features X and labels y in dataset.

    Parameters
    ----------
    X_train : numpy.ndarray
        Array containing the features of the training set.
    X_test : numpy.ndarray
        Array containing the features of the test set.
    method : string, optional
        Selects the preprocessing method we want to apply, if None selected, then "standardize" is chosen by default.
    degree : int, optional
        Selects the degree of the polynomial features. Is only used if the method is "poly".

    Returns
    -------
    numpy.ndarray
        An array containing the preprocessed features of the training set.
    numpy.ndarray
        An array containing the preprocessed features of the test set.   
    """
    # Select  preprocessor
    if(method == 'standardize'):
        preprocessor = StandardScaler()
    elif(method == 'minmax'):
        preprocessor = MinMaxScaler()
    elif(method == 'poly'):
        preprocessor = PolynomialFeatures(degree=degree)
    elif(method == 'robust'):
        preprocessor = RobustScaler()
    else:
        print("WARNING :  'method' can only be set to 'standardize', 'minmax', or 'poly'.\n",
            "No valid method was selected, 'standard' is selected by default.")
        preprocessor = StandardScaler()
    # Preprocess features
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)
    return X_train_pp, X_test_pp