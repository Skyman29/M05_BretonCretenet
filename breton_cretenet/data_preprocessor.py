import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)


def preprocess(X_train, X_test, method="standardize", verbose=1):
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
    verbose : int, optional
        Verbosity level of information, by default 1.

    Returns
    -------
    numpy.ndarray
        An array containing the preprocessed features of the training set.
    numpy.ndarray
        An array containing the preprocessed features of the test set.
    """
    if verbose > 1:
        print(f"Applying {method} scaling to the dataset\n")
    # Select  preprocessor
    if method == "standardize":
        preprocessor = StandardScaler()
    elif method == "minmax":
        preprocessor = MinMaxScaler()
    # elif(method == 'poly'):
    #     preprocessor = PolynomialFeatures(degree=degree)
    elif method == "robust":
        preprocessor = RobustScaler()
    else:
        print(
            "WARNING :  'method' can only be set to 'standardize' or 'minmax'.\n",
            "No valid method was selected, 'standard' is selected by default.",
        )
        preprocessor = StandardScaler()
    # Preprocess features
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)
    return X_train_pp, X_test_pp


def preprocess_polynomialfeatures(data, data_column_names, degree=2, verbose=1):
    """
    Applies polynomial feature expansion to a Numpy array of data, and returns the resulting array and the names of the
    columns in the expanded feature matrix.

    Parameters:
    -----------
    data : numpy.ndarray
        The input array of data to be expanded.
    data_column_names : list
        A list of the names of the columns in the input data array.
    degree : int, optional (default=2)
        The degree of the polynomial features to generate.
    verbose : int, optional
        Verbosity level of information, by default 1.

    Returns:
    --------
    tuple : (numpy.ndarray, list)
        A tuple containing two elements:
            1. A Numpy array representing the expanded feature matrix.
            2. A list of the names of the columns in the expanded feature matrix.
    """
    if verbose > 1:
        print(
            f"Applying polynomial feature expansion of degree {degree} to the dataset\n"
        )

    df_data = pd.DataFrame(data, columns=data_column_names)
    preprocessor = PolynomialFeatures(degree=degree, include_bias=True)

    df_data_expanded = preprocessor.fit_transform(df_data)
    data_column_names_expanded = preprocessor.get_feature_names_out(data_column_names)

    return df_data_expanded, data_column_names_expanded
