import os

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from . import algorithm, data_preparator, data_preprocessor, main


def rand_data():
    """
    A function that returns a random dataset for the tests.

    Parameters:
    -----------
    None

    Returns:
    --------
    numpy.ndarray
        An array of size (10, 6) with random features and labels
    """
    X = np.random.rand(10, 5)
    y = np.random.rand(10, 1)
    return np.concatenate([X, y], axis=1)


def test_preparator_is_random_if_no_seed():
    """
    Test function to ensure that the preparator returns random splits.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    dataset = rand_data()
    X_train, X_test, y_train, y_test = data_preparator.prepare(dataset)
    # After preparation, data should not be in the exact same order as in the begining
    assert not np.allclose(
        np.concatenate([X_train, X_test], axis=0), dataset[:, :-1], atol=1e-12
    )
    assert not np.allclose(
        np.concatenate([y_train, y_test], axis=0), dataset[:, -1], atol=1e-12
    )
    # Without seeds, two preparations should give different results
    X_train2, X_test2, y_train2, y_test2 = data_preparator.prepare(dataset)
    assert not np.allclose(X_train, X_train2, atol=1e-12)
    assert not np.allclose(y_train, y_train2, atol=1e-12)
    assert not np.allclose(X_test, X_test2, atol=1e-12)
    assert not np.allclose(y_test, y_test2, atol=1e-12)


def test_preparator_with_seed():
    """
    Test function to ensure that the preparator gives fixed splits if the seed is set.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    dataset = rand_data()
    X_train, X_test, y_train, y_test = data_preparator.prepare(dataset, random_state=99)
    X_train2, X_test2, y_train2, y_test2 = data_preparator.prepare(
        dataset, random_state=99
    )
    # With the same seed, both preparations should be identical
    assert np.allclose(X_train, X_train2, atol=1e-12)
    assert np.allclose(y_train, y_train2, atol=1e-12)
    assert np.allclose(X_test, X_test2, atol=1e-12)
    assert np.allclose(y_test, y_test2, atol=1e-12)


def test_preparator_xy_alignement():
    """
    Test function to ensure that the preparator keeps the features and the labels grouped correctly after the shuffling.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    dataset = rand_data()
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X_train, X_test, y_train, y_test = data_preparator.prepare(dataset, random_state=99)
    # Check that the X and y are shuffled but still correspond y_train[i] must correspond to X_train[i, :]
    for i in range(len(y_train)):
        assert np.allclose(X_train[i, :], X[y == y_train[i], :], atol=1e-12)
    for i in range(len(y_test)):
        assert (X_test[i, :] == X[y == y_test[i], :]).all()


def test_linear_regression_algorithm():
    """
    Test function to ensure that the linear_regression_algorithm function returns an instance of the LinearRegression
    class.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([10, 20, 30])
    X_train_labels = ["feature1", "feature2"]
    model = algorithm.linear_regression_algorithm(X_train, y_train, X_train_labels)
    assert isinstance(model, LinearRegression)


def test_decision_tree_regressor_algorithm():
    """
    Test function to ensure that the decision_tree_regressor_algorithm function returns an instance of the DecisionTreeRegressor
    class.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([10, 20, 30])
    X_train_labels = ["feature1", "feature2"]
    max_depth = 2
    model = algorithm.decision_tree_regressor_algorithm(
        X_train, y_train, X_train_labels, max_depth
    )
    assert isinstance(model, DecisionTreeRegressor)


def test_predict_from_regressor():
    """
    Test function to ensure that the predict_from_regressor function returns an array of predictions with the same
    length as the input array.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([10, 20, 30])
    X_train_labels = ["feature1", "feature2"]
    model = algorithm.linear_regression_algorithm(X_train, y_train, X_train_labels)

    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_labels = ["feature1", "feature2"]
    y_predicted = algorithm.predict_from_regressor(model, X, X_labels)
    assert isinstance(y_predicted, np.ndarray)
    assert y_predicted.shape == (len(X),)


def test_lasso_regression_feature_selection():
    """
    Test the `lasso_regression_feature_selection` function.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    X_train = np.array([[1, 2, 0], [2, 4, 0], [3, 6, 0]])
    X_test = 3 * X_train
    y_train = np.array([10, 20, 30])
    X_train_labels = ["feature1", "feature2", "feature3"]
    (
        X_train_selected,
        X_train_labels_selected,
        X_test,
    ) = algorithm.lasso_regression_feature_selection(
        X_train, y_train, X_train_labels, X_test
    )
    assert isinstance(X_train_selected, np.ndarray)
    assert isinstance(X_train_labels_selected, list)


def test_preprocessor_standard():
    """
    Test function to ensure that the standard method of the preprocessor is correctly implemented.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X_train = np.random.rand(15, 5)
    X_test = np.random.rand(10, 5)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std
    X_train_check, X_test_check = data_preprocessor.preprocess(
        X_train, X_test, method="standardize"
    )
    assert np.allclose(X_train_check, X_train_standardized, atol=1e-12)
    assert np.allclose(X_test_check, X_test_standardized, atol=1e-12)


def test_preprocessor_minmax():
    """
    Test function to ensure that the MinMax method of the preprocessor is correctly implemented.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X_train = np.random.rand(15, 5)
    X_test = np.random.rand(10, 5)
    mininmum = np.min(X_train, axis=0)
    maximum = np.max(X_train, axis=0)
    X_train_minmax = (X_train - mininmum) / (maximum - mininmum)
    X_test_minmax = (X_test - mininmum) / (maximum - mininmum)
    X_train_check, X_test_check = data_preprocessor.preprocess(
        X_train, X_test, method="minmax"
    )
    assert np.allclose(X_train_check, X_train_minmax, atol=1e-12)
    assert np.allclose(X_test_check, X_test_minmax, atol=1e-12)


def test_preprocessor_robust():
    """
    Test function to ensure that the robust scaler method of the preprocessor is correctly implemented.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X_train = np.random.rand(15, 5)
    X_test = np.random.rand(10, 5)
    median = np.median(X_train, axis=0)
    interquartile = np.percentile(X_train, 75, axis=0) - np.percentile(
        X_train, 25, axis=0
    )
    X_train_robust = (X_train - median) / interquartile
    X_test_robust = (X_test - median) / interquartile
    X_train_check, X_test_check = data_preprocessor.preprocess(
        X_train, X_test, method="robust"
    )
    print("MAX", np.max(np.abs(X_train_check - X_train_robust)))
    assert np.allclose(X_train_check, X_train_robust, atol=1e-12)
    assert np.allclose(X_test_check, X_test_robust, atol=1e-12)


def test_preprocessor_polynomial():
    """
    Test function to ensure that the Polynomial Features method of the preprocessor is correctly implemented.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X = np.random.rand(10, 2)
    bias = np.ones((10, 1))
    col1 = X[:, 0].reshape(-1, 1)
    col2 = X[:, 1].reshape(-1, 1)
    X_poly = np.concatenate(
        [
            bias,
            col1,
            col2,
            col1**2,
            col1 * col2,
            col2**2,
            col1**3,
            col1**2 * col2,
            col1 * col2**2,
            col2**3,
        ],
        axis=1,
    )
    X_check, _ = data_preprocessor.preprocess_polynomialfeatures(
        X, ["x1", "x2"], degree=3
    )
    assert np.allclose(X_check, X_poly, atol=1e-12)


def test_preprocessor_inexistant_method():
    """
    Test function to ensure that the if no existing method is selected, then the standardization is applied.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    X_train = np.random.rand(15, 5)
    X_test = np.random.rand(10, 5)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std
    X_train_check, X_test_check = data_preprocessor.preprocess(
        X_train, X_test, method="wrong_name"
    )  # Should work like standardize
    assert np.allclose(X_train_check, X_train_standardized, atol=1e-12)
    assert np.allclose(X_test_check, X_test_standardized, atol=1e-12)


@pytest.mark.parametrize(
    "input, expected_shape",
    [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            (506, 14),
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            (4898, 12),
        ),
    ],
)
def test_load_data(input, expected_shape):
    """
    Test the `load_data` function of the `data_preparator` module using parameterized inputs.

    Parameters:
    ----------
    input : str
        The URL of the data file to load.
    expected_shape : tuple of int
        The expected shape of the NumPy array returned by the `load_data` function.

    Raises:
    ------
    ValueError
        If the `load_data` function is called with an invalid URL.

    Returns:
    -------
    None
    """
    data = data_preparator.load_data(input)
    assert isinstance(data, np.ndarray)
    assert data.shape == expected_shape

    # Test a nonexistent file
    with pytest.raises(ValueError):
        data_preparator.get_data_column_names(
            "https://upload.wikimedia.org/wikipedia/commons/1/14/KHThisIsFine.jpg"
        )


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names",
            [
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
                "MEDV",
            ],
        ),
        (
            "data/housing.names",
            [
                "CRIM",
                "ZN",
                "INDUS",
                "CHAS",
                "NOX",
                "RM",
                "AGE",
                "DIS",
                "RAD",
                "TAX",
                "PTRATIO",
                "B",
                "LSTAT",
                "MEDV",
            ],
        ),
        ("data/nonexistent.csv", None),
    ],
)
def test_get_data_column_names(input_data, expected_output):
    """
    Test the get_data_column_names function from the data_preparator module.

    Parameters
    ----------
    input_data : str
        The input data to pass to get_data_column_names. This can be a URL or a local file path.
    expected_output : list or None
        The expected output of get_data_column_names when called with input_data. If input_data is an invalid file path,
        this should be None.

    Raises
    ------
    ValueError
        If get_data_column_names is called with an invalid file path and expected_output is None.

    Returns
    -------
    None
    """
    if expected_output is None:
        with pytest.raises(ValueError):
            data_preparator.get_data_column_names(input_data)
    else:
        assert data_preparator.get_data_column_names(input_data) == expected_output


@pytest.mark.pull_request
@pytest.mark.parametrize("dataset", ["housing", "white", "red+white"])
@pytest.mark.parametrize("random_state", [42, 123])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("preprocessing", ["standardize", "minmax"])
@pytest.mark.parametrize("feature_selection", [True, False])
@pytest.mark.parametrize("algorithm", ["linear", "both"])
@pytest.mark.parametrize("max_depth", [2, 3])
def test_main_pull_request(
    dataset,
    random_state,
    degree,
    preprocessing,
    feature_selection,
    algorithm,
    max_depth,
):
    """
    Test function for the main method in the codebase.

    Parameters
    ----------
    dataset : str
        Name of the dataset to use for testing.
    random_state : int
        Seed value for the random number generator.
    degree : int
        Degree of the polynomial features to generate.
    preprocessing : str
        Type of preprocessing to apply to the data.
    feature_selection : bool
        Whether or not to perform feature selection.
    algorithm : str
        Type of algorithm to use for testing.
    max_depth : int
        Maximum depth of the decision tree for testing.

    Returns
    -------
    None
        This function does not return anything.

    Raises
    ------
    AssertionError
        If the test fails.
    """

    args = [
        "-d",
        dataset,
        "-rs",
        str(random_state),
        "-deg",
        str(degree),
        "-p",
        preprocessing,
        "-a",
        algorithm,
        "-md",
        str(max_depth),
        "-v",
        str(1),
    ]
    if feature_selection:
        args.append("-fs")
    main.main(args)


@pytest.mark.parametrize("dataset", ["housing", "red+white"])
def test_main(dataset):
    """
    Test function for the main method in the codebase.

    Parameters
    ----------
    dataset : str
        Name of the dataset to use for testing.

    Returns
    -------
    None
        This function does not return anything.

    Raises
    ------
    AssertionError
        If the test fails.
    """
    args = [
        "-d",
        dataset,
        "-rs",
        str(42),
        "-deg",
        str(2),
        "-p",
        "standardize",
        "-a",
        "both",
        "-md",
        str(2),
        "-v",
        str(3),
        "-fs",
    ]
    main.main(args)
