import numpy as np
from data_preparator import prepare
from data_preprocessor import preprocess


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
    X = np.random.rand(10,5)
    y = np.random.rand(10,1)
    return np.concatenate([X,y], axis = 1)


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
    X_train, X_test, y_train, y_test = prepare(dataset)
    # After preparation, data should not be in the exact same order as in the begining
    assert not np.allclose(np.concatenate([X_train, X_test], axis=0), dataset[:, :-1], atol=1e-12)
    assert not np.allclose(np.concatenate([y_train, y_test], axis=0), dataset[:, -1], atol=1e-12)
    # Without seeds, two preparations should give different results
    X_train2, X_test2, y_train2, y_test2 = prepare(dataset)
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
    X_train, X_test, y_train, y_test = prepare(dataset, random_state=99)
    X_train2, X_test2, y_train2, y_test2 = prepare(dataset, random_state=99)
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
    X_train, X_test, y_train, y_test = prepare(dataset, random_state=99)
    # Check that the X and y are shuffled but still correspond y_train[i] must correspond to X_train[i, :]
    for i in range(len(y_train)):
        assert np.allclose(X_train[i, :], X[y == y_train[i], :], atol=1e-12)
    for i in range(len(y_test)):
        assert np.allclose(X_test[i, :], X[y == y_test[i], :], atol=1e-12)


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
    X_train = np.random.rand(15,5)
    X_test = np.random.rand(10,5)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_standardized = (X_train-mean)/std
    X_test_standardized = (X_test-mean)/std
    X_train_check, X_test_check = preprocess(X_train, X_test, method='standardize')
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
    X_train = np.random.rand(15,5)
    X_test = np.random.rand(10,5)
    mininmum = np.min(X_train, axis=0)
    maximum = np.max(X_train, axis=0)
    X_train_minmax = (X_train-mininmum)/(maximum-mininmum)
    X_test_minmax = (X_test-mininmum)/(maximum-mininmum)
    X_train_check, X_test_check = preprocess(X_train, X_test, method='minmax')
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
    X_train = np.random.rand(15,5)
    X_test = np.random.rand(10,5)
    median = np.median(X_train, axis=0)
    interquartile = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)
    X_train_robust = (X_train-median)/interquartile
    X_test_robust = (X_test-median)/interquartile
    X_train_check, X_test_check = preprocess(X_train, X_test, method='robust')
    print('MAX', np.max(np.abs(X_train_check-X_train_robust)))
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
    X = np.random.rand(10,2)
    bias = np.ones((10,1))
    col1 = X[:, 0].reshape(-1,1)
    col2 = X[:, 1].reshape(-1,1)
    X_poly = np.concatenate([bias, col1, col2, col1**2, col1*col2, col2**2, col1**3, col1**2 * col2, col1 * col2**2, col2**3], axis=1) 
    X_check, _ = preprocess(X, X, method='poly', degree=3)
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
    X_train = np.random.rand(15,5)
    X_test = np.random.rand(10,5)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_standardized = (X_train-mean)/std
    X_test_standardized = (X_test-mean)/std
    X_train_check, X_test_check = preprocess(X_train, X_test, method='wrong_name') # Should work like standardize
    assert np.allclose(X_train_check, X_train_standardized, atol=1e-12)
    assert np.allclose(X_test_check, X_test_standardized, atol=1e-12)