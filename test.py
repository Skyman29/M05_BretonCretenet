import numpy as np
from sklearn.linear_model import LinearRegression

from data_preparator import prepare
from algorithm import LinearRegression_Algorithm


def rand_data():
    X = np.random.rand(10,5)
    y = np.random.rand(10,1)
    return np.concatenate([X,y], axis = 1)


def test_preparator_is_random_if_no_seed():
    dataset = rand_data()
    X_train, X_test, y_train, y_test = prepare(dataset)
    # After preparation, data should not be in the exact same order as in the begining
    assert not (np.concatenate([X_train, X_test], axis=0) == dataset[:, :-1]).all()
    assert not (np.concatenate([y_train, y_test], axis=0) == dataset[:, -1]).all()
    # Without seeds, two preparations should give different results
    X_train2, X_test2, y_train2, y_test2 = prepare(dataset)
    assert not (X_train == X_train2).all()
    assert not (y_train == y_train2).all()
    assert not (X_test == X_test2).all()
    assert not (y_test == y_test2).all()


def test_preparator_with_seed():
    dataset = rand_data()
    X_train, X_test, y_train, y_test = prepare(dataset, random_state=99)
    X_train2, X_test2, y_train2, y_test2 = prepare(dataset, random_state=99)
    # With the same seed, both preparations should be identical
    assert (X_train == X_train2).all()
    assert (y_train == y_train2).all()
    assert (X_test == X_test2).all()
    assert (y_test == y_test2).all()


def test_preparator_xy_alignement():
    dataset = rand_data()
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X_train, X_test, y_train, y_test = prepare(dataset, random_state=99)
    # Check that the X and y are shuffled but still correspond y_train[i] must correspond to X_train[i, :]
    for i in range(len(y_train)):
        assert (X_train[i, :] == X[y == y_train[i], :]).all()
    for i in range(len(y_test)):
        assert (X_test[i, :] == X[y == y_test[i], :]).all()

def test_LinearRegression_Algorithm():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([10, 20, 30])
    X_train_labels = ['feature1', 'feature2']
    model = LinearRegression_Algorithm(X_train, y_train, X_train_labels)
    assert isinstance(model, LinearRegression)