from sklearn.model_selection import train_test_split


def prepare(dataset, random_state=None, stratify=None):
    """
    Creates a training and a test set from the features X and labels y in dataset.

    Parameters
    ----------
    dataset : numpy.ndarray
        Dataset of shape (n_samples, n_features), with labels in the last columns and features in the other columns.
    random_state : int, optional
        Seed chosen for the train test split. If no argument is given, the seed is not fixed.
    stratify : list, optional
        If not None, the dataset is split in a stratified fashion, using this as the class labels.

    Returns
    -------
    numpy.ndarray
        X_train, an array containing the features of the training set.
    numpy.ndarray
        X_test, an array containing the features of the test set.
    numpy.ndarray
        y_train, an array containing the labels of the training set.
    numpy.ndarray
        y_test, an array containing the labels of the test set.    
    """
    # Split the dataset between features X and labels y (y is the last column)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    # Split into train and test set, possibly in a reproductible way (set seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test
