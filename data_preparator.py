from sklearn.model_selection import train_test_split

def prepare(dataset, random_state=None, stratify=None):
    # Split the dataset between features X and labels y (y is the last column)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    # Split into train and test set, possibly in a reproductible way (set seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state, stratify=stratify)
    return X_train, X_test, y_train, y_test