from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, RobustScaler


def preprocess(X_train, X_test, method='standardize', degree=2):
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