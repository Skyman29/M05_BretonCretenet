import argparse
import numpy as np
import urllib.request
import os
import pandas as pd
import re
from data_preparator import prepare, load_data, get_data_column_names
from data_preprocessor import preprocess
from algorithm import linear_regression_algorithm, decision_tree_regressor_algorithm, lasso_regression_feature_selection, predict_from_regressor, score

def main():
    # Define available datasets
    DATASETS = {
        'housing': [['https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', 
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names']],
        'white': [['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', 
                  'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names']],
        'red': [['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', 
                'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names']],
        'red+white': [
            ['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
             'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names'],
            
            ['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
             'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names'],
        ],
        'white+red': [
            ['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
             'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names'],
            
            ['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
             'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names'],
        ],
    }

    # Define CLI arguments
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='Python package to build machine learning model',
                    epilog='')
    parser.add_argument(
        "-d",
        "--dataset",
        choices=DATASETS.keys(),
        default="housing",
        help="""Choose the dataset to use.
        "housing" : Boston House Prices
        "white" : white wine quality
        "red" : red wine quality
        "red+white" : red & white wine quality
        """,
    )
    parser.add_argument(
        "-rs",
        "--random_state",
        type=int,
        default=42,
        help="Random state for train/test split.",
    )
    parser.add_argument(
        "-deg",
        "--degree", 
        type=int, 
        default=2, 
        help="Degree for polynomial preprocessing."
    )
    parser.add_argument(
        "-p",
        "--preprocessing",
        type=str,
        choices=["standardize", "minmax", "robust"],
        default="standardize",
        help="Choose the preprocessing method to use.",
    )
    parser.add_argument(
        "-fs",
        "--feature_selection",
        action="store_true",
        default=False,
        help="Whether to perform feature selection using Lasso regression.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="linear",
        choices=["linear", "tree", "both"],
        help="Choose algorithm for regression analysis",
    )
    parser.add_argument(
        "-md",
        "--max_depth",
        type=int,
        default=2,
        help="Choose max depth for DecisionTreeRegressor()",
    )
    

    # Parse the arguments
    args = parser.parse_args()

    # main workflow
    data = load_data(DATASETS[args.dataset][0][0])

    # Continue iteration if multiple dataset to concatenate i.e wine
    for dataset in DATASETS[args.dataset][1:]:
        temp_data = load_data(dataset[0])
        # Concatenate the new data with the existing data
        data = np.concatenate((data, temp_data))
        
    data_label = get_data_column_names(DATASETS[args.dataset][0][1])
    X_train_labels = data_label[:-1]
    
    X_train, X_test, y_train, y_test = prepare(data, random_state=args.random_state)

    #Polynomial
    # X_train = pd.DataFrame(X_train, columns = X_train_labels)
    X_train, X_test = preprocess(X_train, X_test, method='poly', degree=args.degree)
    #Scaling
    X_train, X_test = preprocess(X_train, X_test, method=args.preprocessing)

    #Feature selection
    if args.feature_selection:
        X_train, data_label = lasso_regression_feature_selection(X_train, y_train, X_train_labels)

    models = {}
    if args.algorithm == "linear":
        models["linear"] = {'model':linear_regression_algorithm(X_train, y_train, X_train_labels)}
    elif args.algorithm == "tree":
        models["tree"] = {'model':decision_tree_regressor_algorithm(X_train, y_train, X_train_labels, max_depth = args.max_depth)}
    elif args.algorithm == "both":
        models["linear"] = {'model':linear_regression_algorithm(X_train, y_train, X_train_labels)}
        models["tree"] = {'model':decision_tree_regressor_algorithm(X_train, y_train, X_train_labels, max_depth = args.max_depth)}
    
    for model_ref, model_data in models.items():
        y_predict_train = predict_from_regressor(model_data['model'],X_train, X_train_labels)
        y_predict_test = predict_from_regressor(model_data['model'],X_test, X_train_labels)

        model_ref['score_train'] = score(y_train, y_predict_train)
        model_ref['score_test'] = score(y_test, y_predict_test)

    df_print = pd.DataFrame(models)
    print(df_print)

    # ... do something with the dataset ...

if __name__ == "__main__":
    main()
