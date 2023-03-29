import argparse
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from algorithm import (
    decision_tree_regressor_algorithm,
    lasso_regression_feature_selection,
    linear_regression_algorithm,
    predict_from_regressor,
    score,
)
from data_preparator import get_data_column_names, load_data, prepare
from data_preprocessor import preprocess, preprocess_polynomialfeatures

# Define available datasets
DATASETS = {
    "housing": [
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names",
        ]
    ],
    "white": [
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
        ]
    ],
    "red": [
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
        ]
    ],
    "red+white": [
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
        ],
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
        ],
    ],
}


def get_args(args=None):
    # Define CLI arguments
    parser = argparse.ArgumentParser(
        prog="ML_Model_Trainer_M05_Cretenet_Breton",
        description="A script to train and evaluate machine learning models on a given dataset. The script applies polynomial feature expansion, scaling, and feature selection to the dataset before fitting the model(s) and generating evaluation scores.",
        epilog="",
    )
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

        Default:housing
        """,
    )
    parser.add_argument(
        "-rs",
        "--random_state",
        type=int,
        default=42,
        help="Random state for train/test split. Default = 42",
    )
    parser.add_argument(
        "-deg",
        "--degree",
        type=int,
        default=2,
        help="Degree for polynomial preprocessing. Default = 2",
    )
    parser.add_argument(
        "-p",
        "--preprocessing",
        type=str,
        choices=["standardize", "minmax", "robust"],
        default="standardize",
        help="""Choose the preprocessing method to use. default:standardize """,
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
        help="""Choose algorithm for regression analysis default:linear""",
    )
    parser.add_argument(
        "-md",
        "--max_depth",
        type=int,
        default=2,
        help="Choose max depth for DecisionTreeRegressor() default = 2",
    )
    parser.add_argument(
        "-vb",
        "--verbose",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="""Choose to display information during the workflow
        1:Minimalist (only the output)
        2:Medium (mention steps)
        3:Maximum""",
    )

    # Parse the arguments
    return parser.parse_args(args)


def main(  # noqa: C901 A lot of if statement due to verbose raise a complexity in flake8, it should be reduce
    args_test=None,
):
    """The function executes a machine learning workflow on a given dataset, performs feature engineering, feature selection, model training, and model evaluation.

    Parameters
    ----------
    args_test (list, optional): list of argument for testing the package. Defaults to None.
    """
    args = get_args(args_test)
    # main workflow
    if args.verbose > 1:
        print("\nLoading the dataset...")
    data = load_data(DATASETS[args.dataset][0][0])

    # Continue iteration if multiple dataset to concatenate i.e wine
    for dataset in DATASETS[args.dataset][1:]:
        temp_data = load_data(dataset[0])
        # Concatenate the new data with the existing data
        data = np.concatenate((data, temp_data))
    data_label = get_data_column_names(DATASETS[args.dataset][0][1])
    X_train_labels = data_label[:-1]
    if args.verbose > 1:
        print("Dataset loaded\n")
        if args.verbose > 2:
            print(tabulate(data[:6, :], headers=data_label), "\n")
        print("Splitting the dataset...")

    X_train, X_test, y_train, y_test = prepare(data, random_state=args.random_state)

    # Polynomial
    if args.verbose > 1:
        print(
            "Applying polynomial feature expansion of degree {} to the dataset\n".format(
                args.degree
            )
        )
    X_test, _ = preprocess_polynomialfeatures(
        X_test, X_train_labels, degree=args.degree
    )
    X_train, X_train_labels = preprocess_polynomialfeatures(
        X_train, X_train_labels, degree=args.degree
    )
    if args.verbose > 2:
        if X_train.shape[1] > 15:
            print_table = X_train[:6, :15]
        else:
            print_table = X_train[:6, :]
        print(tabulate(print_table, headers=X_train_labels), "\n")

    # Scaling
    if args.verbose > 1:
        print("Applying {} scaling to the dataset\n".format(args.preprocessing))
    X_train, X_test = preprocess(X_train, X_test, method=args.preprocessing)
    if args.verbose > 2:
        if X_train.shape[1] > 15:
            print_table = X_train[:6, :16]
        else:
            print_table = X_train[:6, :]
        print(tabulate(print_table, headers=X_train_labels), "\n")

    # Feature selection
    if args.feature_selection:
        if args.verbose > 1:
            print("Features selection using Lasso Regression ongoing...")
        X_train, X_train_labels, X_test = lasso_regression_feature_selection(
            X_train, y_train, X_train_labels, X_test, args.verbose
        )
        if args.verbose > 2:
            print("The selected features are :\n", X_train_labels, "\n")

    # Fitting
    models = {}
    if args.algorithm in ["linear", "both"]:
        if args.verbose > 1:
            print("Fitting LinearRegression()...")
        models["linear"] = {
            "model": linear_regression_algorithm(X_train, y_train, X_train_labels)
        }
        if args.verbose > 1:
            print("LinearRegression() fitted")
    if args.algorithm in ["tree", "both"]:
        if args.verbose > 1:
            print("Fitting DecisionTreeRegressor()...")
        models["tree"] = {
            "model": decision_tree_regressor_algorithm(
                X_train,
                y_train,
                X_train_labels,
                max_depth=args.max_depth,
                random_state=args.random_state,
            )
        }
        if args.verbose > 1:
            print("DecisionTreeRegressor() fitted")

    # Scoring
    if args.verbose > 1:
        print("Model(s) being evaluated on test set")
    for model_ref, model_data in models.items():
        y_predict_train = predict_from_regressor(
            model_data["model"], X_train, X_train_labels
        )
        y_predict_test = predict_from_regressor(
            model_data["model"], X_test, X_train_labels
        )

        models[model_ref]["score_train"] = score(y_train, y_predict_train)
        models[model_ref]["score_test"] = score(y_test, y_predict_test)

    # Output
    df_print = pd.DataFrame(models)
    print(
        "Result of the machine learning model(s), Mean absolute error:\n",
        tabulate(df_print, tablefmt="fancy_grid"),
    )


if __name__ == "__main__":
    main()
