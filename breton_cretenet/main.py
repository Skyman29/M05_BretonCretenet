import argparse
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from . import algorithm, data_preparator, data_preprocessor

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
    df_dict = {}
    random_state_list = [
        args.random_state,
        2 * args.random_state,
        3 * args.random_state,
    ]
    # main workflow
    for random_state in random_state_list:
        data = data_preparator.load_data(DATASETS[args.dataset][0][0], args.verbose)

        # Continue iteration if multiple dataset to concatenate i.e wine
        for dataset in DATASETS[args.dataset][1:]:
            temp_data = data_preparator.load_data(dataset[0])
            # Concatenate the new data with the existing data
            data = np.concatenate((data, temp_data))
        data_label = data_preparator.get_data_column_names(DATASETS[args.dataset][0][1])
        X_train_labels = data_label[:-1]
        if args.verbose > 1:
            print("Dataset loaded\n")
            if args.verbose > 2:
                print(tabulate(data[:6, :], headers=data_label), "\n")
            print("Splitting the dataset...")

        X_train, X_test, y_train, y_test = data_preparator.prepare(
            data, random_state=random_state
        )

        # Polynomial
        X_test, _ = data_preprocessor.preprocess_polynomialfeatures(
            X_test, X_train_labels, degree=args.degree, verbose=args.verbose
        )
        X_train, X_train_labels = data_preprocessor.preprocess_polynomialfeatures(
            X_train, X_train_labels, degree=args.degree
        )
        if args.verbose > 2:
            if X_train.shape[1] > 15:
                print_table = X_train[:6, :15]
            else:
                print_table = X_train[:6, :]
            print(tabulate(print_table, headers=X_train_labels), "\n")

        # Scaling
        X_train, X_test = data_preprocessor.preprocess(
            X_train, X_test, method=args.preprocessing
        )
        if args.verbose > 2:
            if X_train.shape[1] > 15:
                print_table = X_train[:6, :16]
            else:
                print_table = X_train[:6, :]
            print(tabulate(print_table, headers=X_train_labels), "\n")

        # Feature selection
        if args.feature_selection:
            (
                X_train,
                X_train_labels,
                X_test,
            ) = algorithm.lasso_regression_feature_selection(
                X_train, y_train, X_train_labels, X_test, args.verbose
            )

        # Fitting the dataset
        models = {}
        if args.algorithm in ["linear", "both"]:
            models["linear"] = {
                "model": algorithm.linear_regression_algorithm(
                    X_train, y_train, X_train_labels, verbose=args.verbose
                )
            }

        if args.algorithm in ["tree", "both"]:
            models["tree"] = {
                "model": algorithm.decision_tree_regressor_algorithm(
                    X_train,
                    y_train,
                    X_train_labels,
                    max_depth=args.max_depth,
                    random_state=random_state,
                    verbose=args.verbose,
                )
            }

        # Scoring
        if args.verbose > 1:
            print("Model(s) being evaluated on test set")
        for model_ref, model_data in models.items():
            y_predict_train = algorithm.predict_from_regressor(
                model_data["model"], X_train, X_train_labels
            )
            y_predict_test = algorithm.predict_from_regressor(
                model_data["model"], X_test, X_train_labels
            )

            models[model_ref]["score_train"] = algorithm.score(y_train, y_predict_train)
            models[model_ref]["score_test"] = algorithm.score(y_test, y_predict_test)

        # Output

        df_dict[f"df_{random_state}"] = pd.DataFrame(models)
    df_print_total = pd.concat(
        df_dict.values(), keys=[f"RS:{rs}" for rs in random_state_list]
    )
    print(
        "Result of the machine learning model(s), Mean absolute error:\n",
        tabulate(df_print_total, tablefmt="fancy_grid"),
    )
    print(
        "Result of the machine learning model(s), Mean absolute error:\n",
        tabulate(df_print_total.T, tablefmt="fancy_grid"),
    )


if __name__ == "__main__":
    main()
