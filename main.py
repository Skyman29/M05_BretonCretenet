import argparse
import numpy as np
import pandas as pd
import re
from data_preparator import prepare
from data_preprocessor import preprocess
from algorithm import linear_regression_algorithm, decision_tree_regressor_algorithm, lasso_regression_feature_selection, predict_from_regressor, score


def main():
    # Define available datasets
    DATASETS = {
        "housing": (r"./data/housing.data", r"./data/housing.names"),
        "white": (r"./data/winequality-white.csv", r"./data/winequality.names"),
        "red": (r"./data/winequality-red.csv", r"./data/winequality.names"),
        "red+white": [
            (r"./data/winequality-white.csv", r"./data/winequality.names"),
            (r"./data/winequality-red.csv", r"./data/winequality.names"),
        ],
        "white+red": [
            (r"./data/winequality-white.csv", r"./data/winequality.names"),
            (r"./data/winequality-red.csv", r"./data/winequality.names"),
        ],
    }

    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Prepare and preprocess a dataset.")
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
        "--random-state",
        type=int,
        default=None,
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
        "--feature-selection",
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
    
    # Initialize an empty numpy array to store data
    data = np.array([])

    for dataset in DATASETS[args.dataset]:
        temp_data = parse_file(dataset[0])
        
        
        # Concatenate the new data with the existing data
        data = np.concatenate((temp_data, data), axis=0)
    data_label = get_data_label(dataset[1])
    X_train_labels = data_label[:-1]
    
    X_train, X_test, y_train, y_test = prepare(data, random_state=args.rs)

    #Polynomial
    X_train, X_test = preprocess(X_train, X_test, method='poly', degree=args.degree)
    #Scaling
    X_train, X_test = preprocess(X_train, X_test, method=args.preprocessing)

    #Feature selection
    if args.fs:
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

# Pre processor

def parse_file(filepath):
    """
    Loads data from a file into a numpy array.

    Parameters
    ----------
    filepath : str
        The path to the file to be loaded.

    Returns
    -------
    data : numpy.ndarray
        A numpy array containing the data from the file.

    Raises
    ------
    ValueError
        If the file is not in a valid format.

    Notes
    -----
    The function uses the numpy `loadtxt` method to load data from a file. If the file has a '.csv' extension,
    the `delimiter` parameter is set to ';', and `skiprows` is set to 1 to skip the header row.
    """

    if filepath.endswith('.csv'):
        data = np.loadtxt(filepath, delimiter=';', skiprows=1)
    else:
        data = np.loadtxt(filepath)

    return data


# Pre processor
def get_data_label(filepath):
    """
    Extracts the column names from a file containing attribute information for a dataset.

    Parameters
    ----------
    filepath : str
        The path to the file containing attribute information.

    Returns
    -------
    list
        A list of strings representing the column names extracted from the file.

    Raises
    ------
    IOError
        If the file does not exist or cannot be opened for reading.

    Notes
    -----
    The function reads the file line by line and searches for a section labeled "Attribute Information",
    which indicates the start of the section containing the column names. The function then extracts the
    column names using a regular expression pattern that matches lines of the form "n - column_name",
    where "n" is a positive integer representing the index of the column and "column_name" is the name
    of the column.

    The function stops extracting column names when it encounters a section labeled "Missing Attribute Values",
    which indicates the end of the section containing the column names.
    """

    # Open the file for reading
    with open('file.names', 'r') as f:

        # Read the file line by line
        lines = f.readlines()

        # Initialize a flag to indicate when to start and stop appending lines
        start_flag = False
        end_flag = False

        # Initialize an empty list to store the column names
        column_names = []

        # Define a regular expression pattern to match column names
        pattern = r'^\d+\s*-\s*(.+)$'

        # Loop over each line in the file
        for line in lines:

            # Check if the current line contains the start flag
            if 'Attribute Information' in line:
                start_flag = True
                continue

            # Check if the current line contains the end flag
            if 'Missing Attribute Values' in line:
                end_flag = True
                break

            # If the start flag is True and the end flag is False, append the line to the list
            if start_flag and not end_flag:
                match = re.match(pattern, line)
                if match:
                    column_names.append(match.group(1))
        return column_names


if __name__ == "__main__":
    main()
