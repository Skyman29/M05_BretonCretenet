import argparse
import numpy as np
import re
from data_preparator import prepare


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
        "-p",
        "--preprocessing",
        type=str,
        choices=["standardize", "minmax", "robust"],
        default="standardize",
        help="Choose the preprocessing method to use.",
    )
    parser.add_argument(
        "-deg",
        "--degree", 
        type=int, 
        default=2, 
        help="Degree for polynomial preprocessing."
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

    # Get the chosen dataset 
    
    # Initialize an empty numpy array to store data
    data = np.array([])

    for dataset in DATASETS[args.dataset]:
        temp_data = parse_file(dataset[0])
        
        
        # Concatenate the new data with the existing data
        data = np.concatenate((temp_data, data), axis=0)
    data_label = get_data_label(dataset[1])
    
    X_train_, X_test, y_train, y_test = prepare(data, random_state=args.rs)
    # ... do something with the dataset ...

# Pre processor
def parse_file(filepath):
    if filepath.endswith('.csv'):
        data = np.loadtxt(filepath, delimiter=';', skiprows=1)
    else:
        data = np.loadtxt(filepath)
    return data

# Pre processor
def get_data_label(filepath):
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
