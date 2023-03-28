from sklearn.model_selection import train_test_split
import re
import urllib.request
import os
import numpy as np
import pandas as pd


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

def load_data(input):
    """
    Load data from a CSV or fixed-width file into a NumPy array.

    Parameters
    ----------
    input : str
        The path to the input file. Can be a local file path or a URL.

    Returns
    -------
    numpy.ndarray
        A 2-dimensional NumPy array containing the loaded data.

    Raises
    ------
    ValueError
        If the input file format is not recognized.

    Notes
    -----
    This function reads data from CSV or fixed-width files using the pandas library,
    and converts it to a NumPy array. The file format is determined based on the file
    extension: ".csv" files are assumed to be comma-separated, while ".data" files are
    assumed to be fixed-width. The header of the file is ignored.
    """
    # check if input is URL or local path
    if input.endswith('.csv'):
        return pd.read_csv(input, delimiter= ';').to_numpy()
    elif input.endswith('.data'):
        return pd.read_fwf(input, header=None).to_numpy()
    else:
        raise ValueError("File format not recognized")

def get_data_column_names(input):
    """
    Returns the names of columns in a data file.

    Parameters
    ----------
    input : str
        A URL or local path to a data file.

    Returns
    -------
    list
        A list of column names.

    Raises
    ------
    ValueError
        If the file does not exist.

    Notes
    -----
    This function detects the type of data file by checking its extension and
    assumes that the file is either a CSV or a text file with space-separated
    values. It then reads the file and returns the column names.
    """
    # check if input is URL or local path
    if input.startswith('http'):
        with urllib.request.urlopen(input) as file:
            return detect_column_names_from_file(file)
            
    else:
        # check if file exists
        if not os.path.isfile(input):
            raise ValueError("File does not exist")
        
        # determine if file is CSV or text with space-separated values
        with open(input, 'r') as file:
            return detect_column_names_from_file(file)
            
def detect_column_names_from_file(file):
    """
    Given a file, this function reads the file line by line and detects the column names in it.
    
    Args:
    - file: a file object, representing the file to be read.
    
    Returns:
    - column_names: a list of strings, containing the column names detected in the file.
    """

    # Initialize a flag to indicate when to start and stop appending lines
    start_flag = False
    end_flag = False

    # Initialize an empty list to store the column names
    column_names = []

    # Define a regular expression pattern to match column names
    pattern = r'\d{1,2}\s*[.-]\s*([a-zA-Z]+)\s*'

    # Loop over each line in the file
    for b_line in file:
        str_line = b_line.decode('utf-8')

        # Check if the current line contains the start flag
        if 'attribute information' in str_line.lower():
            start_flag = True
            continue

        # Check if the current line contains the end flag
        if 'missing attribute values' in str_line.lower():
            end_flag = True
            break

        # If the start flag is True and the end flag is False, append the line to the list
        if start_flag and not end_flag:
            match = re.search(pattern, str_line)
            if match:
                column_names.append(match.group(1))
    return column_names