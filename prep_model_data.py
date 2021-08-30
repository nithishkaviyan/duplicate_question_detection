"""
    This script contains code to prepare modelling data for duplicate question detection
"""

# Import packages
from pathlib import Path
import pandas as pd
import numpy as np


def split_train_val_test_data(dataset_size: int, train_size: float = 0.7, val_size: float = 0.15, state: int = 123456):
    """Function to return train, val and test indices

    :param dataset_size: Int Length of the dataset
    :param train_size: Float Train set size
    :param val_size: Float Validation set size
    :param state: Int containing state to set seed

    :return: Dict containing train, val and test indices

    """
    if train_size > 1 or train_size < 0 or val_size > 1 or val_size < 0:
        raise AssertionError("train_size and val_size must be between 0 and 1")
    if state is not None:
        np.random.seed(state)

    # Create index with respect to dataset size
    index = np.arange(dataset_size)

    # Shuffle the dataset
    np.random.shuffle(index)
    train_last_index = int(train_size * dataset_size)
    val_last_index = int((train_size + val_size) * dataset_size)

    data_index = dict()
    data_index['train_index'] = index[:train_last_index]
    data_index['val_index'] = index[train_last_index:val_last_index]
    data_index['test_index'] = index[val_last_index:]

    return data_index


# Read question data
data = pd.read_csv(Path.cwd().joinpath('data', 'train.csv'))

# Subset only records with is_duplicate == 1
data = data.loc[data.is_duplicate == 1].reset_index(drop=True)

# Get indices for train, val and test set
data_indices = split_train_val_test_data(len(data))

# Split train, val and test set
train_data = data.loc[data_indices['train_index']]
val_data = data.loc[data_indices['val_index']]
test_data = data.loc[data_indices['test_index']]

# Save the above dfs to csv file
train_data.to_csv('data/model_train_data.csv', index_label=False)
val_data.to_csv('data/model_val_data.csv', index_label=False)
test_data.to_csv('data/model_test_data.csv', index_label=False)
