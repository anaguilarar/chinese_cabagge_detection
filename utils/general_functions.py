import random
import numpy as np


def get_ids_split_datasets(list_images_path, val_perc=20, shuffle=True, seed=123, test_perc=None):
    n_paths = len(list_images_path)
    list_idx = list(range(n_paths))
    if shuffle:
        random.seed(seed)
        random.shuffle(list_idx)

    val_len = int(n_paths * (val_perc / 100))
    n_training = n_paths - val_len
    if test_perc is None:
        val_ids = np.array(list_idx)[n_training:]
        train_ids = np.array(list_idx)[:n_training]
        output = [train_ids, val_ids]
    else:
        test_len = int(n_paths * (test_perc / 100))
        n_training = n_training - test_len
        val_ids = np.array(list_idx)[n_training:(n_training + val_len)]
        test_ids = np.array(list_idx)[(n_training + val_len):]
        train_ids = np.array(list_idx)[:n_training]
        output = [train_ids, val_ids, test_ids]

    return output
