import pickle
import numpy as np


def save_obj(obj, file, protocol=None):
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    with open(file, 'wb') as f:
        pickle.dump(obj, f, protocol)


def load_obj(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def check_binary_array(*arrays):
    is_binary = [np.array_equal(array, array.astype(bool)) for array in arrays]
    if not all(is_binary):
        raise ValueError('Found input arrays which are not binary')

