import pickle
import gzip
import numpy as np
from typing import Tuple, List, Any

def load_data_wrapper(path: str = '../data/mnist.pkl.gz') -> Tuple[List[Any], List[Any], List[Any]]:
    try:
        f = gzip.open(path, 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        f.close()
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}. Please download mnist.pkl.gz")

    def vectorized_result(j: int) -> np.ndarray:
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    train_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    valid_data = list(zip(validation_inputs, validation_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    tst_data = list(zip(test_inputs, test_data[1]))

    return (train_data, valid_data, tst_data)