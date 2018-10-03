import numpy as np


def log_factorial(x):
    """
    Compute the log factorial by computing the sum of logs, rather than log of product.
    """

    assert(isinstance(x, (int, np.int8, np.int32, np.int64, np.int, np.integer))
           & (x >= 0))
    if x == 0:
        return 0
    else:
        return np.sum(np.log(np.arange(1, x+1)))
