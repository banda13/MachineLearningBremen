from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from Exercise_3_3.gradient_descent import gradient_descent


def load_dataset(filename="regression_dataset_1.txt"):
    """Load the given dataset.

    Parameters
    ----------
    filename : string, optional
        Name of the file that contains the data

    Returns
    -------
    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs
    """
    x, y = eval(open(filename, "r").read())
    n_samples = len(x)
    X = np.vstack((np.ones(n_samples), x)).T
    y = np.asarray(y)
    return X, y


def predict(w, X):
    """Predict outputs of a linear model.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    Returns
    -------
    y : array, shape (n_samples,)
        Outputs
    """
    raise NotImplementedError("predict")


def sse(w, X, y):
    """Compute the sum of squared error of a linear model.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs

    Returns
    -------
    SSE : float
        Sum of squared errors
    """
    sum = 0
    for i in range(len(y)):
        sum += np.power(y[i]-(w[i][0]+w[i][1]*X[i]),2)
    return sum


def dSSEdw(w, X, y):
    """Compute the gradient of the sum of squared error.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs

    Returns
    -------
    g : array, shape (n_features + 1,)
        Sum of squared errors
    """
    raise NotImplementedError("dSSEdw")


if __name__ == "__main__":
    X, y = load_dataset()

    # 'partial' creates a new function-like object that only has one argument.
    # The other arguments will contain our dataset.
    grad = partial(dSSEdw, X=X, y=y)

    plt.figure()
    ax = plt.subplot(111)

    # YOUR CODE GOES HERE

    plt.show()
