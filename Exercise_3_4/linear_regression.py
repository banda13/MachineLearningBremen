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
    return w[1]*X[:,1]+w[0]


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
        sum = sum + np.power(y[i]-(w[0]+w[1]*X[i][1]),2)
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
    gradient = [0]*2
    for i in range(len(y)):
        gradient[0] += -2 * (y[i] - (w[1] * X[i][1] + w[0]))
        gradient[1] += -2 * (y[i]-(w[1] *X[i][1] + w[0]))*(X[i][1])
    return gradient


if __name__ == "__main__":
    X, y = load_dataset()

    # 'partial' creates a new function-like object that only has one argument.
    # The other arguments will contain our dataset.
    grad = partial(dSSEdw, X=X, y=y)
    w0 = [-0.5, 0.0]
    w = gradient_descent(w0,0.0001,grad,100,False)
    print(w)
    x = np.linspace(-15, 15, 100)
    line = w[1]*x+w[0]
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.plot(x, line, '-r', label='linear regression')
    plt.scatter(X[:,1],y)
    plt.figure()
    plt.show()
