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
    return np.sum((y - predict(w, X))**2)


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
    gradient[0] = -2 * np.sum(y - predict(w, X))
    gradient[1] = -2 * np.sum((y - predict(w, X)) * X[:,1])
    return gradient


if __name__ == "__main__":
    X, y = load_dataset()

    # 'partial' creates a new function-like object that only has one argument.
    # The other arguments will contain our dataset.
    grad = partial(dSSEdw, X=X, y=y)
    w0 = [-0.5, 0.0]
    alpha = [0.0001, 0.001, 0.002, 0.0025]

    for i in range(len(alpha)):
        sse_list = []
        w = None
        for j in range(101):
            w = gradient_descent(w0,alpha[i],grad,j,False)
            sse_list.append(sse(w, X, y))

        # Plot of the linear regression

        x = np.linspace(-15, 15, 100)
        line = w[1]*x+w[0]
        plt.figure()
        plt.xlim((-10,10))
        plt.ylim((-10,10))
        plt.plot(x, line, '-r', label='Linear regression')
        plt.scatter(X[:,1],y)
        plt.title("Linear regression with α = %s" %alpha[i])

        # Plot of SSE
        plt.figure()
        plt.xlabel("Iterations")
        plt.ylabel("SSE")
        plt.plot(range(0,101), sse_list, '-b', label='SSE value vs. number of iterations')
        plt.title("SSE trend with α = %s" %alpha[i])
    plt.show()

    print("W* is %s" %w)

