import numpy as np


def gradient_descent(x0, alpha, grad, n_iter=100, return_path=False):
    """Gradient descent.

    Parameters
    ----------
    x0 : array-like, shape (n_params,)
        Initial guess for parameter vector that will be optimized

    alpha : float
        Learning rate, should be within (0, 1), typical values are 1e-1, 1e-2,
        1e-3, ...

    grad : callable, array -> array
        Computes the derivative of the objective function with respect to the
        parameter vector

    n_iter : int, optional (default: 100)
        Number of iterations

    return_path : bool, optional (default: False)
        Return the path in parameter space that we took during the optimization

    Returns
    -------
    x : array, shape (n_params,)
        Optimized parameter vector

    path : list of arrays (shape (n_params,)), optional
        Path that we took in parameter space
    """
    if return_path:
        raise NotImplementedError("Return solution and path")
    else:
        raise NotImplementedError("Return solution")
