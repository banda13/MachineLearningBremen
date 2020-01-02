"""Helper functions for artificial neural networks.
"""

import numpy as np


def softmax(A):
    """Softmax activation function.

    The outputs will be interpreted as probabilities and thus have to
    lie within [0, 1] and must sum to unity:

    .. math::

        g(a_f) = \\frac{\\exp(a_f)}{\\sum_{f'} \\exp(a_{f'})}.

    To avoid numerical problems, we substract the maximum component of
    :math:`a` from all of its components before we calculate the output. This
    is mathematically equivalent.

    Parameters
    ----------
    a : array-like, shape (N, F)
        activations

    Returns
    -------
    y : array-like, shape (N, F)
        outputs
    """
    ############################################################################
    # implementation is not required to solve the exercise
    raise NotImplementedError("TODO implement softmax()")
    ############################################################################


def linear(A):
    """Linear activation function.

    Returns the input:

    .. math::

        g(a_f) = a_f

    Parameters
    ----------
    A : array-like, shape = (N, F)
        activations

    Returns
    -------
    Y : array-like, shape = (N, F)
        outputs
    """
    return A


def linear_derivative(Y):
    """Derivative of linear activation function.

    Parameters
    ----------
    Y : array-like, shape (N, F)
        outputs (g(A))

    Returns
    -------
    gd(Y) : array-like, shape (N, F)
        derivatives (gdot(A))
    """
    return 1


def relu(A):
    """Non-saturating activation function: Rectified Linar Unit (ReLU).

    Max-with-zero nonlinearity: :math:`max(0, a)`.

    Parameters
    ----------
    A : array-like, shape (N, J)
        activations

    Returns
    -------
    Y : array-like, shape (N, J)
        outputs
    """
    return np.maximum(A, 0)


def relu_derivative(Y):
    """Derivative of ReLU activation function.

    Parameters
    ----------
    Y : array-like, shape (N, J)
        outputs (g(A))

    Returns
    -------
    gd(Y) : array-like, shape = (N, J)
        derivatives (gdot(A))
    """
    return np.sign(Y)
