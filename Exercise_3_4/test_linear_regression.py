from functools import partial

import numpy as np
from scipy.optimize import approx_fprime
from numpy.testing import assert_array_almost_equal

from linear_regression import predict, sse, dSSEdw


def test_predict():
    X = np.array(
        [[1.0, 2.0],
         [1.0, 0.0],
         [1.0, -2.0]]
    )
    w = np.array([1.5, 0.3])
    y = predict(w, X)
    assert_array_almost_equal(y, np.array([2.1, 1.5, 0.9]))


def test_numerical_gradient_approximation():
    n_features = 1
    n_samples = 100

    # generate random dataset
    random_state = np.random.RandomState(42)
    X = np.hstack((np.ones(n_samples)[:, np.newaxis],
                   random_state.randn(n_samples, n_features)))
    y = random_state.randn(n_samples)

    # specialize objective and gradient with the dataset
    fun = partial(sse, X=X, y=y)
    grad = partial(dSSEdw, X=X, y=y)

    # compare numerical gradient and exact gradient for one weight vector
    w = random_state.randn(n_features + 1)
    g_approx = approx_fprime(w, fun, epsilon=1e-6)
    g_exact = grad(w)
    assert_array_almost_equal(g_approx, g_exact, decimal=4)


if __name__ == "__main__":
    print(test_predict())
    print(test_numerical_gradient_approximation())
