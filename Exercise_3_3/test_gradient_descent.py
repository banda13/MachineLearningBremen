import numpy as np
from numpy.testing import assert_almost_equal

from Exercise_3_3.gradient_descent import gradient_descent


def test_square():
    def fun_square(x):
        return np.linalg.norm(x) ** 2
    def grad_square(x):
        return 2.0 * x

    random_state = np.random.RandomState(42)

    x = gradient_descent(
        x0=random_state.randn(5),
        alpha=0.1,
        grad=grad_square,
        n_iter=100,
        return_path=False
    )
    f = fun_square(x)
    assert_almost_equal(f, 0.0)


if __name__ == "__main__":
    test_square()
