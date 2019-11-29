import numpy as np
import matplotlib.pyplot as plt

from Exercise_3_3.gradient_descent import gradient_descent


def plot_square():
    def fun_square(x):
        return np.linalg.norm(x) ** 2
    def grad_square(x):
        return 2.0 * x

    x, path = gradient_descent(
        x0=np.ones(1),
        alpha=0.7,
        grad=grad_square,
        n_iter=5,
        return_path=True
    )
    f = fun_square(x)
    plt.plot(path, [fun_square(x) for x in path], "-o",
             label="Gradient Descent Path")
    plt.plot(np.linspace(-1, 1, 1000),
             [fun_square(x) for x in np.linspace(-1, 1, 1000)],
             label="$f(x) = x^2$")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    plot_square()
