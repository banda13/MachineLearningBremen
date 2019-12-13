import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import check_random_state

class BayesianOptimizer:
    """ Bayesian Optimizer, this class is already complete! """

    def __init__(self, acquisition_function, initial_random_samples=3,
                 random_state=0):
        """ The constructor, initializes the optimizer

        Parameters
        ----------
        acquisition_function: callable
            The acquisition function for the optimization
        initial_random_samples:
            How many query points should be samples randomly, before the
            GPR model is trained and used.
        random_state:
            Seed for the random number generator
        """

        self.acquisition_function = acquisition_function
        self.initial_random_samples = initial_random_samples

        self.rng = check_random_state(random_state)

        self.X = []
        self.y = []

        kernel = RBF(length_scale=0.1, length_scale_bounds=(0.01, 1.0))
        self.gpr_model = GaussianProcessRegressor(kernel=kernel)
        self.x0 = np.asarray([0])

    def get_next_query_point(self, bounds):
        """ Suggest a new query point to evaluate the objective function at

        Parameters
        ----------
        bounds: tuple, shape(2)
            lower and upper bound for the query point to suggest
        """
        if len(self.X) < self.initial_random_samples:
            x_query = np.asarray([self.rng.uniform(low=bounds[0],
                                                   high=bounds[1])])
        else:
            objective_fun = lambda x: -self.acquisition_function(x=x, model=self.gpr_model)

            result = minimize(fun=objective_fun, x0=self.x0,
                              method='L-BFGS-B', bounds=[bounds])

            x_query = result.x

        return x_query

    def update_model(self, X, y):
        """ Update the Gaussian Process Regeression Model based on the returned
            value of the objective function.

        Parameters
        ----------
        X: ndarray, shape [n_samples, n_dims]
            The point(s) the objective function had been evaluated
        y: ndarraym shape [n_samples,]
            Corresponding values of the objetive function
        """
        self.X.append(X)
        self.y.append(y)

        if len(self.X) >= self.initial_random_samples:
            self.gpr_model.fit(np.array(self.X).reshape(-1, 1), self.y)


class UpperConfidenceBound:
    """ The Upper Confidence Bound Acquistion Function. The class is defined as
        so called Functor class, making each object created from it executable
        (like a regular function). """

    def __init__(self, k):
        """ The constructor, taking the hyperparameters for the
            acquisition function.

        Parameters
        ----------
            ....
        """
        self.k = k


    def __call__(self, x, model):
        """__call__: This method makes any object from the class callable like
           a regular function.

        Parameters
        ----------
        x: ndarray, shape [n_dims,]
            The point to evaluate the acquisition function at
        model: sklearn.gaussian_process.GaussianProcessRegressor
            The gaussian process regression model for the objective function
        """
        mu, sigma = model.predict(x.reshape(-1, 1), return_std=True)
        ucb = mu + self.k * sigma
        return ucb


class ExpectedImprovement:

    def __init__(self, X, xi=0.01):
        self.X = X
        self.xi = xi

    def __call__(self, x, model):
        mu, sigma = model.predict(self.X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu_sample = model.predict(x.reshape(-1, 1))

        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei


def f(x):
    """ The objective function to optimize. Note: Usally this function is not
        available in analytic form, when bayesian optimization is applied. Note
                 that it is noisy.

    Parameters
    ----------
    x:
        The point(s) to evaluate the objective function at.
    """
    return (1.0 - np.tanh(x**2)) * np.sin(x*6) + np.random.normal(loc=0.0,
                                                                  scale=0.02)


if __name__ == "__main__":

    # set the parameters and execute the bayesian optimization
    iter = 50
    bounds = np.array([-1, 1])
    optimizer = BayesianOptimizer(UpperConfidenceBound(10))
    for i in range(iter):
        X = optimizer.get_next_query_point(bounds)
        Y = f(X)
        optimizer.update_model(X[0], Y[0])

    # calculate the real and the predicted y-values for each x
    x = np.atleast_2d(np.linspace(bounds[0], bounds[1], iter)).T
    y_pred, sigma = optimizer.gpr_model.predict(x, return_std=True)
    y = f(x)

    # plot the results
    plt.plot(x, y_pred, 'b-', label='Predictions')
    plt.plot(optimizer.X, optimizer.y, 'r.', markersize=10, label='Observations')
    plt.plot(x, y, 'r:', label='f(x)')

    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')

    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()