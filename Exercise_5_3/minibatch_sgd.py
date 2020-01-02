"""Mini-Batch Stochastic Gradient Descent (MBSGD)."""
from __future__ import print_function
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


def optimize_mbsgd(X, T, net, epochs=10, batch_size=10, alpha=0.01,
                   alpha_decay=1.0, min_alpha=0.01, eta=0.0, eta_inc=0.0,
                   max_eta=1.0, min_gain=0.01, max_gain=100.0,
                   random_state=None, verbose=0):
    """Fit the dataset.

    This implementation of gradient descent has some modifications:

    * it is stochastic, we update the weights with a randomly chosen
      subset of the training set to escape local minima
    * we use a momentum to smooth the search direction
    * each weight has an adaptive learning rate
    * we can decrease the learning rate during optimization
    * we can increase the momentum during optimization

    Standard mini-batch stochastic gradient descent updates the weight
    vector w in step t through

    .. math::

        w^t = w^{t-1} - \\frac{\\alpha}{|B_t|} \\sum_{n \\in B_t}
        \\nabla E_n(w),

    or

    .. math::

        \\Delta w^t = - \\frac{\\alpha}{|B_t|} \\sum_{n \\in B_t}
        \\nabla E_n(w), \\quad
        w^t = w^{t-1} + \\Delta w^t,

    where :math:`\\alpha` is the learning rate and :math:`B_t` is the set
    of indices of the t-th mini-batch, which is drawn randomly. The random
    order of gradients prevents us from getting stuck in local minima. This
    is an advantage over batch gradient descent. However, we must not make
    the batch size too small. A bigger batch size makes the optimization
    more robust against noise of the training set. A reasonable batch size
    is between 10 and 100. The learning rate has to be within (0, 1). High
    learning rates can result in divergence, i.e. the error increases. Too
    low learning rates might make learning too slow, i.e. the number of
    epoch required to find an optimum might be infeasibe. A reasonable
    value for :math:`\\alpha` is usually within [1e-5, 0.1].

    A momentum can increase the optimization stability. In this case, the
    update rule is

    .. math::

        \\Delta w^t = \\eta \\Delta w^{t-1} - \\frac{\\alpha}{|B_t|}
            \\sum_{n \\in B_t} \\nabla E_n(w), \\quad
        w^t = w^{t-1} + \\Delta w^t,

    where :math:`\\eta` is called momentum and must lie in (0, 1). The
    momentum term incorporates past gradients with exponentially decaying
    influence. This reduces changes of the search direction. An intuitive
    explanation of this update rule is: we regard w as the position of
    a ball that is rolling down a hill. The gradient represents its
    acceleration and the acceleration modifies its momentum.

    Another trick is using different learning rates for each weight. For
    each weight :math:`w_{ji}` we can introduce a gain :math:`g_{ji}`
    which will be multiplied with the learning rate so that we obtain
    an update rule for each weight

    .. math::

        \\Delta w_{ji}^t = \\eta \\Delta w_{ji}^{t-1}
            - \\frac{\\alpha g_{ji}^{t-1}}{|B_t|}
            \\sum_{n \\in B_t} \\nabla E_n(w_{ji}), \\quad
        w_{ji}^t = w_{ji}^{t-1} + \\Delta w_{ji}^t,

    where :math:`g_{ji}^0` = 1 and :math:`g_{ji}` will be increased by
    0.05 if :math:`\\Delta w_{ji}^t \\Delta w_{ji}^{t-1} \\geq 0`, i.e.
    the sign of the search direction did not change and :math:`g_{ji}` will
    be multiplied by 0.95 otherwise. We set a minimum and a maximum value
    for each gain. Usually these are 0.1 and 10 or 0.001 and 100
    respectively.

    During optimization it often makes sense to start with a more global
    search, i.e. with a high learning rate and decrease the learning rate
    as we approach the minimum so that we obtain an update rule for the
    learning rate:

    .. math::

        \\alpha^t = max(\\alpha_{decay} \\alpha^{t-1}, \\alpha_{min}).

    In addition, we can allow the optimizer to change the search direction
    more often at the beginning of the optimization and reduce this
    possibility at the end. To do this, we can start with a low momentum
    and increase it over time until we reach a maximum:

    .. math::

        \\eta^t = min(\\eta^{t-1} + \\eta_{inc}, \\eta_{max}).

    Parameters
    ----------
    X : array-like, shape (N, D)
        inputs

    T : array-like, shape (N, F)
        desired outputs

    net : MultilayerNeuralNetwork
        Model that will be trained

    epochs : int
        number of training epochs

    batch_size : int
        size of a mini-batch

    alpha : float
        initial learning rate

    alpha_decay : float
        the learning rate will be multiplied with this factor after each
        mini-batch

    min_alpha : float
        minimum learning rate

    eta : float
        initial momentum

    eta_inc : float
        the momentum will be increased by this amount after each
        mini-batch

    max_eta : float
        maximum momentum

    min_gain : float
        minimum learning rate gain

    max_gain : float
        maximum learning rate gain

    random_state : RandomState or int
        random number generator or seed

    verbose : int
        verbosity level

    Returns
    -------
    errors_ : list
        Accumulated errors during each episode on the training data
    """
    # Random number generator
    random_state = check_random_state(random_state)

    N = len(X)
    if N != len(T):
        raise ValueError("Number of input (%d) and target samples (%d) "
                         "should be equal" % (N, len(T)))

    # Initialize weights randomly
    net.initialize_weights(random_state)
    weights = net.get_weights()
    K = len(weights)

    # Temporary variables for optimization
    g = np.zeros(K)                    # Gradient
    momentum = np.zeros(K)             # Momentum ("smooth" gradient)
    gain = np.ones(K)                  # Individual gain for each weight
    indices = np.arange(N)

    # Measure optimization progress
    errors_ = np.empty(epochs)

    for epoch in range(epochs):
        # For each epoch we will have a different order
        random_state.shuffle(indices)
        # Temporary variables for logging
        accumulated_error = 0.0
        accumulated_gnorm = 0.0

        for n_start in range(0, N, batch_size):
            batch_indices = indices[n_start:n_start + batch_size]
            g, e = net.gradient(X[batch_indices], T[batch_indices],
                                get_error=True)
            g /= batch_size

            # Log temporary state
            gnorm = np.linalg.norm(g)
            avg_gain = np.abs(gain).sum() / len(gain)
            if verbose >= 2:
                print("epoch=%d, pattern=%d, alpha=%f, eta=%f, ||g||=%f, "
                      "e=%f, gain=%f       \r"
                      % (epoch+1, min(N, n_start+batch_size), alpha, eta,
                         gnorm, e, avg_gain), end="")

            # Update
            next_momentum = eta * momentum - alpha * g
            momentum_sign_change = next_momentum * momentum
            momentum = next_momentum
            gain[momentum_sign_change >= 0.0] += 0.05
            gain[momentum_sign_change < 0.0] *= 0.95
            gain = np.clip(gain, min_gain, max_gain)
            weights += momentum * gain

            net.set_weights(weights)

            # Update state of the optimizer
            alpha *= alpha_decay
            alpha = np.max((alpha, min_alpha))
            eta += eta_inc
            eta = np.min((eta, max_eta))

            accumulated_error += e
            accumulated_gnorm += gnorm

        if verbose >= 1:
            print("")
            print("E = %.2f, ||g|| = %.2f" % (accumulated_error,
                                              accumulated_gnorm / N))

        errors_[epoch] = accumulated_error

    return errors_


class MiniBatchSGD(BaseEstimator):
    """Mini-Batch Stochastic Gradient Descent.

    See :func:`optimize_mbsgd` for a detailed documentation of the parameters.
    """
    def __init__(self, net, epochs=10, batch_size=10, alpha=0.01,
                 alpha_decay=1.0, min_alpha=0.01, eta=0.0, eta_inc=0.0,
                 max_eta=1.0, min_gain=0.01, max_gain=100.0, random_state=None,
                 verbose=0):
        self.net = net
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_alpha = min_alpha
        self.eta = eta
        self.eta_inc = eta_inc
        self.max_eta = max_eta
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, T):
        # Optimize the neural net with mini-batch SGD
        self.error_ = optimize_mbsgd(
            X, T, net=self.net, epochs=self.epochs, batch_size=self.batch_size,
            alpha=self.alpha, alpha_decay=self.alpha_decay,
            min_alpha=self.min_alpha, eta=self.eta, eta_inc=self.eta_inc,
            max_eta=self.max_eta, min_gain=self.min_gain,
            max_gain=self.max_gain, random_state=self.random_state,
            verbose=self.verbose)

    def predict(self, X):
        return self.net.predict(X)

    def score(self, X, T):
        # High scores indicate good models in sklearn
        return -self.net.error(X, T)
