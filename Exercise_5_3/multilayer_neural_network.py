"""Multilayer Neural Network."""
import numpy as np
from tools import (softmax, linear, linear_derivative, relu, relu_derivative,
                   convolve, back_convolve)


class FullyConnectedLayer(object):
    """Represents a trainable fully connected layer.

    Parameters
    ----------
    I : int or tuple
        input shape (without bias)

    J : int; outputs

    g : function: array-like -> array-like
        activation function y = g(a)

    gd : function: array-like -> array-like
        derivative g'(a) = gd(y)

    std_dev : float
        standard deviation of the normal distribution that we use to draw
        the initial weights

    verbose : int, optional
        verbosity level
    """
    def __init__(self, I, J, g, gd, std_dev, verbose=0):
        self.I = np.prod(I) + 1  # Add bias component
        self.J = J
        self.g = g
        self.gd = gd
        self.std_dev = std_dev

        self.W = np.empty((self.J, self.I))

        if verbose:
            print("Fully connected layer (%d nodes, %d x %d weights)"
                  % (self.J, self.J, self.I))

    def initialize_weights(self, random_state):
        """Initialize weights randomly.

        Parameters
        ----------
        random_state : RandomState or int
            random number generator or seed
        """
        ######################################################################
        raise NotImplementedError(
            "TODO Implement FullyConnectedLayer.initialize_weights()")
        ######################################################################

    def get_output_shape(self):
        """Get shape of the output.

        Returns
        -------
        shape : tuple
            shape of the output
        """
        return (self.J,)

    def forward(self, X):
        """Forward propagate the output of the previous layer.

        Parameters
        ----------
        X : array-like, shape = [N, I or self.I-1]
            input

        Returns
        -------
        Y : array-like, shape = [N, J]
            output
        """
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        if D != self.I - 1:
            raise ValueError("shape = " + str(X.shape))

        ######################################################################
        raise NotImplementedError(
            "TODO Implement FullyConnectedLayer.forward()")
        ######################################################################
        return self.Y

    def backpropagation(self, dEdY):
        """Backpropagate errors of the next layer.

        Parameters
        ----------
        dEdY : array-like, shape = [N, J]
            errors from the next layer

        Returns
        -------
        dEdX : array-like, shape = [N, I or self.I - 1]
            errors from this layer

        Wd : array-like, shape = [J, self.I]
            derivatives of the weights
        """
        if dEdY.shape[1] != self.J:
            raise ValueError("%r != %r" % (dEdY.shape[1], self.J))

        ######################################################################
        raise NotImplementedError(
            "TODO Implement FullyConnectedLayer.backpropagation()")
        ######################################################################
        return dEdX, Wd

    def get_weights(self):
        """Get current weights.

        Returns
        -------
        W : array-like, shape = [J * I + 1 or self.I]
            weight matrix
        """
        return self.W.flat

    def set_weights(self, W):
        """Set new weights.

        Parameters
        ----------
        W : array-like, shape = [J * I + 1 or self.I]
            weight matrix
        """
        self.W = W.reshape((self.J, self.I))

    def num_weights(self):
        """Get number of weights.

        Returns
        -------
        K : int
            number of weights
        """
        return self.W.size

    def __getstate__(self):
        # This will be called by pickle.dump, so we remove everything that
        # requires too much memory
        d = dict(self.__dict__)
        if "X" in d:
            del d["X"]
        if "Y" in d:
            del d["Y"]
        return d


class MultilayerNeuralNetwork(object):
    """Multilayer Neural Network (MLNN).

    Parameters
    ----------
    D : int or tuple
        input shape

    F : int
        number of outputs

    layers : list of dicts
        layer definitions

    training : string
        must be either classification or regression and defines the
        activation function of the last layer as well as the error function

    std_dev : float
        standard deviation of the normal distribution that we use to draw
        the initial weights

    verbose : int, optional
        verbosity level
    """

    def __init__(self, D, F, layers, training="classification", std_dev=0.05,
                 verbose=0):
        self.D = D
        self.F = F

        # Initialize layers
        self.layers = []
        I = self.D
        for layer in layers:
            l = None
            if layer["type"] == "fully_connected":
                l = FullyConnectedLayer(
                    I, layer["num_nodes"], relu, relu_derivative, std_dev,
                    verbose)
                I = l.get_output_shape()
            else:
                raise NotImplementedException("Layer type '%s' is not "
                                              "implemented." % layer["type"])
            self.layers.append(l)
        if training == "classification":
            self.layers.append(FullyConnectedLayer(
                I, self.F, softmax, linear_derivative, std_dev, verbose))
            self.error_function = "ce"
        elif training == "regression":
            self.layers.append(FullyConnectedLayer(
                I, self.F, linear, linear_derivative, std_dev, verbose))
            self.error_function = "sse"
        else:
            raise ValueError("Unknown 'training': %s" % training)

    def initialize_weights(self, random_state):
        """Initialize weights randomly.

        Parameters
        ----------
        random_state : RandomState or int
            random number generator or seed
        """
        for layer in self.layers:
            layer.initialize_weights(random_state)

    def error(self, X, T):
        """Calculate the Cross Entropy (CE).

        .. math::

            E = -\sum_n \sum_f ln(y^n_f) t^n_f,

        where n is the index of the instance, f is the index of the output
        component, y is the prediction and t is the target.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            each row represents an instance

        T : array-like, shape = [N, F]
            each row represents a target

        Returns
        -------
        E : float
            error: SSE for regression, cross entropy for classification
        """
        if len(X) != len(T):
            raise ValueError("Number of samples and targets must match")

        # Compute error of the dataset
        if self.error_function == "ce":
        ########################################################################
        # implementation is not required to solve the exercise
            raise NotImplementedError(
                "TODO implement MultilayerNeuralNetwork.error()")
        ########################################################################
        elif self.error_function == "sse":
        ######################################################################
            raise NotImplementedError(
                "TODO Implement MultilayerNeuralNetwork.error()")
        ######################################################################

    def numerical_gradient(self, X, T, eps=1e-5):
        """Compute the derivatives of the weights with finite differences.

        This function can be used to check the analytical gradient
        numerically. The partial derivative of E with respect to w is
        approximated through

        .. math::

            \partial E / \partial w = (E(w+\epsilon) - E(w-\epsilon)) /
                                      (2 \epsilon) + O(\epsilon^2),

        where :math:`\epsilon` is a small number.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            input

        T : array-like, shape = [N, F]
            desired output (target)

        eps : float, optional
            small number, you can make eps smaller to increase the accuracy
            of the differentiation until roundoff errors occur

        Returns
        -------
        wd : array-like, shape = [K,]
            weight vector derivative
        """
        w = self.get_weights()
        w_original = w.copy()
        wd = np.empty_like(w)
        for k in range(len(w)):
            w[k] = w_original[k] + eps
            self.set_weights(w)
            Ep = self.error(X, T)
            w[k] = w_original[k] - eps
            self.set_weights(w)
            Em = self.error(X, T)
            w[k] = w_original[k]
            wd[k] = (Ep - Em) / (2.0 * eps)
        self.set_weights(w_original)
        return wd

    def gradient(self, X, T, get_error=False):
        """Calculate the derivatives of the weights.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            input

        T : array-like, shape = [N, F]
            desired output (target)

        Returns
        -------
        g : array-like, shape = [K,]
            gradient of weight vector

        e : float, optional
            error
        """
        Wds = []
        # Forward propagation
        Y = self.predict(X)
        # Backpropagation
        dEdY = Y - T
        for l in reversed(range(len(self.layers))):
            dEdY, Wd = self.layers[l].backpropagation(dEdY)
            Wds.insert(0, Wd)
        g = np.concatenate([Wds[l].flat for l in range(len(self.layers))])
        if get_error:
            if self.error_function == "ce":
                ##############################################################
                raise NotImplementedError(
                    "TODO implement MultilayerNeuralNetwork.gradient()")
                ##############################################################
            elif self.error_function == "sse":
                ##############################################################
                raise NotImplementedError(
                    "TODO Implement MultilayerNeuralNetwork.gradient()")
                ##############################################################
            return g,
        else:
            return g

    def get_weights(self):
        """Get current weight vector.

        Returns
        -------
        w : array-like, shape (K,)
            weight vector
        """
        return np.concatenate([self.layers[l].get_weights()
                               for l in range(len(self.layers))])

    def set_weights(self, w):
        """Set new weight vector.

        Parameters
        ----------
        w : array-like, shape=[K,]
            weight vector
        """
        i = 0
        for l in range(len(self.layers)):
            k = self.layers[l].num_weights()
            self.layers[l].set_weights(w[i:i + k])
            i += k

    def predict(self, X):
        """Predict values.

        Parameters
        ----------
        X : array-like, shape = [N, D]
            each row represents an instance

        Returns
        -------
        Y: array-like, shape = [N, F]
            each row represents a prediction
        """
        # Forward propagation
        ######################################################################
        raise NotImplementedError(
            "TODO Implement MultilayerNeuralNetwork.predict()")
        ######################################################################
        return X
