# Implementation of a Neural Net

The code in this directory has been tested on Ubuntu 14.04 with Python 2.7
and Windows 8.1 with Anaconda and Python 3.4+. If you have any problems
with your machine you should contact me (Alexander.Fabisch@dfki.de).

## Dependencies

In this implementation we assume that you have several packages installed
on your machine:

* NumPy
* SciPy
* Scikit-Learn

## Files

* minibatch_sgd.py - implements mini-batch stochastic gradient descent,
  provides the class `MiniBatchSGD` that is derived from sklearn's
  `BaseEstimator` so that it can be used with model selection tools from
  sklearn
* multilayer_neural_network.py - implements a generic multilayer neural
  network that you have to implement in the exercise
* sarcos.py - downloads and loads the Sarcos dataset
* tools.py - contains some functions that help us to implement the neural
  net (e.g. activation functions)
* test_gradient.py - test script to check the gradients of the neural net
* train_sine.py - a script that trains a neural net on a toy dataset
* train_sarcos.py - a script that trains a neural net on the Sarcos dataset

