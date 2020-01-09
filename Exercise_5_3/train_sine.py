"""Train multilayer neural network with MBSGD on sine dataset."""
import numpy as np
import matplotlib.pyplot as plt
from Exercise_5_3.multilayer_neural_network import MultilayerNeuralNetwork
from Exercise_5_3.minibatch_sgd import MiniBatchSGD


if __name__ == "__main__":
    np.random.seed(0)

    # Generate dataset
    X = np.random.rand(200, 1)
    Y = np.sin(2 * np.pi * X) + np.random.randn(*X.shape) * 0.1

    layers = \
        [
            {
                "type": "fully_connected",
                "num_nodes": 50
            },
            {
                "type": "fully_connected",
                "num_nodes": 20
            }
        ]
    epochs = 150

    mlnn = MultilayerNeuralNetwork(
        D=(1,), F=1, layers=layers, training="regression", std_dev=0.01,
        verbose=1)
    mbsgd = MiniBatchSGD(net=mlnn, epochs=epochs, batch_size=16, alpha=0.1,
                         eta=0.5, random_state=0, verbose=0)
    mbsgd.fit(X, Y)

    X_test = np.linspace(0, 1, 100)[:, np.newaxis]
    Y_test = np.sin(2 * np.pi * X_test)

    Y_test_prediction = mlnn.predict(X_test)

    # The following code assumes that these variables are defined:
    #
    # X : array, shape (200, 1)
    #     Inputs (training set), sampled uniformly from [0, 1] (or [0, 1))
    # Y : array, shape (200, 1)
    #     Targets (training set), sampled from sin(2*pi*x) + N(0, 0.01)
    # X_test : array, shape (100, 1)
    #     Inputs (test set), equally spaced values from [0, 1]
    # Y_test : array, shape (100, 1)
    #     Outputs (test set), sin(2*pi*X)
    # Y_test_prediction : array, shape (100, 1)
    #     Predicted values

    plt.figure()
    plt.title("Prediction")
    plt.scatter(X.ravel(), Y.ravel(), label="Training set (noisy)")
    plt.plot(X_test.ravel(), Y_test.ravel(), lw=3, label="True function")
    plt.plot(X_test.ravel(), Y_test_prediction.ravel(), lw=3,
             label="Prediction")
    plt.legend(loc="best")

    # Uncomment this code to plot the errors during optimization. We assume
    # that 'mbsgd' is an object of type 'MBSGD' that has been used to train
    # the neural network for 'epochs' epochs.
    #plt.figure()
    #plt.title("Learning Curve")
    #plt.plot(np.arange(epochs) + 1, mbsgd.errors_, label="Training error")
    #plt.legend(loc="best")

    plt.show()
