"""Train multilayer neural network with MBSGD on Sarcos data set."""
import numpy as np
import pickle
from sarcos import download_sarcos
from sarcos import load_sarcos
from sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from multilayer_neural_network import MultilayerNeuralNetwork
from minibatch_sgd import MiniBatchSGD


if __name__ == "__main__":
    np.random.seed(0)

    # Download Sarcos dataset if this is required
    download_sarcos()

    # Load training set and test set
    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    # Scale targets
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    # Train model (code for exercise 10.2 1/2/3)
    ############################################################################
    # Train neural network
    D = (X.shape[1],)
    F = Y.shape[1]
    layers = \
        [
            {
                "type": "fully_connected",
                "num_nodes": 10
            }
        ]
    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.01, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=32, alpha=0.005,
                         eta=0.5, random_state=0, verbose=2)
    mbsgd.fit(X, Y)
    ############################################################################

    # Print nMSE on test set
    Y_pred = model.predict(X_test)
    for f in range(F):
        print("Dimension %d: nMSE = %.2f %%"
              % (f + 1, 100 * nMSE(Y_pred[:, f], Y_test[:, f])))

    # Store learned model, you can restore it with
    # model = pickle.load(open("sarcos_model.pickle", "rb"))
    # and use it in your evaluation script
    pickle.dump(model, open("sarcos_model.pickle", "wb"))
