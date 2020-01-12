"""Train multilayer neural network with MBSGD on Sarcos data set."""
import numpy as np
import pickle
from Exercise_5_4.sarcos import download_sarcos
from Exercise_5_4.sarcos import load_sarcos
from Exercise_5_4.sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from Exercise_5_3.multilayer_neural_network import MultilayerNeuralNetwork
from Exercise_5_3.minibatch_sgd import MiniBatchSGD

def a():
    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    # Scale targets
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    D = (X.shape[1],)
    F = Y.shape[1]
    layers = []
    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.1, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=32, alpha=0.005,
                         eta=0.5, random_state=0, verbose=2)
    mbsgd.fit(X, Y)

    # Print nMSE on test set
    Y_pred = model.predict(X_test)
    for f in range(F):
        print("Dimension %d: nMSE = %.2f %%"
              % (f + 1, 100 * nMSE(Y_pred[:, f], Y_test[:, f])))

    # Store learned model, you can restore it with
    # model = pickle.load(open("sarcos_model.pickle", "rb"))
    # and use it in your evaluation script
    pickle.dump(model, open("sarcos_model.pickle", "wb"))

def b():
    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    # Scale targets
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    D = (X.shape[1],)
    F = Y.shape[1]
    layers = \
        [
            {
                "type": "fully_connected",
                "num_nodes": 50
            }
        ]
    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.1, verbose=True)
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

def c_optimal_parameters_supposed():
    batch_size = 50
    alpha = 0.01
    alpha_decay = 0.95
    min_alpha = 0.00005
    eta = 0.0001
    eta_inc = 0.01
    max_eta = 0.95
    layers = \
        [
            {
                "type": "fully_connected",
                "num_nodes": 50
            },
            {
                "type": "fully_connected",
                "num_nodes": 50
            }
        ]

    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    # Scale targets
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    D = (X.shape[1],)
    F = Y.shape[1]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=batch_size, alpha=alpha, alpha_decay=alpha_decay,
                         min_alpha=min_alpha, eta=eta, eta_inc=eta_inc, max_eta=max_eta, random_state=0, verbose=2)
    mbsgd.fit(X, Y)

    # Print nMSE on test set
    Y_pred = model.predict(X_test)
    for f in range(F):
        print("Dimension %d: nMSE = %.2f %%"
              % (f + 1, 100 * nMSE(Y_pred[:, f], Y_test[:, f])))

    # Store learned model, you can restore it with
    # model = pickle.load(open("sarcos_model.pickle", "rb"))
    # and use it in your evaluation script
    pickle.dump(model, open("sarcos_model.pickle", "wb"))


if __name__ == "__main__":
    np.random.seed(0)

    # Download Sarcos dataset if this is required
    download_sarcos()

    #a()

    #b()

    c_optimal_parameters_supposed()


