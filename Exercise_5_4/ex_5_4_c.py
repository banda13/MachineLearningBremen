"""Train multilayer neural network with MBSGD on Sarcos data set."""
import numpy as np
import pickle
from Exercise_5_4.sarcos import download_sarcos
from Exercise_5_4.sarcos import load_sarcos
from Exercise_5_4.sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from Exercise_5_3.multilayer_neural_network import MultilayerNeuralNetwork
from Exercise_5_3.minibatch_sgd import MiniBatchSGD

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA


if __name__ == "__main__":
    np.random.seed(0)

    # Download Sarcos dataset if this is required
    download_sarcos()

    batch_size = 50
    alpha = 0.01
    alpha_decay = 0.95
    min_alpha = 0.00005
    eta = 0.0001  # A che cazzo serve?
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
    pca = PCA(n_components=16)
    target_scaler = StandardScaler()
    pca.fit(X)
    X = pca.fit_transform(X)
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    D = (X.shape[1],)
    F = Y.shape[1]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=10, batch_size=batch_size, alpha=alpha, alpha_decay=alpha_decay,
                         min_alpha=min_alpha, eta=eta, eta_inc=eta_inc, max_eta=max_eta, random_state=0)
    #mbsgd.fit(X, Y)

    ############################################################################

    all_accuracies = cross_val_score(estimator=mbsgd, X=X, y=Y, cv=5)

    print("Mean:")
    print(all_accuracies.mean())

    print("Std deviation:")
    print(all_accuracies.std())
    print()

