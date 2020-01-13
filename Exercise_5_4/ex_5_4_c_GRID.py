"""Train multilayer neural network with MBSGD on Sarcos data set."""
import numpy as np
import pickle
from Exercise_5_4.sarcos import download_sarcos
from Exercise_5_4.sarcos import load_sarcos
from Exercise_5_4.sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from Exercise_5_3.multilayer_neural_network import MultilayerNeuralNetwork
from Exercise_5_3.minibatch_sgd import MiniBatchSGD
from sklearn.model_selection import GridSearchCV, cross_val_score


def find_good_network(X, Y, attempts, random_seed):

    D = (X.shape[1],)
    F = Y.shape[1]
    best_mean = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    best_model = None
    for i in range(attempts):
        rand_layer_num = np.random.randint(low=2, high=5)
        layers = []
        for i in range(rand_layer_num):
            layers.append({"type": "fully_connected", "num_nodes": np.random.randint(low=50, high=250)})
        model = MultilayerNeuralNetwork(D, F, layers, training="regression", std_dev=0.001, verbose=True)
        mbsgd = MiniBatchSGD(net=model, epochs=10, random_state=0)
        all_accuracies = cross_val_score(estimator=mbsgd, X=X, y=Y, cv=10)
        mean = -all_accuracies.mean()
        if mean < best_mean:
            best_mean = mean
            best_model = model
        print()
    return best_model


if __name__ == "__main__":
    np.random.seed(0)

    # Download Sarcos dataset if this is required
    download_sarcos()

    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    best_model = find_good_network(X, Y, attempts=10, random_seed=np.random.seed(183))

    print("found best network:")
    for layer in best_model.layers:
        print("input nodes: "+str(layer.I)+", output nodes: "+str(layer.J))
    print()
    ############################################################################
    tuned_params = {'batch_size': np.arange(64, 256, 64),
                    'alpha': np.arange(0.01, 0.1, 0.02),
                    'alpha_decay': np.arange(0.01, 1, 0.2),
                    'min_alpha': np.arange(0.00001, 0.001, 0.0003),
                    'eta': np.arange(0.0001, 0.001, 0.0007),
                    'eta_inc': np.arange(0.00001, 0.01, 0.0005),
                    'max_eta': np.arange(0.7, 0.9, 0.2)
                    }

    gs = GridSearchCV(MiniBatchSGD(net=best_model, random_state=0), tuned_params, cv=5, n_jobs=8)

    gs.fit(X_test, Y_test)
    print(gs.best_params_)