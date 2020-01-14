"""Train multilayer neural network with MBSGD on Sarcos data set."""
import sys
import numpy as np
import pickle
from Exercise_5_4.sarcos import download_sarcos
from Exercise_5_4.sarcos import load_sarcos
from Exercise_5_4.sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from Exercise_5_3.multilayer_neural_network import MultilayerNeuralNetwork
from Exercise_5_3.minibatch_sgd import MiniBatchSGD
from sklearn.model_selection import GridSearchCV, cross_val_score

def test_params(D, F, X, Y, best_mean, best_model, layers):
    model = MultilayerNeuralNetwork(D, F, layers, training="regression", std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=10, random_state=0)
    all_accuracies = cross_val_score(estimator=mbsgd, X=X, y=Y, cv=10)
    mean = -all_accuracies.mean()
    if mean < best_mean:
        best_mean = mean
        best_model = model
    print()
    return best_model, best_mean

def find_opt_num_layers(X, Y, max_num_layers):
    num_nodes = 100
    D = (X.shape[1],)
    F = Y.shape[1]
    best_mean = sys.maxsize
    best_model = None
    for i in range(max_num_layers):
        layers = []
        for j in range(i):
            layers.append({"type": "fully_connected", "num_nodes": num_nodes/i})
        best_model, best_mean = test_params(D, F, X, Y, best_mean, best_model, layers)
    return best_model.layers.shape[0]

def find_opt_node_distro(X, Y, num_layers):
    network_nodes = 100
    D = (X.shape[1],)
    F = Y.shape[1]
    best_mean = sys.maxsize
    best_model = None
    remaining_nodes = network_nodes
    for i in range(0, network_nodes, 20):
        layers = []
        while remaining_nodes > 0:
            nodes_this_layer = network_nodes/np.random.randint(low=network_nodes/num_layers, high=network_nodes)
            layers.append({"type": "fully_connected", "num_nodes": nodes_this_layer})
            remaining_nodes -= nodes_this_layer
        best_model, best_mean = test_params(D, F, X, Y, best_mean, best_model, layers)
    node_distro = []
    #could be optimized
    for layer in best_model.layers:
        node_distro.append(layer.I)
    return node_distro


def find_opt_num_nodes(X, Y, max_num_nodes, nodes_distro, random_seed=10):
    np.random.seed(random_seed)
    nodes_distro = np.array(nodes_distro)
    D = (X.shape[1],)
    F = Y.shape[1]
    best_mean = sys.maxsize
    best_model = None
    for i in range(0, max_num_nodes, 20):
        layers = []
        for j in range(nodes_distro.shape[0]):
            layers.append({"type": "fully_connected", "num_nodes": max_num_nodes*nodes_distro[j]/100})
        best_model, best_mean = test_params(D, F, X, Y, best_mean, best_model, layers)
    best_num_nodes = 0
    for layer in best_model.layers:
        best_num_nodes+=layer.I
    return best_num_nodes

def make_best_model(X, Y):
    num_layers = find_opt_num_layers(X, Y, 10)
    structure = find_opt_node_distro(X, Y, num_layers)
    num_nodes = find_opt_num_nodes(X, Y, structure, random_seed=345)
    D = (X.shape[1],)
    F = Y.shape[1]
    layers = []
    for i in num_layers:
        layers.append({"type": "fully_connected", "num_nodes": num_nodes*structure[i]/100})
    best_model = MultilayerNeuralNetwork(D, F, layers, training="regression", std_dev=0.001, verbose=True)
    return best_model

def find_good_network(X, Y, attempts, random_seed=10):
    np.random.seed(random_seed)
    D = (X.shape[1],)
    F = Y.shape[1]
    best_mean = sys.maxsize
    best_model = None
    for i in range(attempts):
        rand_layer_num = np.random.randint(low=1, high=3)
        layers = []
        for i in range(rand_layer_num):
            layers.append({"type": "fully_connected", "num_nodes": np.random.randint(low=10, high=30)})
        best_model, best_mean = test_params(D, F, X, Y, best_mean, best_model, layers)
    return best_model


if __name__ == "__main__":

    # Download Sarcos dataset if this is required
    download_sarcos()

    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)

    best_model = make_best_model(X, Y)

    print("found best network:")
    for layer in best_model.layers:
        print("input nodes: "+str(layer.I)+", output nodes: "+str(layer.J))
    print()
    ############################################################################
    tuned_params = {'batch_size': np.arange(32, 128, 32),
                    'alpha': np.arange(0.01, 0.1, 0.001),
                    'alpha_decay': np.arange(0.01, 1, 0.0001),
                    'min_alpha': np.arange(0.00001, 0.001, 0.00001),
                    'eta': np.arange(0.0001, 0.001, 0.00001),
                    'eta_inc': np.arange(0.00001, 0.01, 0.00001),
                    'max_eta': np.arange(0.1, 0.9, 0.01)
                    }

    gs = GridSearchCV(MiniBatchSGD(net=best_model, random_state=0), tuned_params, cv=5, n_jobs=8)

    gs.fit(X_test, Y_test)
    print(gs.best_params_)