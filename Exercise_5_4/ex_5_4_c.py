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
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

download_sarcos()
X, Y = load_sarcos("train")
X_test, Y_test = load_sarcos("test")
# Scale targets
target_scaler = StandardScaler()
Y = target_scaler.fit_transform(Y)
Y_test = target_scaler.transform(Y_test)

def MBSGD(hyperparameters):
    np.random.seed(0)

    # Download Sarcos dataset if this is required

    layers = []
    for i in range(int(hyperparameters['num_layers'])):
        layers.append({"type": "fully_connected", "num_nodes": hyperparameters['num_nodes']})



    D = (X.shape[1],)
    F = Y.shape[1]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=10, batch_size=hyperparameters['batch_size'], alpha=hyperparameters['alpha'],
                         alpha_decay=hyperparameters['alpha_decay'], min_alpha=hyperparameters['min_alpha'],
                         eta=hyperparameters['eta'], eta_inc=hyperparameters['eta_inc'],
                         max_eta=hyperparameters['max_eta'], random_state=0)
    #mbsgd.fit(X, Y)

    ############################################################################

    all_accuracies = cross_val_score(estimator=mbsgd, X=X, y=Y, cv=10)
    return all_accuracies.mean()

if __name__ == '__main__':

    space = {'batch_size': hp.uniformint('batch_size', 32, 256), 'alpha': hp.uniform('alpha', 0.05, 0.1),
             'alpha_decay': hp.uniform('alpha_decay', 0,1), 'min_alpha': hp.uniform('min_alpha', 0.00001, 0.0009),
             'eta': hp.uniform('eta', 0.5, 0.95), 'eta_inc': hp.uniform('eta_inc', 0.001, 0.1),
             'max_eta': hp.uniform('max_eta', 0.5, 0.95), 'num_layers': hp.uniformint('num_layers', 1, 5),
             'num_nodes': hp.uniformint('num_nodes', 20, 100)}

    trials = Trials()
    best = fmin(fn=MBSGD, space=space, algo=tpe.suggest, max_evals=500, trials=trials)

    print(best)

