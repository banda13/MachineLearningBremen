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

    all_accuracies = -cross_val_score(estimator=mbsgd, X=X, y=Y, cv=10)
    return all_accuracies.mean()

def c_optimal_parameters_supposed(parameters):
    np.random.seed(0)
    layers = []
    for i in range(int(parameters['num_layers'])):
        layers.append({"type": "fully_connected", "num_nodes": int(parameters['num_nodes'])})


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
    mbsgd = MiniBatchSGD(net=model, epochs=10, batch_size=int(parameters['batch_size']), alpha=float(parameters['alpha']),
                         alpha_decay=float(parameters['alpha_decay']), min_alpha=float(parameters['min_alpha']),
                         eta=float(parameters['eta']), eta_inc=float(parameters['eta_inc']),
                         max_eta=float(parameters['max_eta']), random_state=0)
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


if __name__ == '__main__':
    alpha = hp.uniform('alpha', 1e-5, 0.1)
    eta = hp.uniform('eta', 0.0001, 0.0002)
    space = {'alpha': alpha,
             'eta': eta,
             'batch_size': hp.uniformint('batch_size', 10, 100),
             'alpha_decay': hp.uniform('alpha_decay', 0.9, 1),
             'min_alpha': hp.uniform('min_alpha', 1e-5, alpha),
             'eta_inc': hp.uniform('eta_inc', 1e-5, 1e-4),
             'max_eta': hp.uniform('max_eta', 0.8, 0.99),
             'num_layers': hp.uniformint('num_layers', 1, 3),
             'num_nodes': hp.uniformint('num_nodes', 20, 100)}

    trials = Trials()
    best = fmin(fn=MBSGD, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    c_optimal_parameters_supposed(best)

