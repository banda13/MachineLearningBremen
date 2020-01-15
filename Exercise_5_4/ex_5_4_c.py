"""Train multilayer neural network with MBSGD on Sarcos data set."""
import numpy as np
from Exercise_5_4.sarcos import download_sarcos
from Exercise_5_4.sarcos import load_sarcos
from sklearn.preprocessing import StandardScaler
from Exercise_5_3.multilayer_neural_network import MultilayerNeuralNetwork
from Exercise_5_3.minibatch_sgd import MiniBatchSGD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
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
    for i in range(2):
        layers.append({"type": "fully_connected", "num_nodes": 50})



    D = (X.shape[1],)
    F = Y.shape[1]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=hyperparameters['batch_size'], alpha=hyperparameters['alpha'],
                         alpha_decay=hyperparameters['alpha_decay'], min_alpha=hyperparameters['min_alpha'],
                         eta=hyperparameters['eta'], eta_inc=hyperparameters['eta_inc'],
                         max_eta=hyperparameters['max_eta'], random_state=0)
    ############################################################################

    score = cross_val_score(estimator=mbsgd, X=X, y=Y, cv=10, n_jobs=-1)

    return score.mean()


if __name__ == '__main__':

    space = {'alpha': hp.uniform('alpha', 0.09, 0.1),
             'eta': hp.uniform('eta', 0.0001, 0.0002),
             'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
             'alpha_decay': hp.uniform('alpha_decay', 0.9, 0.995),
             'min_alpha': hp.uniform('min_alpha', 0.00001, 0.0001),
             'eta_inc': hp.uniform('eta_inc', 1e-5, 1e-4),
             'max_eta': hp.uniform('max_eta', 0.9, 0.95)}

    trials = Trials()
    best = fmin(fn=MBSGD, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    print(best)
