"""Train multilayer neural network with MBSGD on Sarcos data set."""
import joblib
import numpy as np
import pickle

import optuna

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

def MBSGD(trial):
    np.random.seed(0)

    # Download Sarcos dataset if this is required

    layers = []
    for i in range(int(trial.suggest_int('num_layers', 2, 4))):
        layers.append({"type": "fully_connected", "num_nodes": int(trial.suggest_discrete_uniform('num_nodes',40, 200, 10))})



    D = (X.shape[1],)
    F = Y.shape[1]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100,
                         batch_size=int(trial.suggest_discrete_uniform('batch_size', 32, 256, 32)),
                         alpha= trial.suggest_uniform('alpha', 0.09, 0.1),
                         alpha_decay=trial.suggest_uniform('alpha_decay', 0.9, 0.995),
                         min_alpha=trial.suggest_uniform('min_alpha', 0.00001, 0.0001),
                         eta=trial.suggest_uniform('eta', 0.0001, 0.0002),
                         eta_inc=trial.suggest_uniform('eta_inc', 1e-5, 1e-4),
                         max_eta=trial.suggest_uniform('max_eta', 0.9, 0.95),
                         random_state=0)

    ############################################################################

    all_accuracies = cross_val_score(estimator=mbsgd, X=X, y=Y, cv=10, n_jobs=-1)
    return all_accuracies.mean()


if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(sampler=sampler)
    study.optimize(MBSGD, n_trials=5)
    joblib.dump(study, 'optuna.pkl')

