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
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Download Sarcos dataset if this is required
download_sarcos()
X, Y = load_sarcos("train")
X_test, Y_test = load_sarcos("test")
# Scale targets
target_scaler = StandardScaler()
Y = target_scaler.fit_transform(Y)
Y_test = target_scaler.transform(Y_test)
D = (X.shape[1],)
F = Y.shape[1]

def MBSGD(hyperparameters):
    np.random.seed(0)

    #Net structure
    layers = []
    layers.append({"type": "fully_connected", "num_nodes": 39})
    layers.append({"type": "fully_connected", "num_nodes": 61})

    D = (X.shape[1],)
    F = Y.shape[1]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=32,
                         alpha=hyperparameters['alpha'],
                         alpha_decay=hyperparameters['alpha_decay'],
                         min_alpha=hyperparameters['min_alpha'],
                         eta=hyperparameters['eta'],
                         eta_inc=hyperparameters['eta_inc'],
                         max_eta=hyperparameters['max_eta'],
                         random_state=0)

    scores = cross_val_score(estimator=mbsgd, X=X, y=Y, cv=15, n_jobs=-1)
    return -scores.mean()

    ############################################################################


def find_hyperparam():
    """
    This function creates the space of hyperparameters for the Bayesian optimization and run it.

    :return: best: (dictionary) best hyperparameters found by Bayesian optimization
    """
    space = {'alpha': hp.uniform('alpha', 0.09, 0.095),
             'alpha_decay': hp.uniform('alpha_decay', 0.97, 0.99),
             'min_alpha': hp.uniform('min_alpha', 0.00001, 0.00002),
             'eta': hp.uniform('eta', 0.0001, 0.0006),
             'eta_inc': hp.uniform('eta_inc', 0.0001, 0.0003),
             'max_eta': hp.uniform('max_eta', 0.975, 0.991)}

    trials = Trials()
    best = fmin(fn=MBSGD, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    return best

def create_optimized_net(hyperparameters):
    global X, Y, X_test, Y_test, D, F

    batch_size = 32
    alpha = hyperparameters['alpha']
    alpha_decay = hyperparameters['alpha_decay']
    min_alpha = hyperparameters['min_alpha']
    eta = hyperparameters['eta']
    eta_inc = hyperparameters['eta_inc']
    max_eta = hyperparameters['max_eta']
    layers = \
        [
            {
                "type": "fully_connected",
                "num_nodes": 156
            },
            {
                "type": "fully_connected",
                "num_nodes": 244
            }
        ]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=batch_size, alpha=alpha, alpha_decay=alpha_decay,
                         min_alpha=min_alpha, eta=eta, eta_inc=eta_inc, max_eta=max_eta, random_state=0, verbose=2)
    mbsgd.fit(X, Y)

    # Store learned model
    pickle.dump(model, open("sarcos_model.pickle", "wb"))

    return model

if __name__ == '__main__':

    # You find optimal hyperparameters with the function "find_hyperparam()" and create the optimized net
    # with "create_optimezed_net()" or you can load the model that we provided as result of the assignment
    # with:
    # model = pickle.load(open("sarcos_model.pickle", "rb"))

    # Comment these two lines of code if you load a model
    hyperparam = find_hyperparam()
    model = create_optimized_net(hyperparam)
    Y_pred = model.predict(X_test)
    for f in range(F):
        print("Dimension %d: nMSE = %.2f %%"
              % (f + 1, 100 * nMSE(Y_pred[:, f], Y_test[:, f])))
