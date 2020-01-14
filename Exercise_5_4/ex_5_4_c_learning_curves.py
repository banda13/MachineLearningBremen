import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from Exercise_5_3.minibatch_sgd import MiniBatchSGD
from Exercise_5_3.multilayer_neural_network import MultilayerNeuralNetwork
from Exercise_5_4.sarcos import download_sarcos, load_sarcos

# TODO find the best ones
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


def plot_learning_curve(estimator, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.show()

    # Plot n_samples vs fit_times
    plt.grid()
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    plt.xlabel("Training examples")
    plt.ylabel("fit_times")
    plt.title("Scalability of the model")
    plt.show()

    # Plot fit_time vs score
    plt.grid()
    plt.plot(fit_times_mean, test_scores_mean, 'o-')
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    plt.xlabel("fit_times")
    plt.ylabel("Score")
    plt.title("Performance of the model")
    plt.show()

    return plt


if __name__ == '__main__':
    download_sarcos()
    X, Y = load_sarcos("train")
    X_test, Y_test = load_sarcos("test")
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(Y)
    Y_test = target_scaler.transform(Y_test)
    D = (X.shape[1],)
    F = Y.shape[1]

    model = MultilayerNeuralNetwork(D, F, layers, training="regression",
                                    std_dev=0.001, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=batch_size, alpha=alpha, alpha_decay=alpha_decay,
                         min_alpha=min_alpha, eta=eta, eta_inc=eta_inc, max_eta=max_eta, random_state=0, verbose=2)
    plot_learning_curve(mbsgd, X, Y, cv=None, n_jobs=4)
