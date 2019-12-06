import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class SVM(object):

    def __init__(self, C=1, kernel='rbf', gamma='scale'):
        """
        Initialize the support vector classifier
        :param C: regularization parameter
        :param kernel: kernel type used by the algorithm: linear, poly, rbf, sigmoid or precomputed
        :param gamma: kernel coefficient
        """

        self.classifier = SVC(C=C, kernel=kernel, gamma=gamma)

    def load_data(self, first_two_feature=False):
        """
        Load iris dataset
        :param first_two_feature: loads only the first 2 features from iris dataset, required for exercise 4.1 -e
        """
        iris = datasets.load_iris()
        if first_two_feature:
            self.X = iris.data[:, :2]
        else:
            self.X = iris.data
        self.y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30)

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def test(self):
        scores = self.classifier.score(self.X_test, self.y_test)
        return scores

    def make_meshgrid(self, x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(self, ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def plot(self):
        fig, ax = plt.subplots()
        title = 'Decision boundaries of SVM'

        # create and set up a meshgrid for plotting
        X0, X1 = self.X_test[:, 0], self.X_test[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        # predict the classes and plot the contours
        self.plot_contours(ax, self.classifier, xx, yy, cmap=plt.cm.coolwarm)

        # plot the data points on the grid
        ax.scatter(X0, X1, c=self.y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_title(title)
        plt.show()


# tries every combinations of the lists below, and returns the params which results the highest accuracy
def find_best_parameter_set(test_iter=100):
    C = [0.5, 1, 1.5]
    kernels = ['linear', 'poly', 'rbf']
    gamma = ['scale', 'auto']

    p = [C, kernels, gamma]
    best_params = None
    best_accuracy = 0
    print('Params\t\t\t\t\t -> accuracy')
    for param in list(itertools.product(*p)):
        accuracies = []
        for i in range(test_iter):
            svm = SVM(C=param[0], kernel=param[1], gamma=param[2])
            svm.load_data()
            svm.train()
            accuracies.append(svm.test())
        avg_accuracy = sum(accuracies) / len(accuracies)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = param
        print('{} -> {}'.format(param, avg_accuracy))
    return best_accuracy, best_params


if __name__ == '__main__':
    # exercise D
    acc, params = find_best_parameter_set()
    print('Best params are {} with accuracy {}'.format(params, acc))

    # exercise E
    svm = SVM(params[0], params[1], params[2])
    svm.load_data(first_two_feature=True)
    svm.train()
    svm.plot()
