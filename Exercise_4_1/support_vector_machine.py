import itertools
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
            X = iris.data[:, :2]
        else:
            X = iris.data
        y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.30)

    def train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def test(self):
        scores = self.classifier.score(self.X_test, self.y_test)
        return scores


def find_best_parameter_set(test_iter=100):
    C = [0.5, 1, 1.5]
    kernels = ['linear', 'poly', 'rbf']
    gamma = ['scale', 'auto']

    p = [C, kernels, gamma]
    print('Params\t\t\t\t\t -> accuracy')
    for param in list(itertools.product(*p)):
        accuracies = []
        for i in range(test_iter):
            svm = SVM(C=param[0], kernel=param[1], gamma=param[2])
            svm.load_data()
            svm.train()
            accuracies.append(svm.test())
        print('{} -> {}'.format(param, sum(accuracies) / len(accuracies)))


if __name__ == '__main__':
    find_best_parameter_set()