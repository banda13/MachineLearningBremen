import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def plot_results(accuracies, k):
    plt.xlim(1, k)
    plt.xlabel('k')
    plt.plot(range(1,k+1), accuracies, 'r', label='Accuracy')
    plt.legend()
    plt.show()


class Classification:
    """This class holds different classification algorithms and the cv function
    """

    def standardDeviation(self, X_train):
        st_dev = []
        for i in range(len(X_train[0])):
            st_dev.append(np.std([item[i] for item in X_train]))
        return st_dev

    def apply_k_fold_cv(self, X, y, classifier=None, n_folds=5, **kwargs):
        """K fold cross validation.

        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            The data for the cross validation

        y : array-like, shape (n-samples, label_dim)
            The labels of the data used in the cross validation

        classifier : function
            The function that is used for classification of the training data

        n_splits : int, optional (default: 5)
            The number of folds for the cross validation

        kwargs :
            Further parameters that get used e.g. by the classifier

        Returns
        -------
        accuracies : array, shape (n_splits,)
            Vector of classification accuracies for the n_splits folds.
        """
        assert X.shape[0] == y.shape[0]

        if len(X.shape) < 2:
            X = np.atleast_2d(X).T
        if len(y.shape) < 2:
            y = np.atleast_2d(y).T

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for train_index, test_index in cv.split(X):
            train_data = X[train_index, :]
            train_label = y[train_index, :]
            test_data = X[test_index, :]
            test_label = y[test_index, :]

            score = classifier(train_data, test_data,
                               train_label, test_label, **kwargs)

            scores.append(score)

        return np.array(scores)

    def kNN_classifier(self, X_train, X_test, y_train, y_test,
                       neighbors=1, metric=None, **kwargs):
        """K nearest neighbor classifier.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, feature_dim)
            The data for the training of the classifier

        X_test : array-like, shape (n_samples, feature_dim)
            The data for the test of the classifier

        y_train : array-like, shape (n-samples, label_dim)
            The labels for the training of the classifier

        y_test : array-like, shape (n-samples, label_dim)
            The labels for the test of the classifier

        neighbors : int, optional (default: 1)
            The number of neighbors considered for the classification

        metric : function
            The function that is used as a metric for the kNN classifier

        Returns
        -------
        accuracy : double
            Accuracy of the correct classified test data
        """
        predicted_Label = []
        st_dev = self.standardDeviation(X_train)
        for i in range(len(X_test)):
            min_Dist = [float("inf")] * neighbors
            min_Label = [None] * neighbors
            for j in range(len(X_train)):
                distance = metric(X_test[i], X_train[j], st_dev=st_dev)
                ind_max = np.argmax(min_Dist)
                if distance < min_Dist[ind_max]:
                    min_Dist[ind_max] = distance
                    min_Label[ind_max] = y_train[j]
            unique_labels, count_labels = np.unique(min_Label, return_counts=True)
            max_freq_label = np.argmax(count_labels)
            predicted_Label.append(unique_labels[max_freq_label])

        correct_labels = 0

        for i in range(len(predicted_Label)):
            if predicted_Label[i] == y_test[i]:
                correct_labels += 1

        accuracy = correct_labels/len(predicted_Label)

        return accuracy



    def normalized_euclidean_distance(self, data_a, data_b, **kwargs):
        """Normalized euclidean distance metric"""
        result = 0
        st_dev = kwargs.get("st_dev")
        for i in range(len(data_a)):
            result += np.power(data_a[i]-data_b[i], 2)/np.power(st_dev[i], 2)
        result = np.sqrt(result)
        return result

    def manhattan_Distance(self, data_a, data_b, **kwargs):
        
        """Distance metric of your choice"""

        result = 0
        for i in range(len(data_a)):
            result += abs(data_a[i]-data_b[i])
        return result

    def chebyshev_distance(self, data_a, data_b, **kwargs):
        """Distance metric of your choice"""

        distances = []
        for i in range(len(data_a)):
            distances.append(np.abs(data_a[i]-data_b[i]))

        return np.max(distances)


if __name__ == '__main__':

    # Instance of the Classification class holding the distance metrics and
    # classification algorithm
    c = Classification()
    data = load_iris(True)
    scores = []
    avg_scores = []
    k = 100
    i = 1
    while i <=k:
        scores.append(c.apply_k_fold_cv(data[0], data[1], c.kNN_classifier, metric=c.normalized_euclidean_distance,
                                   n_folds=5, neighbors=i))
        i += 1
    print(scores)

    for i in range(len(scores)):
        sum_scores = 0
        for score in scores[i]:
            sum_scores += score
        avg_score = sum_scores/len(scores[0])
        avg_scores.append(avg_score)
    print(avg_scores)

    plot_results(avg_scores, k)
