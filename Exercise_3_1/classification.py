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
        #for each sample in the test test (i.e. the first folder)...
        for i in range(len(X_test)):
            #inizialize the min_dist to infinite and the min_label to None for each neighbors
            min_Dist = [float("inf")] * neighbors
            min_Label = [None] * neighbors
            #for each sample in the training test (i.e. the set of the others n-1 folders)...
            for j in range(len(X_train)):
                #calculate the distance between the current test sample and the train sample
                distance = metric(X_test[i], X_train[j], st_dev=st_dev)
                #take the index of the corrisponding maximum distance in the array min_Dist (e.g. at the start they are all infinite, so ind_max=0)
                ind_max = np.argmax(min_Dist)
                #if the distance calculated above is lower than the one extracted from min_distance we can exchange the values and save the label of j in the min_Label array
                if distance < min_Dist[ind_max]:
                    min_Dist[ind_max] = distance
                    min_Label[ind_max] = y_train[j]
            #then we take the most frequent values in the min_Label array and it will be the predicted label for the current sample in the testing set
            unique_labels, count_labels = np.unique(min_Label, return_counts=True)
            max_freq_label = np.argmax(count_labels)
            predicted_Label.append(unique_labels[max_freq_label])

        correct_labels = 0

        #calculate the accuracy by checking how many times the predicted label corresponds to the original (correct) one, and dividing by the total number of lable's predictions

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

    def manhattan_distance(self, data_a, data_b, **kwargs):
        
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
    k = 134
    i = 1

    #try with different values of neighbors, from i=1 to k (i.e. append in scores the accuracy of each classification with different neighborns)
    #the apply_k_fold_cv return an array, so that scores will be an array of array at the end of the while (each array has the same dimension of the number of fold in the cv)
    while i <=k:
        scores.append(c.apply_k_fold_cv(data[0], data[1], c.kNN_classifier, metric=c.normalized_euclidean_distance,
                                   n_folds=10, neighbors=i))
        i += 1
    print(scores)

    #for each array in scores calculate the mean of the accuracies
    #at the end avg_scores will be an array containing the mean accuracies (one for each number of neighbors' number)
    for i in range(len(scores)):
        sum_scores = 0
        #calculating the average score
        for score in scores[i]:
            sum_scores += score
        avg_score = sum_scores/len(scores[0])
        avg_scores.append(avg_score)
    #select the optimal number of neighbors, by picking the index of the array with corresponding max element
    optimal_k = np.argmax(avg_scores)
    print("The optimal value for k is: %s" % optimal_k)
    print("Where the accuracy is: %s" % avg_scores[optimal_k])
    #plot the scores variation with respect to the size of neighbors
    plot_results(avg_scores, k)
    #plot the value of the accuracies with manhattan and chebyschev distance with respect to the optimal value of the neighborns (foreach folder in cv)
    score_manhattan = c.apply_k_fold_cv(data[0], data[1], c.kNN_classifier, metric=c.manhattan_distance,
                                        n_folds=10, neighbors=optimal_k)
    score_chebyshev = c.apply_k_fold_cv(data[0], data[1], c.kNN_classifier, metric=c.chebyshev_distance,
                                        n_folds=10, neighbors=optimal_k)
    print("Accuracy vector with Manhattan distance: %s" %score_manhattan)
    plt.title("Accuracy vector with Manhattan distance (k=12)")
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], score_manhattan, label='Accuracy')
    plt.ylim((0, 2))
    plt.xlabel("Folder")
    plt.ylabel("Accuracy")
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.legend()
    plt.show()
    plt.title("Accuracy vector with Chebyshev distance (k=12)")
    print("Accuracy vector with Chebyshev distance: %s" %score_chebyshev)
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], score_chebyshev, label='Accuracy')
    plt.ylim((0, 2))
    plt.xlabel("Folder")
    plt.ylabel("Accuracy")
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.legend()
    plt.show()

