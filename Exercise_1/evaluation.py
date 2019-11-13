#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris


class Evaluation:
    """This class provides functions for evaluating classifiers """

    def stratification(self, n_folds, ind_array, y):

        """ This function does the stratification of the data. More precisely,
            it uses an array of counters, with dimension equal to the number of
            classes of the dataset,  to organize the data in a way such that each folder
            will have the same number of instances of each class. This is realized by coping
            the indices array and visiting it sequentially until an element of a class with
            the corresponding counter less than necessary quantity (=n_elem_class) is found.
            At this point the index of this element is added at the result array and is cancelled
            in the copied array.

            Parameters
            ----------

            n_folds : int
                The number of the folders in which the dataset will be divided

            ind_array: array-like, shape (n_samples)
                Array containing the indices of the data

            y : array-like, shape (n_samples)
                labels of the data
            """

        new_array = []
        k = 0
        cl_tp = 0
        class_array, counts = np.unique(y, return_counts=True)
        counts = np.floor_divide(counts,n_folds)
        counters_array = np.copy(counts)
        temp_array = np.copy(ind_array)
        n_elem_class = (len(data[0]) / n_folds) / len(class_array)
        for i in range(n_folds):
            while k < len(temp_array):
                for j in range(len(class_array)):
                    if y[temp_array[k]] == class_array[j]:
                        cl_tp = j
                        break

                if counters_array[cl_tp] > 0:
                    new_array.append(temp_array[k])
                    temp_array = np.delete(temp_array,k)
                    counters_array[cl_tp] -= 1
                else:
                    k += 1

                if np.sum(counters_array) == 0:
                    counters_array = np.copy(counts)
                    k = 0
                    break
        return new_array

    def generate_cv_pairs(self, n_samples, n_folds=5, n_rep=1, rand=False,
                          y=None):
        """ Train and test pairs according to k-fold cross validation

        Parameters
        ----------

        n_samples : int
            The number of samples in the dataset

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation

        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation

        rand : boolean, optional (default: False)
            If True the data is randomly assigned to the folds. The order of the
            data is maintained otherwise. Note, *n_rep* > 1 has no effect if
            *random* is False.

        y : array-like, shape (n_samples), optional (default: None)
            If not None, cross validation is performed with stratification and
            y provides the labels of the data.

        Returns
        -------

        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """

        indices_array = np.arange(0,n_samples)
        cv_splits = []
        working_data = ()
        folder_dimension = n_samples/n_folds
        if rand == False:
            n_rep = 1

        for i in range(n_rep):
            if rand == True:
                np.random.shuffle(indices_array)
                if y is not None:
                    indices_array = self.stratification(n_folds, indices_array, y)
                else:
                    working_data = data
            else:
                working_data = data

            ini = 0
            end = folder_dimension
            actual_shift = 0

            for k in range(n_folds):
                if k == 0:
                    cv_splits.append((indices_array[int(folder_dimension):], indices_array[:int(folder_dimension)]))

                elif k == n_folds-1:
                    cv_splits.append(
                        (indices_array[:int(ini + actual_shift)], indices_array[int(ini + actual_shift):]))

                else:
                    cv_splits.append((np.concatenate(
                        (indices_array[:int(ini + actual_shift)], indices_array[int(end + actual_shift):])),
                                      indices_array[int(ini + actual_shift):int(end + actual_shift)]))

                actual_shift = actual_shift + folder_dimension

        return cv_splits


    def apply_cv(self, X, y, train_test_pairs, classifier):
        """ Use cross validation to evaluate predictions and return performance

        Apply the metric calculation to all test pairs

        Parameters
        ----------

        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation

        y : array-like, shape (n-samples)
            The actual labels for the samples

        train_test_pairs : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split

        classifier : function
            Function that trains and tests a classifier and returns a
            performance measure. Arguments of the functions are the training
            data, the testing data, the correct labels for the training data,
            and the correct labels for the testing data.

        Returns
        -------

        performance : float
            The average metric value across train-test-pairs
        """
        ### YOUR IMPLEMENTATION GOES HERE ###
        sum = 0
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for i in range(len(train_test_pairs)):
            for j in range(len(train_test_pairs[i][0])+len(train_test_pairs[i][1])):
                if j < len(train_test_pairs[i][0]):
                    X_train.append(X[train_test_pairs[i][0][j]])
                    Y_train.append(y[train_test_pairs[i][0][j]])
                else:
                    X_test.append((X[train_test_pairs[i][1][j-len(train_test_pairs[i][0])]]))
                    Y_test.append((y[train_test_pairs[i][1][j-len(train_test_pairs[i][0])]]))
            sum += classifier(X_train,X_test,Y_train,Y_test)

        return sum/len(train_test_pairs)



    def black_box_classifier(self, X_train, X_test, y_train, y_test):
        """ Learn a model on the training data and apply it on the testing data

        Parameters
        ----------

        X_train : array-like, shape (n_samples, feature_dim)
            The data used for training

        X_test : array-like, shape (n_samples, feature_dim)
            The data used for testing

        y_train : array-like, shape (n-samples)
            The actual labels for the training data

        y_test : array-like, shape (n-samples)
            The actual labels for the testing data

        Returns
        -------

        accuracy : float
            Accuracy of the model on the testing data
        """
        bbc = BlackBoxClassifier(n_neighbors=10)
        bbc.fit(X_train, y_train)
        acc = bbc.score(X_test, y_test)
        return acc

if __name__ == '__main__':
    # Instance of the Evaluation class

    label_pairs = []
    eval = Evaluation()
    n_rep = 10
    rand = True
    data = load_iris(True)
    n_samples = len(data[0])
    n_folds = 10
    cv_pairs = eval.generate_cv_pairs(n_samples, n_folds, n_rep, rand, data[1])
    print(eval.apply_cv(data[0],data[1],cv_pairs,eval.black_box_classifier))
    #print(cv_pairs)
