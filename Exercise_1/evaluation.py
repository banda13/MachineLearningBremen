#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)




class Evaluation:
    """This class provides functions for evaluating classifiers """

    # scikit-learn implementaion
    # https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/model_selection/_split.py#L360

    def generate_label_pairs(self, label_samples, n_folds=10, n_rep=1, rand=False):

        folder_dimension = label_samples/n_folds
        label_splits = []

        if rand == False:
                    ini = 0
                    end = folder_dimension
                    actual_shift = 0
                    for k in range(num_folds):
                        if k == 0:
                            label_splits.append((data_target_iris[1][:int(folder_dimension)],data_target_iris[1][int(folder_dimension):]))

                        elif k == range(num_folds):
                            label_splits.append((data_target_iris[1][int(ini+actual_shift):],data_target_iris[1][:int(ini+actual_shift)]))

                        else:
                            label_splits.append((data_target_iris[1][int(ini+actual_shift):int(end+actual_shift)],np.concatenate((data_target_iris[1][:int(ini+actual_shift)], data_target_iris[1][int(end+actual_shift):]))))


                        actual_shift = actual_shift + folder_dimension
        else: 
            print("I AM NOTHING")

        return label_splits


    def generate_cv_pairs(self, n_samples, n_folds=10, n_rep=1, rand=False, y=None):
        

        """ Train and test pairs according to k-fold cross validation

        Parameters
        ----------

        n_samples : int
            The number of samples in the dataset = 150

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation = 10
 
        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation (1 or 10 repetitions)

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
        
        ### YOUR IMPLEMENTATION GOES HERE ###

        # if random is true, shuffle both the data and labels from iris dataset in the same way
        if rand == True:
            shuffle_in_unison(data_target_iris[0],y)

        # dimension of folder, used in cycle above
        folder_dimension = n_samples/n_folds

        # instantiation of list of tuples
        cv_splits = []
        
        
        #utils for the following cycle
        ini = 0
        end = folder_dimension
        actual_shift = 0

        # crating the folders..
        for k in range(n_folds):
            if k == 0:
                cv_splits.append((data_target_iris[0][:int(folder_dimension)],data_target_iris[0][int(folder_dimension):]))

            elif k == range(n_folds):
                cv_splits.append((data_target_iris[0][int(ini+actual_shift):],data_target_iris[0][:int(ini+actual_shift)]))

            else:
                cv_splits.append((     data_target_iris[0][int(ini+actual_shift):int(end+actual_shift)]    ,   np.concatenate((data_target_iris[0][:int(ini+actual_shift)], data_target_iris[0][int(end+actual_shift):]))    ))


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
    eval = Evaluation()

    ### YOUR IMPLEMENTATION FOR PROBLEM 1.1 GOES HERE ###


    # returns the IRIS dataset as ``(data, target)``
    data_target_iris = load_iris(True)

    # choose here if you want random folders
    random = False

    # choose the number of the folds
    num_folds = 10
    repetitions = 1


    #if random == True:
          #  shuffle_in_unison(n_samples,y)
    


    samples = len(data_target_iris[0])

    cv_splits = eval.generate_cv_pairs(samples,num_folds,repetitions,random,data_target_iris[1])
    label_splits = eval.generate_label_pairs(samples,num_folds,repetitions,random)


    for k in range(num_folds):
        print(eval.black_box_classifier(cv_splits[k][1],cv_splits[k][0],label_splits[k][1],label_splits[k][0]))