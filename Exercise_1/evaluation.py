#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier
from sklearn.datasets import load_iris


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def stratification(n_samples,n_folds, labels):
    cl_tp = 0
    vec_classes = np.unique(labels)
    new_data = []
    new_label = []
    copy_data = []
    k = 0
    n_elem_class = (n_samples/n_folds)/len(vec_classes)
    copy_data.append(np.copy(data_target_iris[0])) 
    copy_data.append(np.copy(data_target_iris[1]))
    counters_array = [0] * len(vec_classes)
    for i in range(n_folds):
        while k < len(copy_data[0]):
            for j in  range(len(vec_classes)):
                if labels[k] == vec_classes[j]:
                    cl_tp = j
                    break

            if counters_array[cl_tp] < n_elem_class:
                new_data.append(copy_data[0][k])
                new_label.append(copy_data[1][k])
                copy_data[0] = np.delete(copy_data[0],k,0)
                copy_data[1] = np.delete(copy_data[1],k)
                counters_array[cl_tp] += 1
            else:
                k += 1

            if np.sum(counters_array) == n_elem_class * len(vec_classes):
                counters_array = [0] * len(vec_classes)
                k = 0
                break

    result = (np.asarray(new_data), np.asarray(new_label))
    return result



class Evaluation:
    """This class provides functions for evaluating classifiers """

    # scikit-learn implementaion
    # https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/model_selection/_split.py#L360

    def generate_label_pairs(self, label_samples, n_folds, labels):

        folder_dimension = label_samples/n_folds
        label_splits = []

      
        ini = 0
        end = folder_dimension
        actual_shift = 0
        for k in range(num_folds):
            if k == 0:
                label_splits.append((labels[int(folder_dimension):],labels[:int(folder_dimension)]))

            elif k == range(num_folds):
                label_splits.append((labels[:int(ini+actual_shift)],labels[int(ini+actual_shift):]))

            else:
                label_splits.append((np.concatenate((labels[:int(ini+actual_shift)], labels[int(end+actual_shift):])),labels[int(ini+actual_shift):int(end+actual_shift)]))


            actual_shift = actual_shift + folder_dimension
   
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
        global data_target_iris 
        global y_ref
        if rand == True:
            shuffle_in_unison(data_target_iris[0],data_target_iris[1])

      
        if y is not None:
            working_data = stratification(n_samples,n_folds,y)
            y_ref = working_data[1]
           
            
            #print(working_data)
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
                cv_splits.append((working_data[0][int(folder_dimension):],working_data[0][:int(folder_dimension)]))

            elif k == range(n_folds):
                cv_splits.append((working_data[0][:int(ini+actual_shift)],working_data[0][int(ini+actual_shift):]))

            else:
                cv_splits.append(( np.concatenate((working_data[0][:int(ini+actual_shift)], working_data[0][int(end+actual_shift):])) ,  working_data[0][int(ini+actual_shift):int(end+actual_shift)]     ))


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
    y_ref = data_target_iris[1]

    # choose here if you want random folders
    random = True

    #chose if you want stratification
    strat = True

    if strat == False:
        y_ref = None
    
    # choose the number of the folds
    num_folds = 10
    repetitions = 1

    #if random == True:
          #  shuffle_in_unison(n_samples,y)
    


    samples = len(data_target_iris[0])

    cv_splits = eval.generate_cv_pairs(samples,num_folds,repetitions,random,y_ref)

    label_splits = eval.generate_label_pairs(samples,num_folds,y_ref)


    for k in range(num_folds):
        print(label_splits[k][1])
       #print(eval.black_box_classifier(cv_splits[k][0],cv_splits[k][1],label_splits[k][0],label_splits[k][1]))