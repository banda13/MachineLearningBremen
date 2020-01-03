#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

class Evaluation:
    """This class provides functions for evaluating classifiers """
        
    def generate_cv_pairs(self, n_samples, y, n_folds=5, n_rep=1):
        """ Train and test pairs according to stratified k-fold cross validation with randomization

        Parameters
        ----------
        n_samples : int
            The number of samples in the dataset

        y : array-like, shape (n_samples),
            The labels of the data.

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation

        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation

        Returns
        -------
        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
        cv_splits = []
        for i in range(n_rep):
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
            cv_splits.extend(list(cv.split(np.zeros(n_samples),y)))
        return cv_splits
    
    def apply_cv(self, X, y, train_test_pairs, classifier, **kwargs):
        """ Evaluate classifier on testing data and return confusion matrix 
        
        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation
        
        y : array-like, shape (n_samples)
            The actual labels for the samples
        
        train_test_pairs : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split
        
        classifier : function
            Function that trains a classifier and returns the predictions for 
            the testing data. Arguments of the functions are the training
            data, the correct labels for the training data, the testing data.
            Further, keyword arguments given to *apply_cv* are passed to this
            function.
        
        Returns
        -------
        confusion_matrix : array-like, shape (n_classes, n_classes)
            The averaged confusion matrix (rows actual values, columns predicted
            values) across all test splits.
        """
        
        n_labels = np.unique(y).size
        confusion_matrix = np.zeros((n_labels, n_labels))
        n = len(train_test_pairs)
        for train_index, test_index in train_test_pairs:
            predictions = classifier(X[train_index,:], y[train_index], 
                X[test_index,:], **kwargs)
            confusion_matrix += self.confusion_matrix(predictions, y[test_index])
        confusion_matrix /= float(n)
        return confusion_matrix
    
    def confusion_matrix(self, predictions, labels):
        """ Return normalized confusion matrix 
        
        Computes the confusion matrix and normalizes it, so that all values sum
        up to one.
        
        Parameters
        ----------
        predictions : array-like, shape (n_samples)
            The classification outcome
        
        labels : array-like, shape (n_samples)
            The actual labels for the samples
        
        Returns
        -------
        confusion_matrix : array-like, shape (n_classes, n_classes)
            A normalized confusion matrix, where all entries sum up to 1.
        """
        confusion_matrix = [[0, 0], [0, 0]]
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                if predictions[i] == 0:
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[1][1] += 1
            else:
                if predictions[i] == 0:
                    confusion_matrix[0][1] += 1
                else:
                    confusion_matrix[1][0] += 1
        confusion_matrix = np.divide(confusion_matrix, len(labels))
        return confusion_matrix



class LearningAlgorithms:
    """ Algorithms for supervised learning """
    
    def decision_tree(self, X_train, y_train, X_test, **kwargs):
        """ Train a decision tree and return test predictions 
        
        Parameters
        ----------
        X_train : array-like, shape (n_train-samples, feature_dim)
            The data used for training 
        
        y_train : array-like, shape (n_train-samples)
            The actual labels for the training data
        
        X_test : array-like, shape (n_test-samples, feature_dim)
            The data used for testing
        
        Returns
        -------
        predictions : array-like, shape (n_test-samples)
            The classification outcome
        
        """
        c = DecisionTreeClassifier(**kwargs)
        c.fit(X_train, y_train)
        return c.predict(X_test)
    
    def bagging(self, X_train, y_train, X_test, bag_size=None, 
            num_bags=10, base_learner=DecisionTreeClassifier, **kwargs):
        """ Build Bagging model on training data and return test predictions 
            
        ### THE SUMMARY OF YOUR ALGORITHM GOES HERE ###
            
        Parameters
        ----------
        X_train : array-like, shape (n_train-samples, feature_dim)
            The data used for training 
        
        y_train : array-like, shape (n_train-samples)
            The actual labels for the training data

        X_test : array-like, shape (n_test-samples, feature_dim)
            The data used for testing
        
        bag_size : int or None, optional (default: None)
            Number of instances used for training of an ensemble member. If
            None *bag_size* is set to n_train-samples.

        num_bags : int, optional (default: 10)
            Number of bags and hence number of ensemble learners.
            
        base_learner : class, optional (default: DecisionTreeClassifier)
            Scikit-learn classifier. Keyword arguments are passed to the class
            if an instance is created.
            
        Returns
        -------
        predictions : array-like, shape (n_test-samples)
            The classification outcome
        
        """
        c = base_learner(**kwargs)
        if bag_size is None:
            bag_size = len(X_train)

        all_pred = []

        # For each iteration it creates a bag with random elements of X_train and with replacement, trains the
        # classifier with the current bag and obtains the prediction for X_test
        for i in range(num_bags):
            # randomly draw indices of elements of X_train with replacement
            random_indices = np.random.choice(np.arange(len(X_train)), bag_size, replace=True)
            # train the classifier with current bag
            c.fit(X_train[random_indices, :], y_train[random_indices])
            # obtaining predictions
            result = c.predict(X_test)
            all_pred.append(result)
        all_pred = np.asarray(all_pred)

        predictions = []
        # For each iteration it considers the predictions of the different classifiers for the j-th sample of X_test
        # and select the most voted as final prediction
        for j in range(len(X_test)):
            # considering the predictions of the different classifiers for the sample j-th of X_test
            votes = all_pred[:, j]
            # obtaining the different labels predicted and the number of occurrences of each
            labels, occur = np.unique(votes, return_counts=True)
            # selecting, as final prediction, the most "voted"
            predictions.append(labels[np.argmax(occur)])
        return np.asarray(predictions)


def accuracy(conf_matrix):
    return (conf_matrix[0][0] + conf_matrix[1][1]) / (
            conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][1] + conf_matrix[1][0])


if __name__ == '__main__':
    
    # load diabetis dataset
    data = np.loadtxt('diabetes_data.csv', delimiter=',')
    X, y = data[:,1:], data[:,0]
    
    c = Evaluation()
    a = LearningAlgorithms()
    
    cv_splits = c.generate_cv_pairs(len(X), y, 10, 10)
    conf_matrix_simple= c.apply_cv(X, y, cv_splits, a.decision_tree)
    conf_matrix_ensemble = c.apply_cv(X, y, cv_splits, a.bagging)
    acc_simple = accuracy(conf_matrix_simple)
    acc_ensemble = accuracy(conf_matrix_ensemble)
    print(conf_matrix_simple)
    print(acc_simple)
    print(conf_matrix_ensemble)
    print(acc_ensemble)



