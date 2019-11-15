#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
    
    def __init__(self, k=1, epochs=1):
        """ Init stuff

        Parameters
        ----------

        k : int
            Number of clusters
        
        epochs : int
            Number of epochs to run 

        """
        self.k = k
        self. epochs = epochs

    def fit(self, X):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        """

        centers = []
        # Generate the first k centroid randomly
        for i in range(self.k):
            centers = np.append(centers, random.choice([i for i in range(len(X)) if i not in centers]))


    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        ### YOUR IMPLEMENTATION GOES HERE ###
        

if __name__ == '__main__':
    # Load data
    X = np.genfromtxt('cluster_dataset2d.txt', delimiter=',')

    # Instance of the Kmeans class
    c = KMeans(5)
    c.fit(X)
    c.predict(X)
