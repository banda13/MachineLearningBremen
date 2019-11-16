#!/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random

class KMeans:
    
    def __init__(self, k=1, epochs=3):
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
        self.centroids = []



    def fit(self, X):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
        """
        exclude = []
        # Generate the first k centroid randomly
        for j in range(self.k):
            rand = random.choice([i for i in range(len(X)) if i not in exclude])
            self.centroids.append(X[rand])
            exclude.append(rand)
        labels = self.predict(X)
        self.plotResult(labels)
        for i in range(self.epochs):
            sum_X = [0] * self.k
            sum_Y = [0] * self.k
            mean_X = [0] * self.k
            mean_Y = [0] * self.k
            counter = [0] * self.k
            for j in range(len(labels)):
                sum_X[labels[j]] += X[j][0]
                sum_Y[labels[j]] += X[j][1]
                counter[labels[j]] +=1
            if counter.__contains__(0):
                raise Exception("One or more clusters become empty: restarting the method")
            for l in range(self.k):
                mean_X[l] = sum_X[l]/counter[l]
                mean_Y[l] = sum_Y[l]/counter[l]
                self.centroids[l] = [mean_X[l],mean_Y[l]]
            labels = self.predict(X)
            self.plotResult(labels)
        print(self.centroids)









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
        labels = []
        for observation in X:
            min = 999999
            centr_min = None
            for j in range(len(self.centroids)):
                distance_centr = np.sqrt(np.power(observation[0]-self.centroids[j][0], 2) + np.power(observation[1]-self.centroids[j][1],2))
                if distance_centr < min:
                    min = distance_centr
                    centr_min = j
            labels.append(centr_min)
        return labels

    def plotResult(self, labels):
        scatter = plt.scatter([item[0] for item in X], [item[1] for item in X], s=50, c=[item for item in labels])
        plt.scatter([item[0] for item in self.centroids], [item[1] for item in self.centroids], s=100, c='g', marker='*')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(*scatter.legend_elements(), loc="upper right", title="Cluster")
        plt.axis([0, 40, 0, 40])
        plt.show()

if __name__ == '__main__':
    # Load data
    X = np.genfromtxt('cluster_dataset2d.txt', delimiter=',')
    # Instance of the Kmeans class
    while True:
        try:
            c = KMeans(3)
            c.fit(X)
            labels = c.predict(X)
            break
        except Exception as e:
            print(e)
