import math
import itertools
import numpy as np

# Calculate Mean
def Mean(values):
    sum = 0
    for value in values:
        sum += value
    return sum/len(values)

# Calculate Variance
def Variance(values):
    mean = Mean(values)
    sum = 0
    for value in values:
        sum += pow(value-mean,2)
    return sum/len(values)

# Calculate Covariance
def Covariance(values1, values2):
    mean1 = Mean(values1)
    mean2 = Mean(values2)
    sum = 0
    for i in range(len(values1)):
        sum += (values1[i] - mean1) * (values2[i] - mean2)
    return sum/(len(values1))

# Calculate the standard deviation
def StandardDeviation(values):
    variance = Variance(values)
    return math.sqrt(variance)

# Calculate the Pearson product-moment correlation coefficient
def Pearson(features1, features2):

    covariance = Covariance(features1,features2)
    stdDev1 = StandardDeviation(features1)
    stdDev2 = StandardDeviation(features2)

    return covariance/(stdDev1 * stdDev2)

# Calculate the merit of a subset of features
def Merit(features,classes):
    card = len(features)
    sumff = 0
    sumfc = 0
    couple = 0
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            sumff += abs(Pearson(features[i], features[j]))
            couple += 1
        sumfc += abs(Pearson(features[i], classes))
    if len(features)>1:
        merit = (abs(card)*sumfc/card)/math.sqrt(card+card*(card-1)*sumff/couple)
    else:
        merit = sumfc
    return merit

# It finds all the possible different subset of n elements
def FindSubset(n):
    s = [int(i) for i in range(n)]
    subset = list()
    for j in range(len(s)):
        subset += list(itertools.combinations(s, j+1))
    return subset


if __name__ == "__main__":

    #data = np.array([[-0.118, 0.934, 0.979, 0.152,-1], [-0.800, -0.489, 1.082, 0.024, -1], [-0.511, 0.475, 1.029, 0.089, -1], [0.120, -0.076, 1.291, 0.067, -1], [1.747, -1.276, -0.546, 0.307, 1], [0.897, 0.327, -1.291, 0.294, 1], [1.157, 0.432, -0.991, 0.031, 1], [0.573, -0.371, -1.037, 0.076, 1]])
    data = np.array([[-2,-1,0,-1],[-2,-1,1.4,-1],[2,1,-1,1],[-0.2,-0.1,-1,1]])
    subsets = FindSubset(len(data[0])-1)
    classes = data[:,len(data[0])-1]
    features = list()
    merits = list()
    for subset in subsets:
        features.clear()
        for element in subset:
            features.append(data[:,element])
        merits.append(Merit(features,classes))

    print("Subset of features             Merit")
    for i in range(len(subsets)):
        print(f"{subsets[i]}                      {merits[i]}")
    print("\n Answer 1.2-a")
    print("Feature - Class correlations")
    print("1) %s" % Pearson(data[:, 0], classes))
    print("2) %s" % Pearson(data[:, 1], classes))
    print("3) %s" % Pearson(data[:, 2], classes))
    print("\nIf k=2 Naive selector will choose feature 1 and feature 2")
    print("It's because these two features are the ones with the highest feature-class correlation.")
    print("\nAnswer 1.2-b")
    print("According to the merit the subsets with cardinality 2 that we can choose are both (0,2)=(A,C) and (1,2)=(B,C) because they have highest merit")
    print("\nAnswer 1.2-c")
    print("If we could choose from an arbitrary cardinality of subset, we will choose one of the same two subsets")














