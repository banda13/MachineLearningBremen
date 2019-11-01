import math

def expectation(set):

    exp = 0

    for i in set:
        exp += i

    return exp / len(set)

def covariance(featureA, featureB):

    sum = 0


    expA = expectation(featureA)
    expB = expectation(featureB)

    for i in range(len(featureA)):

        sum += (featureA[i] - expA)*(featureB[i] - expB)

    return sum

def stdDeviation(feature):

    exp = expectation(feature)
    sum = 0

    for instance in feature:
        sum += math.pow(instance-exp,2)

    return math.sqrt(sum)

def calculatePearsonCorCoeff(featureA, featureB):

    return covariance(featureA,featureB)/(stdDeviation(featureA)*stdDeviation(featureB))

#featureA on first row, featureB on second, ....
X = [[-2,-2,2,0.2],[-1,-1,1,0.1],[0,1.4,-1,-1],[-1,-1,1,1]]

feature_class_correlation = [calculatePearsonCorCoeff(X[0], X[3]), calculatePearsonCorCoeff(X[1], X[3]), calculatePearsonCorCoeff(X[2], X[3])]
feature_feature_correlation = [calculatePearsonCorCoeff(X[0], X[1]), calculatePearsonCorCoeff(X[0], X[2]), calculatePearsonCorCoeff(X[1], X[2])]

print('correlation feature A and class = {}'.format(feature_class_correlation[0]))
print('correlation feature B and class = {}'.format(feature_class_correlation[1]))
print('correlation feature C and class = {}'.format(feature_class_correlation[2]))

print('correlation feature A and B = {}'.format(feature_feature_correlation[0]))
print('correlation feature A and C = {}'.format(feature_feature_correlation[1]))
print('correlation feature B and C = {}'.format(feature_feature_correlation[2]))

exp_feature_class = expectation(feature_class_correlation)
exp_feature_feature = expectation(feature_feature_correlation)

print('mean feature class = {}'.format(exp_feature_class))
print('mean feature feature = {}'.format(exp_feature_feature))
