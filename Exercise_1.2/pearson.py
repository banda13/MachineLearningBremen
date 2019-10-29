import math

def expectation(feature):

    exp = 0

    for i in feature:
        exp += i

    return exp / len(feature)

def covariance(featureA, featureB):

    sumA = 0
    sumB = 0

    expA = expectation(featureA)
    expB = expectation(featureB)

    for i in range(len(featureA)):

        sumA += featureA[i] - expA
        sumB += featureB[i] - expB

    return sumA*sumB

def stdDeviation(feature):

    exp = expectation(feature)
    sum = 0

    for instance in feature:
        sum += math.pow(instance-exp,2)

    return math.sqrt(sum)

def calculatePearsonCorCoeff(featureA, featureB):

    return covariance(featureA,featureB)/stdDeviation(featureA)*stdDeviation(featureB)

#featureA on first row, featureB on second, ....


X = [[-2,-2,2,0.2],[-1,-1,1,0.1],[0,1.4,-1,-1],[-1,-1,1,1]]



print('correlation feature A and class = {}'.format(calculatePearsonCorCoeff(X[0], X[3])))
print('correlation feature B and class = {}'.format(calculatePearsonCorCoeff(X[1], X[3])))
print('correlation feature C and class = {}'.format(calculatePearsonCorCoeff(X[2], X[3])))

print('correlation feature A and B = {}'.format(calculatePearsonCorCoeff(X[0], X[1])))
print('correlation feature A and C = {}'.format(calculatePearsonCorCoeff(X[0], X[2])))
print('correlation feature B and C = {}'.format(calculatePearsonCorCoeff(X[1], X[2])))
