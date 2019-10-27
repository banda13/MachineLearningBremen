# this is the code what we wrote on the class
# I tried to translate it to python but I'm not sure if its working properly
# But maybe this is a good start to implement the PCA

import numpy as np
import matplotlib.pyplot as plt

X = [[1, 2], [3, 1]]
X = np.array(X)
mean = np.mean(X)
# plt.plot(X, 'g')
X = X - mean
# plt.plot(X, 'r')
# plt.show()

sigma = np.cov(X)
print(sigma)

w, v = np.linalg.eig(X)
print(w)
print(v)

Y = X * w
print(Y)
plt.plot(Y, 'g')
plt.show()