import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class supportVectorRegression(object):
    def __init__(self, kernel='rbf', C=1, gamma='auto', degree=3, epsilon=.1, coef0=1):
        self.regressor = SVR(kernel=kernel, C=C, gamma=gamma, degree=degree, epsilon=epsilon, coef0=coef0)
        if isinstance(kernel, str):
            self.kernel_type = kernel
        else:
            self.kernel_type = 'custom kernel'

    def load_data(self, data=pd.read_csv("schwefel.csv")):
        self.X = data.iloc[:, 0:1].values.astype(float)
        self.y = data.iloc[:, 1:2].values.astype(float)

    def fit(self):

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        self.X = sc_X.fit_transform(self.X)
        self.y = sc_y.fit_transform(self.y).ravel()

        self.regressor.fit(self.X, self.y.ravel())

    def predict(self):
        self.prediction = self.regressor.predict(self.X)

    def plot_result(self, color):
        plt.plot(self.X, self.prediction, color=color, lw=2, label='{} model'.format(self.kernel_type))
        plt.scatter(self.X[self.regressor.support_], self.y[self.regressor.support_], facecolor="none",
                    edgecolor=color, s=50, label='{} support vectors'.format(self.kernel_type))
        plt.scatter(self.X[np.setdiff1d(np.arange(len(self.X)), self.regressor.support_)],
                     self.y[np.setdiff1d(np.arange(len(self.X)), self.regressor.support_)],
                     facecolor="none", edgecolor="k", s=50,
                     label='other training data')
        plt.legend(loc='upper center', bbox_to_anchor=(0.8, 0.15),
                   ncol=1, fancybox=True, shadow=True)
        plt.title(self.kernel_type+" support vector regression")
        plt.show()


def my_kernel(X,Y):
    #NOTE: this is an RBF kernel, it still must be defined
    K = np.zeros((X.shape[0],Y.shape[0]))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            K[i,j] = np.exp(-1*np.linalg.norm(x-y)**2)
    return K

if __name__ == '__main__':
    # a)
    C = 10000
    epsilon = 0.1
    gamma = 0.1

    data = pd.read_csv("schwefel.csv")
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(data.iloc[:, 0:1].values.astype(float))
    Y = sc_y.fit_transform(data.iloc[:, 1:2].values.astype(float))

    svrPoly = supportVectorRegression(kernel='poly', C=C, gamma=gamma, degree=3, epsilon=epsilon, coef0=1)
    svrRBF = supportVectorRegression(C=C, epsilon=epsilon, gamma=gamma)
    svrLinear = supportVectorRegression(kernel='linear', C=C, epsilon=epsilon, gamma=gamma)
    svrCustom = supportVectorRegression(kernel=my_kernel, C=C, epsilon=epsilon, gamma=gamma)

    #svr execution and plotting results
    svrs = [('m', svrPoly), ('c', svrRBF), ('g', svrLinear), ('b', svrCustom)]
    for color, svr in svrs:
        svr.load_data(data=data)
        svr.fit()
        svr.predict()
        svr.plot_result(color=color)

