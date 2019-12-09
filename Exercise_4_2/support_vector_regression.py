import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class supportVectorRegression(object):
    def __init__(self, kernel='rbf', C=1, gamma='auto', degree=3, epsilon=.1, coef0=1):
        self.regressor = SVR(kernel=kernel, C=C, gamma=gamma, degree=degree, epsilon=epsilon, coef0=coef0)
        self.kernel_type = kernel

    def load_data(self):
        data = pd.read_csv("schwefel.csv")
        self.X = data.iloc[:, 0:1].values.astype(float)
        self.y = data.iloc[:, 1:2].values.astype(float)

    def fit(self):
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        self.X = sc_X.fit_transform(self.X)
        self.y = sc_y.fit_transform(self.y).ravel()
        self.regressor.fit(self.X, self.y)

    def predict(self):
        self.prediction = self.regressor.predict(self.X)

    def plot_result(self, color):
        plt.plot(self.X, self.prediction, color=color, lw=2, label='{} model'.format(self.kernel_type))
        plt.scatter(self.X[self.regressor.support_], self.y[self.regressor.support_], facecolor="none",
                    edgecolor=color, s=50, label='{} support vectors'.format(self.kernel_type))
        plt.legend(loc='upper center', bbox_to_anchor=(0.8, 0.15),
                   ncol=1, fancybox=True, shadow=True)
        plt.title(self.kernel_type+" support vector regression")
        plt.show()


if __name__ == '__main__':
    # a)
    svrPoly = supportVectorRegression(kernel='poly', C=100, gamma='auto', degree=5, epsilon=.1, coef0=1)
    svrRBF = supportVectorRegression()
    svrLinear = supportVectorRegression(kernel='linear')

    #svr execution and plotting results
    svrs = [('m', svrPoly), ('c', svrRBF), ('g', svrLinear)]
    for color, svr in svrs:
        svr.load_data()
        svr.fit()
        svr.predict()
        svr.plot_result(color=color)

