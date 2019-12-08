import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt


class supportVectorRegression(object):
    def __init__(self, kernel='rbf', C=1, gamma='auto', degree=3, epsilon=.1,coef0=1):
        self.regressor = SVR(kernel=kernel, C=C, gamma=gamma, degree=degree, epsilon=epsilon,coef0=coef0)
        self.label = kernel

    def loadData(self, data_path="schwefel.csv"):
        data = pd.read_csv(data_path)
        self.X = data.iloc[:, 0:1].values.astype(float).ravel().reshape(-1, 1)
        self.y = data.iloc[:, 1:2].values.astype(float).ravel()

    def fit(self):
        self.regressor.fit(self.X, self.y)

    def predict(self):
        self.prediction = self.regressor.predict(self.X)

    def plotResult(self,color):
        plt.plot(self.X,self.prediction,color=color,lw=2,label='{} model'.format(self.label))
        plt.scatter(self.X[self.regressor.support_],self.y[self.regressor.support_], facecolor="none",
                          edgecolor=color, s=50, label='{} support vectors'.format(self.label))
        plt.legend(loc='upper center', bbox_to_anchor=(0.8, 0.15),
                         ncol=1, fancybox=True, shadow=True)
        plt.title("Support Vector Regression")
        plt.show()

if __name__ == '__main__':

    #to be finished
    svr = supportVectorRegression(kernel='poly',C=20, degree=2, coef0=1)
    svr.loadData()
    svr.fit()
    svr.predict()
    svr.plotResult(color='g')
