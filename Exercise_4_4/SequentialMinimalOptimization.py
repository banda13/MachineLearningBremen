import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linearKernel(x, y, b=1):
    return x @ y.T + b


def load_data(filename):
    data = pd.read_csv(filename, sep=";")
    X = data.iloc[:, 0:2].values.astype(float)
    y = data.iloc[:, 2].values.astype(float)
    return X, y


class SMOmodel:

    def __init__(self, X, y, C, kernel, alphas, b, errors, tol, eps):
        self.X = X  # training data vector
        self.y = y  # class label vector
        self.C = C  # regularization parameter
        self.kernel = kernel  # kernel function
        self.alphas = alphas  # lagrange multiplier vector
        self.b = b  # scalar bias term
        self.errors = errors  # error cache
        self.tol = tol  # error tolerance
        self.eps = eps  # alpha tolerance
        self.w = np.zeros(len(X[0, :]))

    def objectiveFunction(self, alphas=None):
        if alphas is None:
            alphas = self.alphas
        sum_1 = np.sum(alphas)
        sum_2 = 0
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                sum_2 += 0.5 * alphas[i] * alphas[j] * self.y[i] * self.y[j] * self.kernel(self.X[i, :], self.X[j, :])
        return sum_1 - sum_2

    def evaluate(self, X_test):
        return self.w @ X_test.T - self.b

    def examineExample(self, i_2):

        y_2 = self.y[i_2]
        alpha_2 = self.alphas[i_2]
        E_2 = self.errors[i_2]
        r_2 = E_2 * y_2

        if (r_2 < - self.tol and alpha_2 < self.C) or (r_2 > self.tol and alpha_2 > 0):
            num = len(self.alphas[(self.alphas > 0) & (self.alphas < self.C)])
            if num > 1:
                if E_2 > 0:
                    i_1 = np.argmin(self.errors)
                elif E_2 <= 0:
                    i_1 = np.argmax(self.errors)
                if self.takeStep(i_1, i_2):
                    return 1

            random_indices = np.random.permutation(len(self.y))
            for i_1 in random_indices:
                if self.alphas[i_1] > 0 and self.alphas[i_1] < self.C:
                    if self.takeStep(i_1, i_2):
                        return 1

            random_indices = np.random.permutation(len(self.y))
            for i_1 in random_indices:
                if self.takeStep(i_1, i_2):
                    return 1
        return 0

    def takeStep(self, i_1, i_2):

        if i_1 == i_2:
            return 0

        alpha_1 = self.alphas[i_1]
        alpha_2 = self.alphas[i_2]
        y_1 = self.y[i_1]
        y_2 = self.y[i_2]
        E_1 = self.errors[i_1]
        E_2 = self.errors[i_2]
        s = y_1 * y_2
        if y_1 != y_2:
            L = max(0, alpha_2 - alpha_1)
            H = min(self.C, self.C + alpha_2 - alpha_1)
        else:
            L = max(0, alpha_2 + alpha_1 - self.C)
            H = min(self.C, alpha_2 + alpha_1)
        if (L == H):
            return 0
        k_11 = self.kernel(self.X[i_1], self.X[i_1])
        k_12 = self.kernel(self.X[i_1], self.X[i_2])
        k_22 = self.kernel(self.X[i_2], self.X[i_2])

        eta = k_11 + k_22 - 2 * k_12
        if eta > 0:
            new_alpha_2 = alpha_2 + y_2 * (E_1 - E_2) / eta
            if new_alpha_2 >= H:
                new_alpha_2 = H
            elif new_alpha_2 <= L:
                new_alpha_2 = L
        else:
            alpha_adj = self.alphas.copy()
            alpha_adj[i_2] = L
            Lobj = self.objectiveFunction(alpha_adj)
            alpha_adj[i_2] = H
            Hobj = self.objectiveFunction(alpha_adj)
            if Lobj > (Hobj + self.eps):
                new_alpha_2 = L
            elif Lobj < (Hobj - self.eps):
                new_alpha_2 = H
            else:
                new_alpha_2 = alpha_2

        if (np.abs(new_alpha_2 - alpha_2) < self.eps * (new_alpha_2 + alpha_2 + self.eps)):
            return 0

        new_alpha_1 = alpha_1 + s * (alpha_2 - new_alpha_2)

        b_1 = E_1 + y_1 * (new_alpha_1 - alpha_1) * k_11 + y_2 * (new_alpha_2 - alpha_2) * k_12 + self.b
        b_2 = E_2 + y_1 * (new_alpha_1 - alpha_1) * k_12 + y_2 * (new_alpha_2 - alpha_2) * k_22 + self.b
        if 0 < new_alpha_1 < self.C:
            new_b = b_1
        elif 0 < new_alpha_2 < self.C:
            new_b = b_2
        else:
            new_b = (b_1 + b_2) * 0.5

        self.b = new_b
        new_w = self.w + y_1 * (new_alpha_1 - alpha_1) * self.X[i_1, :] + y_2 * (new_alpha_2 - alpha_2) * self.X[i_2, :]
        self.w = new_w

        self.errors = self.evaluate(self.X) - self.y

        self.alphas[i_1] = new_alpha_1
        self.alphas[i_2] = new_alpha_2

        return 1

    def plot_decision_boundary(self, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 40)
        yy = np.linspace(ylim[0], ylim[1], 40)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.evaluate(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        # plot support vectors
        support_vectors = self.X[np.round(self.alphas, decimals=3) != 0.0]
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                   linewidth=0.5, facecolors='none', edgecolors='k')
        plt.show()


if __name__ == "__main__":

    X_train, y_train = load_data("smo_dataset.csv")
    C = 0.1
    m = len(X_train)
    alphas = np.zeros(m)
    errors = np.zeros(m)
    b = 0.0
    tol = 0.1
    eps = 0.1

    model = SMOmodel(X_train, y_train, C, linearKernel, alphas, b, errors, tol, eps)
    initial_error = model.evaluate(model.X) - model.y
    model.errors = initial_error

    numChanged = 0
    examineAll = 1
    while numChanged > 0 or examineAll:
        numChanged = 0
        if examineAll:
            for i in range(len(X_train)):
                numChanged += model.examineExample(i)
        else:
            for i in range(len(X_train)):
                if model.alphas[i] > 0 and model.alphas[i] < model.C:
                    numChanged += model.examineExample(i)
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1

    print(f"hyperplane = {model.w}*x - {model.b}")
    fig, ax = plt.subplots()
    ax = model.plot_decision_boundary(ax)
