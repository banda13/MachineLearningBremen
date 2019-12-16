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
        self. tol = tol # error tolerance
        self. eps = eps # alpha tolerance
        self.w = 0


    def objectiveFunction(self, alphas = None):

        sum_1 = np.sum(self.alphas)
        sum_2 = 0
        for i in range(len(self.alphas)):
            for j in range(len(self.alphas)):
                sum_2 += 0.5 * self.alphas[i] * self.alphas[j] * self.y[i] * self.y[j] * self.kernel(self.X[i], self.X[j])
        return sum_1 - sum_2

    def decisionFunction(self, X_test):
        # return (alphas * self.y) @ self.kernel(self.X, X_test) - self.b
        return np.sign(self.w.T * X_test -b)

    def examineExample(self, i_2):

        y_2 = self.y[i_2]
        alpha_2 = self.alphas[i_2]
        E_2 = self.errors[i_2]
        r_2 = E_2 * y_2

        if ((r_2 < - self.tol and alpha_2 < self.C) or (r_2 > self.tol and alpha_2 > 0)):
            if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
                if E_2 <= 0:
                    i_1 = np.argmax(self.errors)
                elif E_2 > 0:
                    i_1 = np.argmin(self.errors)
                if self.takeStep(i_1, i_2):
                    return 1

            for i_1 in np.roll(np.where((self.alphas != 0) & (self.alphas != self.C))[0],
                              np.random.choice(np.arange(len(self.X)))):
                if self.takeStep(i_1, i_2):
                    return 1

            for i_1 in np.roll(np.arange(len(self.X)), np.random.choice(np.arange(len(self.X)))):
                if self.takeStep(i_1, i_2):
                    return 1
        return 0

    def takeStep(self, i_1, i_2):

        if (i_1 == i_2): return 0

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
            L = max(0, alpha_2 - alpha_1 - self.C)
            H = min(self.C, alpha_2 + alpha_1)
        if (L == H):
            return 0
        k_11 = self.kernel(self.X[i_1], self.X[i_1])
        k_12 = self.kernel(self.X[i_1], self.X[i_2])
        k_22 = self.kernel(self.X[i_2], self.X[i_2])
        second_derivative = k_11 + k_22 - 2 * k_12
        if (second_derivative > 0):
            new_alpha_2 = alpha_2 + y_2 * (E_1 - E_2)/second_derivative
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

        if(np.abs(new_alpha_2 - alpha_2) < self.eps * (new_alpha_2 + alpha_2 + self.eps)):
            return 0

        new_alpha_1 = alpha_1 + s * (alpha_2 - new_alpha_2)

        b_1 = E_1 + y_1 * (new_alpha_1 - alpha_1) * k_11 + y_2 * (new_alpha_2 - alpha_2) * k_12 + self.b
        b_2 = E_2 + y_1 * (new_alpha_1 - alpha_1) * k_12 + y_2 * (new_alpha_2 - alpha_2) * k_22 + self.b
        if 0 < new_alpha_1 and new_alpha_1 < self.C:
            new_b = b_1
        elif 0 < new_alpha_2 and new_alpha_2 < self.C:
            new_b = b_2
        else:
            new_b = (b_1+b_2) * 0.5

        self.alphas[i_1] = new_alpha_1
        self.alphas[i_2] = new_alpha_2
        self.b = new_b

        new_w = self.w + y_1 * (new_alpha_1 - alpha_1) * self.X[i_1] + y_2 * (new_alpha_2 - alpha_2) * self.X[i_2]
        self.w = new_w

        for i in range(len(self.errors)):
            u_i = 0
            for j in range(len(self.y)):
                u_i = self.y[j] * self.alphas[j] * self.kernel(self.X[i], self.X[j]) - self.b
            self.errors[i] = u_i - self.y[i]

        return 1

    def plot_decision_boundary(self, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        xrange = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), resolution)
        yrange = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), resolution)
        grid = [[self.decisionFunction(np.array([xr, yr])) for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))

        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(self.X[:, 0], self.X[:, 1],
                   c=self.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)

        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = self.alphas != 0.0
        ax.scatter(self.X[mask, 0], self.X[mask, 1],
                   c=self.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')

        return grid, ax


if __name__ == "__main__":

    X_train, y_train = load_data("smo_dataset.csv")
    C = 0.1
    m = len(X_train)
    alphas = np.zeros(m)
    errors = np.zeros(m)
    b = 0.0
    tol = 0.01
    eps = 0.01

    model = SMOmodel(X_train, y_train, C, linearKernel, alphas, b, errors, tol, eps)
    initial_error = model.decisionFunction(model.X) - model.y
    model.errors = initial_error

    numChanged = 0
    examineAll = 1
    while (numChanged > 0 or examineAll):
        numChanged = 0
        if(examineAll):
            for i in range(len(X_train)):
                numChanged += model.examineExample(i)
        else:
            for i in range(len(X_train)):
                if alphas[i] != 0 and alphas[i] != C:
                    numChanged += model.examineExample(i)
            examineAll = 1
        if(examineAll):
            examineAll = 0



    fig, ax = plt.subplots()
    grid, ax = model.plot_decision_boundary(ax)
    plt.show()














