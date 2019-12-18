import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler


def linearKernel(x, y):

    return x @ y.T

def gaussian_kernel(x, y, sigma=1):
    
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result


def load_data(filename):
    data = pd.read_csv(filename, sep=";")
    X = data.iloc[:, 0:2].values.astype(float)
    y = data.iloc[:, 2].values.astype(float)
    return X, y


class SMO:

    def __init__(self, dataset, C, kernel, tol, eps):
        self.X, self.y = self.chooseDataset(dataset)  # data and label
        self.C = C  # Regularization parameter
        self.kernel = self.chooseKernel(kernel)  # kernel function
        self.alphas = np.zeros(len(self.X))  # Lagrange multiplier vector
        self.b = 0.0  # bias
        self.errors = self.evaluate(self.X) - self.y  # error cache
        self.tol = tol  # KKT tolerance
        self.eps = eps  # alpha tolerance
        self.w = np.zeros(len(self.X[0, :]))  # w vector

    def chooseDataset(self, dataset_name):

        """ Given the name of dataset chosen, it return the samples X_train and the labels Y_train"""

        if dataset_name == "circle":
            X_train, y_train = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
            scaler = StandardScaler()
            scaler.fit_transform(X_train, y_train)
            y_train[y_train == 0] = -1
        elif dataset_name == "moon":
            X_train, y_train = make_moons(n_samples=500, noise=0.1, random_state=1)
            scaler = StandardScaler()
            scaler.fit_transform(X_train, y_train)
            y_train[y_train == 0] = -1
        else:
            X_train, y_train = load_data("smo_dataset.csv")

        return X_train, y_train

    def chooseKernel(self, kernel_name):

        """Given the name of the kernel chosen, it returns the correspondent kernel function"""

        if kernel_name == "linear":
            return linearKernel
        else:
            return gaussian_kernel

    def objectiveFunction(self, alphas=None):

        """Returns the objective function of the SVM"""

        if alphas is None:
            alphas = self.alphas
        sum_1 = np.sum(alphas)
        sum_2 = 0
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                sum_2 += 0.5 * alphas[i] * alphas[j] * self.y[i] * self.y[j] * self.kernel(self.X[i, :], self.X[j, :])
        return sum_1 - sum_2

    def evaluate(self, X_test):

        """Applies the SVM decision function to the X_test features vector"""

        return (self.alphas * self.y) @ self.kernel(self.X, X_test) - self.b

    def mainRoutine(self):

        """Applies the Main Routine of SMO algorithm. In particular, here is realized the choice of the firs
           Lagrange multiplier to optimize"""

        numChanged = 0
        examineAll = 1
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in range(len(self.X)):
                    numChanged += self.examineExample(i)
            else:
                for i in range(len(self.X)):
                    if self.alphas[i] != 0 and self.alphas[i] != self.C:
                        numChanged += self.examineExample(i)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

        return True

    def examineExample(self, i_2):

        """Applies the Examine Example procedure of SMO algorithm. It implements the choice of the second Lagrange
           multiplier with the heuristic hierarchy and it passes the indices of the two multipliers to the takeStep
           function.
        """

        y_2 = self.y[i_2]
        alpha_2 = self.alphas[i_2]
        E_2 = self.errors[i_2]
        r_2 = E_2 * y_2

        # KKT conditions
        if (r_2 < - self.tol and alpha_2 < self.C) or (r_2 > self.tol and alpha_2 > 0):
            # Hierarchy of Heuristics
            # 1st heuristic
            num = len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)])
            if num > 1:
                if E_2 > 0:
                    i_1 = np.argmin(self.errors)
                elif E_2 <= 0:
                    i_1 = np.argmax(self.errors)
                if self.takeStep(i_1, i_2):
                    return 1

            # 2nd heuristic: "Loop through non-zero and non-C alphas, starting at a random point"
            random_indices = np.random.permutation(len(self.y))
            for i_1 in random_indices:
                if self.alphas[i_1] != 0 and self.alphas[i_1] != self.C:
                    if self.takeStep(i_1, i_2):
                        return 1

            # 3rd heuristic: "Loop through all alphas, starting at a random point"
            random_indices = np.random.permutation(len(self.y))
            for i_1 in random_indices:
                if self.takeStep(i_1, i_2):
                    return 1
        return 0

    def takeStep(self, i_1, i_2):

        """ Given the indices of the two Lagrange multipliers to optimize, it calculates the new values for alpha_1,
            alpha_2 and b and updates the error cache"""

        if i_1 == i_2:
            return 0

        alpha_1 = self.alphas[i_1]
        alpha_2 = self.alphas[i_2]
        y_1 = self.y[i_1]
        y_2 = self.y[i_2]
        E_1 = self.errors[i_1]
        E_2 = self.errors[i_2]
        s = y_1 * y_2
        # Choosing the bound for new_alpha_2
        if y_1 != y_2:
            L = max(0, alpha_2 - alpha_1)
            H = min(self.C, self.C + alpha_2 - alpha_1)
        else:
            L = max(0, alpha_2 + alpha_1 - self.C)
            H = min(self.C, alpha_2 + alpha_1)
        if L == H:
            return 0

        k_11 = self.kernel(self.X[i_1], self.X[i_1])
        k_12 = self.kernel(self.X[i_1], self.X[i_2])
        k_22 = self.kernel(self.X[i_2], self.X[i_2])
        # second derivative of the objective function along the diagonal line
        second_derivative = k_11 + k_22 - 2 * k_12
        if second_derivative > 0:
            new_alpha_2 = alpha_2 + y_2 * (E_1 - E_2) / second_derivative
            if new_alpha_2 >= H:
                new_alpha_2 = H
            elif new_alpha_2 <= L:
                new_alpha_2 = L
        # Handling the case of non positive second derivative
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

        if np.abs(new_alpha_2 - alpha_2) < self.eps * (new_alpha_2 + alpha_2 + self.eps):
            return 0

        # Calculating the new_alpha_1
        new_alpha_1 = alpha_1 + s * (alpha_2 - new_alpha_2)

        # Updating the threshold
        b_1 = E_1 + y_1 * (new_alpha_1 - alpha_1) * k_11 + y_2 * (new_alpha_2 - alpha_2) * k_12 + self.b
        b_2 = E_2 + y_1 * (new_alpha_1 - alpha_1) * k_12 + y_2 * (new_alpha_2 - alpha_2) * k_22 + self.b
        if 0 < new_alpha_1 < self.C:
            new_b = b_1
        elif 0 < new_alpha_2 < self.C:
            new_b = b_2
        else:
            new_b = (b_1 + b_2) * 0.5

        self.b = new_b
        self.alphas[i_1] = new_alpha_1
        self.alphas[i_2] = new_alpha_2

        # updating w if the SVM is linear
        if self.kernel is linearKernel:
            new_w = self.w + y_1 * (new_alpha_1 - alpha_1) * self.X[i_1, :] + y_2 * (new_alpha_2 - alpha_2) * self.X[i_2, :]
            self.w = new_w

        # updating the error cache
        self.errors = self.evaluate(self.X) - self.y


        return 1

    def plot_decision_boundary(self, ax, dataset, kernel, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):

        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        plt.title("SVM with SMO, dataset: {}, kernel: {}".format(dataset, kernel))
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
        support_vectors = self.X[np.round(self.alphas, decimals=2) != 0.0]
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                   linewidth=0.5, facecolors='none', edgecolors='k', label='Support Vectors')
        plt.legend(loc='best', bbox_to_anchor=(0.75, 0.2),
                   ncol=1, fancybox=True, shadow=True)
        plt.show()


if __name__ == "__main__":

    """ 
    Configuration parameters
        
        - C: Regularization parameter --> More C is higher and more the method will prefer to avoid the violations
                                          and the margin will be more thin
                                          
        - tol: tolerance for respect of KKT conditions (scikit-learn has 0.01 as default)
        
        - eps: alpha tolerance
        
        - dataset: dataset on which apply the SMO (possibilities: "exercise_sheet", "moon", "circle") 
                                                    N.B. Last two datasets are from scikit-learn
                                                    
        - kernel : kernel to use for SMO (possibilities: "linear", "gaussian")
    """

    C = 100.
    tol = 0.01
    eps = 0.05
    dataset = "exercise_sheet"
    kernel = "linear"

    smo = SMO(dataset, C, kernel, tol, eps)

    if smo.mainRoutine():
        fig, ax = plt.subplots()
        smo.plot_decision_boundary(ax, dataset, kernel)
