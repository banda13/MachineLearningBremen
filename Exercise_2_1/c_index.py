import numpy as np
import matplotlib.pyplot as plt

from Exercise_2_1.kmeans import KMeans


def calculate_C_index(X, clusters):
    P = [(X[i], X[j]) for i in range(len(X)) for j in range(i + 1, len(X))]
    Q = [(X[i], X[j]) for i in range(len(X)) for j in range(i + 1, len(X)) if clusters[i] == clusters[j]]
    Scl = sum([np.linalg.norm(np.array(i[0]) - np.array(i[1])) for i in Q])
    q = len(Q)
    P.sort(key=lambda x: np.linalg.norm(np.array(x[0]) - np.array(x[1])))
    Smin = sum([np.linalg.norm(np.array(i[0]) - np.array(i[1])) for i in P[:q]])
    Smax = sum([np.linalg.norm(np.array(i[0]) - np.array(i[1])) for i in P[(len(P) - q):]])
    return (Scl - Smin) / (Smax - Smin)


def plot_results(min_c_values, avg_c_values, k):
    plt.xlim(2, 9)
    plt.xlabel('k')
    plt.plot(k, min_c_values, 'r', label='Min c')
    plt.plot(k, avg_c_values, 'b', label='Avg c')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X = np.genfromtxt('cluster_dataset2d.txt', delimiter=',')
    min_c_values = []
    avg_c_values = []
    k_range = range(2, 10)

    for k in k_range:
        c_idxs = np.array([])
        for i in range(50):
            while True:
                try:
                    c = KMeans(k)
                    c.plotting = False
                    c.fit(X)
                    clusters = c.predict(X)
                    c_idxs = np.append(c_idxs, calculate_C_index(X, clusters))
                    break
                except Exception as e:
                    pass  # print(e)

        print('k: {} -> min: {}, avg: {}'.format(k, np.min(c_idxs), c_idxs.mean()))
        min_c_values.append(np.min(c_idxs))
        avg_c_values.append(c_idxs.mean())
    plot_results(min_c_values, avg_c_values, k_range)
