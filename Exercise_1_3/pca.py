import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def emp_mean(data):
    mean_vector = [np.mean(x) for x in data]
    X = []
    for i in range(len(data)):
        X.append((data[i, :] - mean_vector[i]))
    return np.array(X)


def pca(data, k):
    '''
    Computate the pca.
    @param data: numpy.array of data vectors (NxM)
    @param k: number of eigenvectors to return

    returns (eigenvectors (NxM), eigenvalues (N))
    '''
    X = emp_mean(data)
    cov = np.cov(X)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenValues = eigenvalues[idx]
    eigenVectors = eigenvectors[:, idx]
    return eigenVectors[:, :k], eigenValues[:k]


def showVec(img, shape):
    '''
    Reshape vector to given image size and plots it
    len(img) must be shape[0]*shape[1]
    @param img: given image as 1d vector
    @param shape: shape of image
    '''
    img = np.reshape(img, shape)
    plt.imshow(img, cmap="gray")
    plt.show()


def show2Vec(img1, img2, shape1, shape2):
    img1 = np.reshape(img1, shape1)
    img2 = np.reshape(img2, shape2)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    plt.show()


def normalized_linear_combination(vecs, weights):
    '''
    Compute a linear combination of given vectors and weights.
    len(weights) must be <= vecs.shape[0]
    @param vecs: numpy.array of numpy.arrays to combine (NxM, N = number of basis vectors (unitary or eigenvectors))
    @param weights: list of weights (S) for the first S vectors
    returns numpy.array [M,1]
    '''
    return np.dot(vecs.T, weights)


def load_dataset_from_folder(dir):
    '''
    Load all pgm files from given folder
    Dataset is taken from https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    @param dir: folder to look for pgm files
    returns tuple of data (N,M) and shape of one img, N = number of samples, M = size of sample
    '''
    datalist = []
    datashape = []
    for path, _, files in os.walk(dir):
        files = glob.glob(path + "/*.pgm")
        for file in files:
            img = misc.imread(file)

            # scale down for faster computation
            img = misc.imresize(img, (50, 50))
            datashape = img.shape

            d = np.ravel(img)
            datalist.append(d)

    data = np.array(datalist)
    return data, datashape


'''
1) Load dataset
2) compute principal components
3) compute linear combinations
4) display stuff
'''

k = 1000
data, datashape = load_dataset_from_folder('data/')

# data = np.array(([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1], [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]), dtype='float')

eigenvectors, eigenvalues = pca(data, k)
# print(f"Eigenvectors: {eigenvectors}")
# print(f"Eigenvalues: {eigenvalues}")

transformed_data = []
mean_adjusted_data = emp_mean(data)
for x in mean_adjusted_data.T:
    transformed_data.append(normalized_linear_combination(eigenvectors, x))
transformed_data = np.array(transformed_data, dtype='float').T

show2Vec(data[0], transformed_data[0], datashape, datashape)
show2Vec(data[6], transformed_data[6], datashape, datashape)
