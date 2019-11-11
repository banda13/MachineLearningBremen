import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import cv2


# returns the mean for each column as a vector
def get_mean_vector(data):
    return [np.mean(x) for x in data.T]


# subtract the mean vector from each column
def subtract_mean_vector(data, mean_vector):
    X = []
    for i in range(len(data[0, :])):
        X.append((data[:, i] - mean_vector[i]))
    return np.array(X).T


# add the mean vector to each column
def add_mean_vector(data, mean_vector):
    X = []
    for i in range(len(data[0, :])):
        X.append((data[:, i] + mean_vector[i]))
    return np.array(X).T


def pca(data, k):
    '''
    Computate the pca.
    @param data: numpy.array of data vectors (NxM)
    @param k: number of eigenvectors to return

    returns (eigenvectors (NxM), eigenvalues (N))
    '''
    print('Executing pca with k == %d' % k)

    # subtract the mean vector from the data, calculate covariance matrix and get the eigen vectors and values
    X = subtract_mean_vector(data, get_mean_vector(data))
    cov = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # sort the eigenvalues and the vectors from the smallest value to largest
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # returns the first k sorted eigenvectors and values as a float (conversion needed the round the complex numbers)
    return sorted_eigenvectors[:, :k].real.astype('float'), sorted_eigenvalues[:k].real.astype('float')


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


# display two image side by side to compare
def show2Vec(img1, img2, shape1, shape2, title):
    img1 = np.reshape(img1, shape1)
    img2 = np.reshape(img2, shape2)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    f.suptitle(title, fontsize=16)
    plt.show()


def normalized_linear_combination(vecs, weights):
    '''
    Compute a linear combination of given vectors and weights.
    len(weights) must be <= vecs.shape[0]
    @param vecs: numpy.array of numpy.arrays to combine (NxM, N = number of basis vectors (unitary or eigenvectors))
    @param weights: list of weights (S) for the first S vectors
    returns numpy.array [M,1]
    '''
    return np.dot(vecs.T, weights.T)


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

if __name__ == '__main__':
    # number of components to keep
    k = 4

    # read the dataset and convert it to float
    data, datashape = load_dataset_from_folder('data/')
    data = np.array(data, dtype='float')

    eigenvectors, eigenvalues = pca(data, k)

    mean_vector = get_mean_vector(data)
    mean_adjusted_data = subtract_mean_vector(data, mean_vector)

    # multiply the transpose eigenvector matrix with the data without the mean to get or transformed data matrix
    transformed_data = np.array(np.dot(eigenvectors.T, mean_adjusted_data.T), dtype='float').T

    # get back the origin data as multiplying the eigenvector matrix with the transposed transformed data
    inverse_transformed_data = np.array(add_mean_vector(np.dot(eigenvectors, transformed_data.T), mean_vector),
                                        dtype='float').T

    # calculations for different k-s
    k = 100
    eigenvectors, eigenvalues = pca(data, k)
    transformed_data_100 = np.array(np.dot(eigenvectors.T, mean_adjusted_data.T), dtype='float').T
    inverse_transformed_data_100 = np.array(add_mean_vector(np.dot(eigenvectors, transformed_data_100.T), mean_vector),
                                            dtype='float').T

    k = 400
    eigenvectors, eigenvalues = pca(data, k)
    transformed_data_400 = np.array(np.dot(eigenvectors.T, mean_adjusted_data.T), dtype='float').T
    inverse_transformed_data_400 = np.array(add_mean_vector(np.dot(eigenvectors, transformed_data_400.T), mean_vector),
                                            dtype='float').T

    """ Exercise 1.3 """

    print('Showing the first 4 principal component')
    show2Vec(data[0], transformed_data[0], datashape, (2, 2), "First 4 principal component")

    print('Showing the first 100 principal component')
    show2Vec(data[0], transformed_data_100[0], datashape, (10, 10), "First 100 principal component")

    print('Showing the first 100 reconstructed picture from 4 principal component')
    fig, axes = plt.subplots(10, 10, figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        ax.imshow(inverse_transformed_data[i].reshape(50, 50), cmap="gray")
    fig.suptitle('First 100 reconstructed picture from 4 principal component', fontsize=16)
    plt.show()

    print('Showing the difference between the origin and the reconstructed data with k= 4, 100, 400')
    show2Vec(data[0], inverse_transformed_data[0], datashape, datashape, "Origin vs Reconstructed data (k == 4)")
    show2Vec(data[0], inverse_transformed_data_100[0], datashape, datashape, "Origin vs Reconstructed data (k == 100)")
    show2Vec(data[0], inverse_transformed_data_400[0], datashape, datashape, "Origin vs Reconstructed data (k == 400)")

    print('Showing the difference between different k-values: 4 vs 100, 100 vs 400')
    show2Vec(inverse_transformed_data[0], inverse_transformed_data_100[0], datashape, datashape, "Reconstructed datas (4 vs 100)")
    show2Vec(inverse_transformed_data_100[0], inverse_transformed_data_400[0], datashape, datashape, "Reconstructed datas (100 vs 400)")
