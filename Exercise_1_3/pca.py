import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def pca(data, k):
    '''
    Computate the pca.
    @param data: numpy.array of data vectors (NxM)
    @param k: number of eigenvectors to return

    returns (eigenvectors (NxM), eigenvalues (N))
    '''
    mean_vector = [np.mean(x) for x in data.T]
    X = []
    for i in range(len(data)):
        X.append((data[i, :] - mean_vector[i]))
    X = np.array(X)
    cov = np.cov(X)
    eigenvalues, eigenvectors = np.linalg.eig(cov.T)
    return eigenvectors[:, :k], eigenvalues[:k]


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

def normalized_linear_combination(vecs, weights):
    '''
    Compute a linear combination of given vectors and weights.
    len(weights) must be <= vecs.shape[0]
    @param vecs: numpy.array of numpy.arrays to combine (NxM, N = number of basis vectors (unitary or eigenvectors))
    @param weights: list of weights (S) for the first S vectors
    returns numpy.array [M,1]
    '''

    #TODO: Your implementation goes here
    return np.array(np.dot(vecs, weights))

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
        files = glob.glob(path+"/*.pgm")
        for file in files:
            img = misc.imread(file)
            
            #scale down for faster computation
            img = misc.imresize(img, (50,50))
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


k = 3
# data, datashape = load_dataset_from_folder('data/')
# showVec(data[100], datashape)

# data = np.array(([3, 2, 1], [2, 3, 1], [1, 2, 3]), dtype='float')
data = np.array(([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1], [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]), dtype='float')

eigenvectors, eigenvalues = pca(data, k)
print(f"Eigenvectors: {eigenvectors}")
print(f"Eigenvalues: {eigenvalues}")

p = normalized_linear_combination(eigenvectors, eigenvalues)
print(f"Normalized linear compination: {p}")


