import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.preprocessing import StandardScaler


def pca(data, k):
    '''
    Computate the pca.
    @param data: numpy.array of data vectors (NxM)
    @param k: number of eigenvectors to return

    returns (eigenvectors (NxM), eigenvalues (N))
    '''
    n = data.shape[1]
    m = data.shape[0]

    mean_vector = [np.mean(x) for x in data]
    X = []
    for i in range(len(data)):
        X.append((data[i, :] - mean_vector[i]))
    X = np.array(X)

    cov = (1/(n-1)) * np.matmul(X.T, X)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
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


#TODO: Your implementation goes here
k = 3
# data, datashape = load_dataset_from_folder('data/')

data = np.array(([3, 2], [2, 3]), dtype='float')

eigenvectors, eigenvalues = pca(data, k)
print(f"Eigenvectors: {eigenvectors}")
print(f"Eigenvalues: {eigenvalues}")

