"""SARCOS Inverse Dynamics Problem.

The SARCOS dataset is taken from 'http://www.gaussianprocess.org/gpml/data'.
It is an inverse dynamics problem, i.e. we have to predict the 7 joint torques
given the joint positions, velocities and accelerations. Hence, we have to
solve a regression problem with 21 inputs and 7 outputs and a nonlinear
relation.

Results will be compared based on the normalized mean squared error (nMSE) for
each output dimension on the test set. The nMSE is the mean squared error
divided by the variance of the corresponding output dimension.
"""

import os
import sys
try:
    # Python 2
    from urllib import urlopen
except:
    # Python 3
    from urllib.request import urlopen
import numpy as np
try:
    import scipy.io
except:
    print("SciPy is required for this benchmark.")
    exit(1)


FILES = ["sarcos_inv.mat", "sarcos_inv_test.mat"]
URLS = ["http://www.gaussianprocess.org/gpml/data/%s" % f for f in FILES]


def download_sarcos():
    """Download Sarcos dataset."""
    if all(os.path.exists(f) for f in FILES):
        print("Download is not required.")
        return

    for i in range(len(URLS)):
        print("Downloading %s" % URLS[i])
        downloader = urlopen(URLS[i])

        with open(FILES[i], "wb") as out:
            while True:
                data = downloader.read(1024)
                if len(data) == 0: break
                out.write(data)
        print("Done.")


def load_sarcos(dataset="train"):
    """Load Sarcos dataset.

    Parameters
    ----------
    dataset : string
        Either 'train' or 'test'

    Returns
    -------
    X : array, shape (n_samples, 21)
        Samples (positions, velocities and accelerations of 7 joints)

    Y : array, shape (n_samples, 7)
        Targets (torques of 7 joints)
    """
    if dataset not in ["train", "test"]:
        raise ValueError("'dataset' must be either 'train' or 'test'")

    if dataset == "train":
        filename = "sarcos_inv"
    elif dataset == "test":
        filename = "sarcos_inv_test"

    data = scipy.io.loadmat(filename + ".mat")
    X = data[filename][:, :21]
    Y = data[filename][:, 21:]

    return X, Y


def nMSE(y, y_pred):
    """Compute the normalized mean squared error of predicted values.

    Parameters
    ----------
    y : array, shape (n_samples,)
        True values

    y_pred : array, shape (n_samples,)
        Predicted values
    """
    if y.shape != y_pred.shape:
        raise ValueError("Shapes don't match")

    nMSE = np.mean((y - y_pred) ** 2)
    return nMSE

