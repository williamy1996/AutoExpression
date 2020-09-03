import numpy as np
import os

from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file


def gisette2npz():
    data_path = "./dataset/GISETTE"
    train_data = np.loadtxt(os.path.join(data_path, "GISETTE/gisette_train.data"))
    train_labels = np.loadtxt(os.path.join(data_path, "GISETTE/gisette_train.labels"))
    valid_data = np.loadtxt(os.path.join(data_path, "GISETTE/gisette_valid.data"))
    valid_labels = np.loadtxt(os.path.join(data_path, "gisette_valid.labels"))

    x = np.vstack((train_data, valid_data))
    y = np.hstack((train_labels, valid_labels))

    np.savez(os.path.join(data_path, "gisette.npz"), x=x, y=y)


def madelon2npz():
    data_path = "./dataset/MADELON"
    train_data = np.loadtxt(os.path.join(data_path, "MADELON/madelon_train.data"))
    train_labels = np.loadtxt(os.path.join(data_path, "MADELON/madelon_train.labels"))
    valid_data = np.loadtxt(os.path.join(data_path, "MADELON/madelon_valid.data"))
    valid_labels = np.loadtxt(os.path.join(data_path, "madelon_valid.labels"))

    x = np.vstack((train_data, valid_data))
    y = np.hstack((train_labels, valid_labels))

    np.savez(os.path.join(data_path, "madelon.npz"), x=x, y=y)


def load_basehock():
    data_path = "../dataset/BASEHOCK.mat"
    basehock = loadmat(data_path)

    return basehock['X'], basehock['Y'].reshape(-1)


def load_madelon():
    data_path = "../dataset/MADELON"
    madelon = np.load(os.path.join(data_path, "madelon.npz"))
    x, y = madelon['x'], madelon['y']

    return x, y


def load_gisette():
    data_path = "../dataset/GISETTE"
    gisette = np.load(os.path.join(data_path, "gisette.npz"))
    x, y = gisette['x'], gisette['y']
    return x, y


def load_coil20():
    data_path = "../dataset/COIL20.mat"
    coil20 = loadmat(data_path)

    return coil20['X'], coil20['Y'].reshape(-1)


def load_usps():
    data_path = "../dataset/USPS.mat"
    usps = loadmat(data_path)

    return usps['X'], usps['Y'].reshape(-1)


def load_news20():
    data_path = "../dataset/news20.binary"
    news20 = load_svmlight_file(data_path)
    x, y = news20[0].toarray(), news20[1].astype(int)
    return x, y


def load_dataset(dataset):
    x = None
    y = None

    if dataset == "madelon":
        x, y = load_madelon()
    if dataset == 'gisette':
        x, y = load_gisette()
    if dataset == "basehock":
        x, y = load_basehock()
    if dataset == "coil20":
        x, y = load_coil20()
    if dataset == "usps":
        x, y = load_usps()
    if dataset == "news20":
        x, y = load_news20()

    if x is None and y is None:
        FileNotFoundError("Not such dataset")

    return x, y


def generate_synthetic_data(n=100, datatype=""):
    """
        Generate data (X,y)
        Args:
            n(int): number of samples
            datatype(string): The type of data
            choices: 'orange_skin', 'XOR', 'regression'.
        Return:
            X(float): [n,d].
            y(float): n dimensional array.
        """
    if datatype == 'orange_skin':
        X = []

        i = 0
        while i < n // 2:
            x = np.random.randn(10)
            if 9 < sum(x[:4] ** 2) < 16:
                X.append(x)
                i += 1
        X = np.array(X)

        X = np.concatenate((X, np.random.randn(n // 2, 10)))

        y = np.concatenate((-np.ones(n // 2), np.ones(n // 2)))

        perm_inds = np.random.permutation(n)
        X, y = X[perm_inds], y[perm_inds]

    elif datatype == 'XOR':
        X = np.random.randn(n, 10)
        y = np.zeros(n)
        splits = np.linspace(0, n, num=8 + 1, dtype=int)
        signals = [[1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [1, -1, 1]]
        for i in range(8):
            X[splits[i]:splits[i + 1], :3] += np.array([signals[i]])
            y[splits[i]:splits[i + 1]] = i // 2

        perm_inds = np.random.permutation(n)
        X, y = X[perm_inds], y[perm_inds]

    elif datatype == 'regression':
        X = np.random.randn(n, 10)

        y = -2 * np.sin(2 * X[:, 0]) + np.maximum(X[:, 1], 0) + X[:, 2] + np.exp(-X[:, 3]) + np.random.randn(n)

    elif datatype == 'regression_approx':
        X = np.random.randn(n, 10)

        y = -2 * np.sin(2 * X[:, 0]) + np.maximum(X[:, 1], 0) + X[:, 2] + np.exp(-X[:, 3]) + np.random.randn(n)

    else:
        raise AttributeError("not such datatype")

    return X, y


if __name__ == '__main__':
    x, y = load_gisette()
    print(x.shape, y.shape)
    print(x.dtype, y.dtype)
