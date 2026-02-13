from pathlib import Path

import numpy as np
import pandas as pd
from clustpy.data import load_multiple_features, load_webkb
from clustpy.data.real_medical_mnist_data import load_blood_mnist
from clustpy.data.real_torchvision_data import (
    load_fmnist,
    load_mnist,
    load_usps,
)
from clustpy.data.real_uci_data import (
    load_har,
    load_optdigits,
    load_pendigits,
    load_reuters21578,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def get_data_path():
    # create a results dir
    data_dir = Path(__file__).parent / "data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


def standardize_and_split(X, y, test_train_split=False, normalize=True, normalizing_const=None):
    if normalizing_const is None:
        # use z-scaling
        if test_train_split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            return X_train, X_test, y_train, y_test
        else:
            if normalize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            return X, y
    else:
        # normalize by dividing by a constant
        if normalize:
            X = X / normalizing_const

        if test_train_split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            return X, y


def get_optdigits(normalize=True, test_train_split=False):
    # clusters = 10, d = 64
    data_dir = str(get_data_path())
    X, y = load_optdigits(return_X_y=True, downloads_path=data_dir)    
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=None)


def get_har(normalize=True, test_train_split=False):
    # clusters = 6, d = 561
    data_dir = str(get_data_path())
    X, y = load_har(return_X_y=True, downloads_path=data_dir)
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=None)


def get_pendigits(normalize=True, test_train_split=False):
    # clusters = 10, d = 16
    data_dir = str(get_data_path())
    X, y = load_pendigits(return_X_y=True, downloads_path=data_dir)
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=None)


def get_mnist(normalize=True, test_train_split=False):
    # clusters = 10, d = 784
    data = load_mnist("all")
    X = data["data"]
    y = data["target"]
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=255.0)


def get_usps(normalize=True, test_train_split=False):
    # clusters = 10, d = 256
    data_dir = str(get_data_path())
    X, y = load_usps(return_X_y=True, downloads_path=data_dir)
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=255.0)


def get_fashion_mnist(normalize=True, test_train_split=False):
    # clusters = 10, d = 784
    data = load_fmnist("train")
    X = data["data"]
    y = data["target"]
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=255.0)


def get_handwritten(normalize=True, test_train_split=False):
    # 'handwritten' dataset from A3S
    # clusters = 10, d = 76
    
    data_dir = str(get_data_path())
    X, y = load_multiple_features(return_X_y=True, downloads_path=data_dir)
    X = X[:, 216:216+76]
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=None)


def get_blood_mnist(normalize=True, test_train_split=False):
    # clusters = 8, d = 2352
    data_dir = str(get_data_path())
    X, y = load_blood_mnist(return_X_y=True, downloads_path=data_dir)
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=255.0)


def get_reuters(normalize=True, test_train_split=False):
    # clusters = 5, d = 2000
    X, y = load_reuters21578(return_X_y=True)
    return standardize_and_split(X, y, test_train_split, normalize=False, normalizing_const=None)


def get_waveform(normalize=True, test_train_split=False):
    # clusters = 3, d = 21
    waveform = fetch_ucirepo(id=107) 
    # data (as pandas dataframes) 
    X = waveform.data.features 
    y = waveform.data.targets
    X = X.to_numpy()
    y = y.to_numpy().flatten()
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=None)


def get_webkb(normalize=True, test_train_split=False):
    X, y = load_webkb(use_tfidf=True, use_universities=None, use_categories=("student", "staff", "project", "faculty", "department", "course"), min_df=1, max_features=2000, min_variance=0, return_X_y=True)
    y = y[:, 0]
    return standardize_and_split(X, y, test_train_split, normalize=False, normalizing_const=None)


def get_segmentation(normalize=True, test_train_split=False):
    data_dir = get_data_path() / "image+segmentation"
    df = pd.concat([pd.read_csv(data_dir / "segmentation.data.txt"), pd.read_csv(data_dir / "segmentation.test.txt")])
    X = df.to_numpy()
    y = df.index.to_numpy()
    labels = np.zeros(len(y), dtype=int)
    labels[y == "SKY"] = 1
    labels[y == "FOLIAGE"] = 2
    labels[y == "CEMENT"] = 3
    labels[y == "WINDOW"] = 4
    labels[y == "PATH"] = 5
    labels[y == "GRASS"] = 6
    y = labels
    return standardize_and_split(X, y, test_train_split, normalize, normalizing_const=None)


if __name__ == "__main__":

    # load and save datasets 
    _, _ = get_optdigits()
    _, _ = get_waveform()
    _, _ = get_handwritten()
    _, _ = get_blood_mnist()
    _, _ = get_mnist()
    _, _ = get_fashion_mnist()
    _, _ = get_pendigits()
    _, _ = get_usps()
    _, _ = get_reuters()
    _, _ = get_webkb()
    _, _ = get_segmentation()
