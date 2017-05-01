import cPickle as pickle
import numpy as np
import os


def load_cifar_batch(filename):
    with open(filename, "r") as f:
        datadict = pickle.load(f)
        X = datadict["data"]
        Y = datadict["labels"]
        # resphape creates a new strucutre for the tensor using the given tuple
        X = X.reshape(10000 , 3 , 32 , 32) # 10000 images of 3 channels of width and height 32 , 32 ?
        # transpose interchanges the axis as per the typle passed in
        X = X.transpose(0 , 2 , 3 , 1).astype("float")
        Y = np.array(Y)
        return X , Y 


def load_cifar10(dir_):
    xs = []
    ys = []
    for b in range(1,6): # CIFAR has 6 , 10000 image batches ? 
        f = os.path.join(dir_ , "data_batch_%d" %(b,))
        X , Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X , Y 
    Xte , Yte = load_cifar_batch(os.path.join(dir_ , "test_batch"))
    return Xtr , Ytr, Xte , Yte



