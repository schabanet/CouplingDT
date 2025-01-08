import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os



def loadApplience(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "energydata_complete.csv"))
    df = df.values
    X = df[:,2:]
    X = X.astype(np.float32)

    Y = df[:,1]
    Y = Y.astype(np.float32)
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

def loadCCPP(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "CCPP/Folds5x2_pp.csv"), sep = ";", decimal = ",")
    df = df.values
    X = df[:,:4]
    X = X.astype(np.float32)

    Y = df[:,4]
    Y = Y.astype(np.float32)
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

def loadConcrete(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "Concrete_Data.csv"), sep = ";")
    df = df.values
    X = df[:,:8]
    X = X.astype(np.float32)

    Y = df[:,8]
    Y = Y.astype(np.float32)
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

def loadGaz(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "gt_2011.csv"), )
    df = df.values
    X = df[:,:9]
    X = X.astype(np.float32)

    Y = df[:,-2:]
    Y = Y.astype(np.float32)
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

def loadGridElec(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "Elec_grid.csv"))
    df = df.values
    X = df[:,:12]
    X = X.astype(np.float32)

    Y = df[:,12]
    Y = Y.astype(np.float32)
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

def loadMusic(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "default_features_1059_tracks.txt"), sep = ",")
    df = df.values
    X = df[:,:8]
    X = X.astype(np.float32)

    Y = df[:,8]
    Y = Y.astype(np.float32)
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

def loadSuperConductor(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "train.csv"), sep = ",")
    df = df.values
    X = df[:,:81]
    X = X.astype(np.float32)

    Y = df[:,81]
    Y = Y.astype(np.float32)
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

def loadWine(data_path, Shuffle = True):
    df = pd.read_csv(os.path.join(data_path, "winequality-white.csv"), sep=";")
    df = df.values
    X = df[:,:11]
    Y = df[:,11]
    if Shuffle:
        X, Y = shuffle(X, Y)
    return X, Y

