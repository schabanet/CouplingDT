import numpy as np 
from ..stream import StreamBase, StreamBDD
from ...Utils import DataItem
from skmultiflow.data import RegressionGenerator
from copy import deepcopy
from sklearn.utils import shuffle

class StreamAbt(StreamBase):
    def __init__(self, n, arrivalRate = 1, *ags, **kwargs):
        """
        stream avec drift abrupt réel
        """
        super().__init__(arrivalRate)
        self.arrivalNumber = 0
        self.drift_idx = int(n/2)
        self.X = []
        self.Y = []
        self.streamLength = n
        self.generator1 = RegressionGenerator(n_samples = self.drift_idx, n_features = 10)
        self.generator2 = RegressionGenerator(n_samples = n - self.drift_idx, n_features = 10)


    def generateDataItem(self):
        if self.arrivalNumber >= self.streamLength-1:
            self.generateTime = lambda : np.inf #pas très propre...
        
        if self.arrivalNumber < self.drift_idx:
            x, y = self.generator1.next_sample()
            x = x[0]
            y = y[0]

        else:
            x, y = self.generator2.next_sample()
            x = x[0]
            y = y[0]

        self.X.append(x)
        self.Y.append(y)
        self.arrivalNumber += 1

        return DataItem(self.arrivalNumber-1, data = x)

    def getOracle(self, idx):
        """
        assime idx isn't from the future
        """
        return self.Y[idx]
    
class StreamSortAbt(StreamBDD):
    def __init__(self, X, Y, arrivalRate = 1, *ags, **kwargs):
        """
        stream avec drift abrupt virtuel
        """
        super().__init__(X, Y, arrivalRate)
        self.arrivalNumber = 0
        n, p = X.shape
        self.drift_idx = int(n/2)
        ftsort = np.random.randint(p)
        trs = np.median(X[:,ftsort])
        X1 = X[X[:, ftsort] < trs]
        X2 = X[X[:, ftsort] >= trs]
        Y1 = Y[X[:, ftsort] < trs]
        Y2 = Y[X[:, ftsort] >= trs]
        self.X = np.concatenate([X1, X2], axis = 0)
        #self.X = np.delete(self.X, obj = ftsort, axis = 1)
        self.Y = np.concatenate([Y1, Y2], axis = 0)
        self.streamLength = n


class StreamSwapReal(StreamBDD):
    def __init__(self, X, Y, arrivalRate = 1, Shuffle = True, *ags, **kwargs):
        """
        stream avec drift abrupt virtuel
        """
        if Shuffle:
            X, Y = shuffle(X, Y)
        super().__init__(X, Y, arrivalRate)
        self.arrivalNumber = 0
        n, p = X.shape
        self.drift_idx = int(n/2)
        self.X = deepcopy(X)
        #self.X[:self.drift_idx] = self.X[:self.drift_idx][:, np.random.permutation(p)]
        self.Y = deepcopy(Y)

        self.Y[self.drift_idx:] = self.Y[self.drift_idx:][np.random.permutation(n - self.drift_idx)]
        self.streamLength = n
    
class StreamSwapAbt(StreamBDD):
    def __init__(self, X, Y, arrivalRate = 1, Shuffle = True, *ags, **kwargs):
        """
        stream avec drift abrupt virtuel
        """
        if Shuffle:
            X, Y = shuffle(X, Y)
        super().__init__(X, Y, arrivalRate)
        self.arrivalNumber = 0
        n, p = X.shape
        self.drift_idx = int(n/2)
        self.X = deepcopy(X)
        self.X[:self.drift_idx] = self.X[:self.drift_idx][:, np.random.permutation(p)]
        self.Y = deepcopy(Y)
        self.streamLength = n