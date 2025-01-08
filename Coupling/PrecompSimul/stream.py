from ..stream import StreamBase
from ..Utils import DataItem
import numpy as np
from sklearn.utils import shuffle

class StreamBDD(StreamBase):
    def __init__(self, X, Y, arrivalRate = 1, init = 0, Shuffle = True):
        super().__init__(arrivalRate)
        self.arrivalNumber = init
        if Shuffle:
            X, Y = shuffle(X, Y)
        self.X = X
        self.Y = Y
        self.streamLength = len(X)

    def generateDataItem(self):
        if self.arrivalNumber >= self.streamLength-1:
            self.generateTime = lambda : np.inf #pas très propre...

        self.arrivalNumber += 1
        return DataItem(self.arrivalNumber-1, data = self.X[self.arrivalNumber-1])
        

    def getOracle(self, idx):
        return self.Y[idx]