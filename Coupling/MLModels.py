from .RFmodels import MyRFRegressor
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import numpy as np



class MLModel(BaseEstimator):
    """
    basic example with a simple 1NN algorithm, based on data as 1D feature vectors
    """
    def __init__(self):
        self.model = KNeighborsRegressor(n_neighbors = 1)

    def dataItemBDDFormating(self, dataItemBDD):
        
        ListInput, ListTarget = dataItemBDD.getSimulatedDataLists()
        X = np.array(ListInput)
        y = np.array(ListTarget)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        return X, y

    def inputFormating(self, dataItem):
        x = dataItem.data
        x = x.reshape(1, -1)
        return x

    def train(self, dataItemBDD):
        X, y = self.dataItemBDDFormating(dataItemBDD)
        self.model.fit(X, y)

    def predict(self, dataItem):
        x = self.inputFormating(dataItem)
        pred = self.model.predict(x)
        dataItem.addTargetML(pred[0])

    def evaluatePrediction(self, dataItem):
        x = self.inputFormating(dataItem)
        dist, _ = self.model.kneighbors(x)
        dataItem.addConfidenceML(dist[0])

class RFModel(MLModel):
    def __init__(self, n_tree, error = "disagreement"):
        super().__init__()
        self.model = MyRFRegressor(n_tree)
        self.error = error
    
    def evaluatePrediction(self, dataItem):
        x = self.inputFormating(dataItem)
        if self.error in ["disagreement", "sampling"]:
            _, unc = self.model.predict(x, return_error = self.error)

        elif self.error == "diversity":
            unc, _ = self.nn.kneighbors(x)

        elif self.error == "consistency":
            x_noise = x.squeeze() + np.random.normal(np.zeros(self.n_feat), np.ones((self.n_feat,)), size = (50, self.n_feat))*self.sigma
            pred_noise = self.model.predict(x_noise)
            unc = np.sum(np.std(pred_noise, axis = 0)**2)
        else:
            unc = np.random.random()
        dataItem.addConfidenceML(unc)

    def train(self, dataItemBDD):
        super().train(dataItemBDD)
        if self.error == "diversity":
            X, y = self.dataItemBDDFormating(dataItemBDD)
            self.nn = NearestNeighbors(n_neighbors=1).fit(X)
        if self.error == "consistency":
            X, y = self.dataItemBDDFormating(dataItemBDD)
            self.sigma = np.std(X, axis = 0)/10
            self.n_feat = X.shape[1]
        


    

    



