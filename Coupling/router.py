from .Utils import Distribution
from .MLModels import RFModel
import numpy as np

class RouterBase:
    def __init__(self, mlModel, simulatorMaster, dataItemBDD):
        self.mlModel = mlModel
        self.simulatorMaster = simulatorMaster
        self.dataItemBDD = dataItemBDD
        

    def shouldSimulate(confidence):
        return False

    def train(self):
        self.mlModel.train(self.dataItemBDD)

    def shouldRetrainModel(self):
        return False

    def completeDataItem(self, dataItem):
        self.mlModel.predict(dataItem)
        self.mlModel.evaluatePrediction(dataItem)
        to_simul = self.shouldSimulate(dataItem.confidenceML)
        if to_simul:
            to_simul = self.simulatorMaster.lauchSimulation(dataItem)
        return to_simul


class RouterRF(RouterBase):
    """"
    ATTENTION Telle qu'implémentée, elle ne fonctionne que pour le cas sans limite sur nSlots (infini)
    """
    def __init__(self, n_tree, simulator, dataItemBDD, error_tresh, n_retrain = np.inf, strategy = "disagreement"):
        
        model = RFModel(n_tree, error = strategy)
        super().__init__(model, simulator, dataItemBDD)
        self.distrib = Distribution(1, 100)
        self.error_tresh = error_tresh
        self.n_retrain = n_retrain
        self.n_simul = 0
        self.strategy = strategy
    
    def shouldSimulate(self, confidence):
        if self.strategy == "max":
            return len(self.simulatorMaster.counters) < self.simulatorMaster.nSlots
        elif self.strategy == "random":
            return np.random.random() > self.error_tresh
        else:
            if self.distrib.n < 10:
                return True
            rep_m = self.distrib.compute_ranks([confidence]) 
            min_m = np.min(rep_m)
            min_m = min_m/self.distrib.n
            return min_m > self.error_tresh

    def completeDataItem(self, dataItem):
        to_simul = super().completeDataItem(dataItem)
        self.distrib.update([dataItem.confidenceML])
        return to_simul
    
    def train(self):
        super().train()
        self.n_simul = 0
    
    def shouldRetrainModel(self):
        return (self.n_simul >= self.n_retrain)







