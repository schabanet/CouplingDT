from ..master import MasterBase
from sklearn.utils import shuffle
from ..Utils import Clock, DataItemBDD,  evaluateBDD
from .stream import StreamBDD
from ..Simulator import PseudoSimulator, PseudoSimulator1inN
from ..router import RouterRF
import numpy as np
from copy import deepcopy



class MasterBddRF(MasterBase):
    def __init__(self, X, Y, 
                    budget, 
                    n_tree = 100, 
                    arrivalRate = 1, 
                    simulationRate = 1, 
                    init = 500,
                    n_retrain = np.inf,
                    nSlots = np.inf,
                    strategy = "disagreement"):

        self.n_tree = n_tree
        self.budget = budget
        self.n_retrain = n_retrain
        self.simulationRate = simulationRate
        self.history = {}

        X, Y = shuffle(X, Y)
        
        self.stream = StreamBDD(X, Y, arrivalRate)
        self.clock = Clock()
        self.init = init
        
        #get the init first items to train the RF model
        bdd = DataItemBDD()
        for i in range(init):
            dataItem = self.stream.generateDataItem()
            dataItem.addTargetSimul(self.stream.Y[i])
            bdd.add(dataItem)

        if strategy == '1inN':
            simulator = PseudoSimulator1inN(nSlots = nSlots, rate = simulationRate, targets = self.stream.Y)
        else:
            simulator = PseudoSimulator(nSlots = nSlots, rate = simulationRate, targets = self.stream.Y)

        self.router = RouterRF(n_tree, simulator, bdd, 1-budget, n_retrain, strategy = strategy) 
        
        self.router.mlModel.train(bdd)


    def evaluate(self, Print = True):
        ev = evaluateBDD(self.router.dataItemBDD, self.stream.Y, self.init)
        if Print:
            for k, v in ev.items():
                print(k + " : ", v)
        return ev
    
    def stepSimulationEnd(self, finished):
        super().stepSimulationEnd(finished)
        self.router.n_simul += len(finished)
        if self.router.shouldRetrainModel():
            self.router.train()


