import numpy as np 
from ...Drift.master import MasterDriftBase
from .stream import StreamSortAbt, StreamSwapAbt
from ...Simulator import SimulatorMasterPrecomp, PseudoSimulator, PseudoSimulator1inN
from .router import RouterRFDrift
from ...Utils import Clock, DataItemBDD
import matplotlib.pyplot as plt


class MasterDriftFeat(MasterDriftBase):
    def __init__(self,X, Y, 
                budget, n_tree = 100, 
                arrivalRate = 1, 
                simulationRate = 1, 
                init = 500, 
                n_retrain = np.inf, 
                nSlots = np.inf, 
                strategy = "disagreement", 
                driftType = "sort", 
                driftDetection = "AD",
                driftDetectionParams = {}):

        if driftType == "sort":
            self.stream = StreamSortAbt(X, Y, arrivalRate = arrivalRate)
        else:
            self.stream = StreamSwapAbt(X, Y)

        self.clock = Clock()
        self.init = init
        self.drift_idx = self.stream.drift_idx
        
        #get the init first items to train the RF model
        bdd = DataItemBDD()
        for i in range(init):
            dataItem = self.stream.generateDataItem()
            dataItem.addTargetSimul(self.stream.getOracle(i))
            bdd.add(dataItem)


        if strategy == '1inN':
            simulator = SimulatorMasterPrecomp(PseudoSimulator1inN(nSlots = nSlots, rate = simulationRate), Y)
            
        else:
            simulator = SimulatorMasterPrecomp(PseudoSimulator(nSlots = nSlots, rate = simulationRate), Y)

        self.router = RouterRFDrift(n_tree, simulator, bdd, 1-budget, n_retrain, strategy = strategy, driftDetection = driftDetection, driftDetectionParams = driftDetectionParams) 

        self.router.mlModel.train(bdd)
        self.history = []

