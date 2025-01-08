import numpy as np 
from ...router import RouterRF
from ...Drift.driftDetector import IncKSDriftDetector, PHDriftDetector, ControlChartDriftDetector, IncADDriftDetector

class RouterRFDrift(RouterRF):
    def __init__(self, n_tree, simulator, dataItemBDD, n_retrain = np.inf, strategy = "disagreement", driftDetection = "AD", driftDetectionParams = {}):
        super().__init__(n_tree, simulator, dataItemBDD, None, n_retrain = n_retrain, strategy = strategy)
        if driftDetection == "PH":
            self.driftDetector = PHDriftDetector(**driftDetectionParams)
        elif driftDetection == "CC":
            self.driftDetector = ControlChartDriftDetector(**driftDetectionParams)
        elif driftDetection == "KS":
            self.driftDetector = IncKSDriftDetector(**driftDetectionParams)
        else:
            self.driftDetector = IncADDriftDetector(**driftDetectionParams) #faire un vrai else

    def shouldSimulate(self, confidence):
        if self.strategy == "max":
            return len(self.simulatorMaster.counters) < self.simulatorMaster.nSlots
        elif self.strategy == "random":
            return np.random.random() > self.simulatorMaster.nRunningSimulation()/self.simulatorMaster.nSlots
        else:
            if self.distrib.n < 10:
                return True
            rep_m = self.distrib.compute_ranks([confidence]) 
            min_m = np.min(rep_m)
            min_m = min_m/self.distrib.n
            return min_m > self.simulatorMaster.nRunningSimulation()/self.simulatorMaster.nSlots

