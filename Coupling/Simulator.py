from scipy.stats import expon
import numpy as np

class PseudoSimulator:
    def __init__(self, targets = None, nSlots = np.inf, rate = 1):
        self.targets = targets
        self.nSlots = nSlots
        self.counters = []
        self.itemList = []
        self.distribution = lambda x :expon.rvs(scale = 1/rate)
    

    def sampleSimulationTime(self):
        return self.distribution(1)

    def lauchSimulation(self, dataItem):
        if len(self.counters) >= self.nSlots:
            return False
        simulTime = self.sampleSimulationTime()
        self.counters.append(simulTime)
        self.itemList.append(dataItem)
        return True

    def updateSimulationsRemainingTime(self, t):
        self.counters = [ct-t for ct in self.counters]
        nSlots = len(self.counters)
        finished = []
        for i in range(nSlots):
            if round(self.counters[nSlots - i -1], 6) <= 0:
                finished.append(self.itemList[nSlots - i -1])
                del self.counters[nSlots - i -1]
                del self.itemList[nSlots - i -1]
        return finished
    

    def nRunningSimulation(self):
        return len(self.counters)

    def predict(self, dataItem):
        dataItem.targetSimul = self.targets[dataItem.id]

class PseudoSimulator1inN(PseudoSimulator):
    def __init__(self, nSlots = 1, rate = 1/5, targets = None,):
        super().__init__(nSlots = nSlots, rate = rate, targets = targets)
        self.distribution = lambda x : 1/rate
    def updateSimulationsRemainingTime(self, *args, **kwargs):
        finished = super().updateSimulationsRemainingTime(1)
        return finished

