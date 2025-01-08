import numpy as np 
from .Utils import Clock, DataItemBDD,  evaluateBDD
from .MLModels import MLModel
from .router import RouterBase
from .stream import StreamBase
from .Simulator import PseudoSimulator




class MasterBase:
    def __init__(self):
        self.clock = Clock()
        self.stream = StreamBase()
        bdd = DataItemBDD()
        mlModel = MLModel()
        simulatorMaster = PseudoSimulator()
        self.router = RouterBase(mlModel, simulatorMaster, bdd)
        self.history = {}


    def stepArrival(self, Time):
        dataItem = self.stream.generateDataItem()

        t = self.stream.generateTime()
        self.clock.addEvent(Time + t, "Arrival")
        is_simulated = self.router.completeDataItem(dataItem)
        self.router.dataItemBDD.add(dataItem)
        self.history[dataItem.id] = [is_simulated, Time, None]
        if is_simulated:
            simulTime = self.router.simulatorMaster.counters[-1]
            self.clock.addEvent(Time + simulTime, "SimulEnd")
        
        
    def stepSimulationEnd(self, finished):
        nextTime, _ = self.clock.nextEvent()
        for dataitem in finished:
            self.router.simulatorMaster.predict(dataitem)
            self.router.dataItemBDD.add(dataitem)
            self.history[dataitem.id][2] = nextTime


    def run(self):
        t = self.stream.generateTime()
        self.clock.addEvent(t, "Arrival")
        while len(self.clock.events) > 0:
            nextTime, nextEvent = self.clock.nextEvent()
            finished = self.router.simulatorMaster.updateSimulationsRemainingTime(nextTime - self.clock.time)
            if nextEvent == "Arrival":
                self.stepArrival(nextTime)
            elif nextEvent == "SimulEnd":
                self.stepSimulationEnd(finished)
            self.clock.update()
        return self

    def evaluate(self):
        print("not Implemented")




    