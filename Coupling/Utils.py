import numpy as np
from scipy.stats import expon
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

class DataItem:
    def __init__(self, id, data = None, targetML = None, confidenceML = None, targetSimul = None):
        self.id = id
        self.data = data
        self.targetML = targetML
        self.confidenceML = confidenceML
        self.targetSimul = targetSimul

    def addTargetML(self, target):
        self.targetML = target
    
    def addConfidenceML(self, confidence):
        self.confidenceML = confidence
    
    def addTargetSimul(self, target):
        self.targetSimul = target


class DataItemBDD:
    def __init__(self):
        self.dico = {}

    def add(self, dataItem):
        self.dico[dataItem.id] = {"data" : dataItem.data,
                                    "targetML" : dataItem.targetML,
                                    "confidenceML" : dataItem.confidenceML,
                                    "targetSimul" : dataItem.targetSimul}

    def getSimulatedDataLists(self):
        dataList = []
        targetList = []
        for v in self.dico.values():
            if v["targetSimul"] is not None:
                dataList.append(v["data"])
                targetList.append(v["targetSimul"])
        return dataList, targetList




class Distribution:
    def __init__(self, n_m, w):
        self.n_m = n_m
        self.w = w
        self.samples = [[] for i in range(n_m)]
        self.sorted_samples = [[] for i in range(n_m)]
        self.ranks = [[] for i in range(n_m)]
        self.n = 0
    
    def compute_ranks_insert(self, ms):
        """
        retourne le nombres d'éléments dans chaque sample plus petit que ms
        """
        ranks = []
        if len(self.samples[0]) == 0:
            ranks = [0]*self.n_m #cas si les distributions sont vides
        else:
            for i, m in enumerate(ms):
                for j, s in enumerate(self.sorted_samples[i]):
                    if m < s:
                        ranks.append(j) #le rang du premier element de s plus grand que m
                        break
                    if j == len(self.sorted_samples[i]) - 1:
                        # itéré sur toute la liste sans sortir
                        #m est donc superieur a tous
                        ranks.append(len(self.sorted_samples[i])) 
        return ranks


    def compute_ranks(self, ms):
        """
        retourne le nombres d'éléments dans chaque sample plus petit que ms
        """
        ranks = []
        if len(self.samples[0]) == 0:
            ranks = [0]*self.n_m #cas si les distributions sont vides
        else:
            for i, m in enumerate(ms):
                ml = 0
                for j, s in enumerate(self.sorted_samples[i]):
                    if m > s:
                        ml  = j + 1
                    if m < s:
                        ranks.append(0.5*(j + ml)) #le rang du premier element de s plus grand que m
                        break
                    if j == len(self.sorted_samples[i]) - 1:
                        # itéré sur toute la liste sans sortir
                        #m est donc superieur a tous
                        ranks.append(len(self.sorted_samples[i])) 
        return ranks
                
    
    def update(self, ms):
        #first, drop the oldest sample if necessary
        if len(self.samples[0]) == self.w:
            for i, l in enumerate(self.samples):
                del l[0]
                rk = self.ranks[i][0]
                self.ranks[i] = [k-1 if k > rk else k for k in self.ranks[i]]
                del self.ranks[i][0]
                del self.sorted_samples[i][rk]
            self.n -= 1
        #calcule le rang dans la liste updaté
        ranks = self.compute_ranks_insert(ms)
        
        #met les listes a jour
        for i in range(self.n_m):
            self.samples[i].append(ms[i])
            self.ranks[i] = [k+1 if k >= ranks[i] else k for k in self.ranks[i]]
            self.ranks[i].append(ranks[i])
            if ranks[i] == 0:
                self.sorted_samples[i] = [ms[i]] + self.sorted_samples[i]
            elif ranks[i] == len(self.sorted_samples[i]):
                self.sorted_samples[i] = self.sorted_samples[i] + [ms[i]]
            else:
                self.sorted_samples[i] = self.sorted_samples[i][:ranks[i]] + [ms[i]] + self.sorted_samples[i][ranks[i]:]
        
        self.n += 1

class Clock:
    def __init__(self):
        self.time = 0
        self.events = {}

    def addEvent(self, t, event):
        if not np.isinf(t):
            self.events[t] = event

    def update(self):
        t = min(self.events.keys())
        self.time = t
        del self.events[t]

    def nextEvent(self):
        nextTime = min(self.events.keys())
        nextEvent = self.events[nextTime]
        return nextTime, nextEvent


def evaluateBDD(bdd, Y, init = 0):
    MLPreds = np.zeros_like(Y[init:])
    CouplePreds = np.zeros_like(Y[init:])
    notSimul = np.ones(len(Y), dtype = np.bool8)
    notSimul[:init] = False
    n_simul = 0
    for i in range(init, len(Y)):
        MLPreds[i-init] = bdd.dico[i]["targetML"]
        CouplePreds[i-init] = bdd.dico[i]["targetML"]
        if bdd.dico[i]["targetSimul"] is not None:
            n_simul += 1
            CouplePreds[i-init] = bdd.dico[i]["targetSimul"]
            notSimul[i] = False

    res = {"ML error" : np.sum((MLPreds - Y[init:])**2)/(len(Y)-init),
            "Couple regret" : np.sum((CouplePreds - Y[init:])**2)/(len(Y)-init),
            "simulator usage" : n_simul/(len(Y)-init),
            "ML regret" : np.sum((MLPreds[notSimul[init:]] - Y[notSimul])**2)/(np.sum(notSimul))}
    return res



