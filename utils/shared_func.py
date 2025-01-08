
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin

def is_fitted(model):
    '''
    Checks if a scikit-learn estimator/transformer has already been fit.
    
    
    Parameters
    ----------
    model: scikit-learn estimator (e.g. RandomForestClassifier) 
        or transformer (e.g. MinMaxScaler) object
        
    
    Returns
    -------
    Boolean that indicates if ``model`` has already been fit (True) or not (False).
    '''
    
    attrs = [v for v in vars(model)
             if v.endswith("_") and not v.startswith("__")]
    
    return len(attrs) != 0

def dselect(M, n, randomstate = None):
    #forcer à ne jamais prendre deux fois le même
    M = 0.5*(M + M.T)
    ldm = [np.random.randint(0, len(M))]
    diss_M = np.zeros((len(M), n))
    for i, idx in enumerate(ldm):
        diss_x = M[:, idx]
        diss_M[:, i] = diss_x
        diss_M[idx, :] = -1

    while len(ldm) < n:
        idx = np.argmax(np.min(diss_M[:, :len(ldm)], axis = -1))
        ldm.append(idx)
        diss_x = M[:, idx]
        diss_M[:, len(ldm)-1] = diss_x
        diss_M[ldm, :] = -1

    return ldm
  

def generate_abrupt_drift(log_idx, classes, seed = None):
    if seed is not None:
        rd = np.random.RandomState(seed)
    else:
        rd = np.random.RandomState()
    class_order = [[2], [0 ,1 ,3]]
    #class_order = [[2]]
    stream = []
    for c in class_order:
        sub_train = [idx for idx in log_idx if classes[idx] in c]
        stream += rd.permutation(sub_train).tolist()
    return stream

def generate_stream(log_idx, seed = None):
    if seed is not None:
        rd = np.random.RandomState(seed)
    else:
        rd = np.random.RandomState()
    stream = rd.permutation(log_idx).tolist()
    return stream

def Score(i, j):
    s = (np.sum((i-j)**2))
    return s



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


class Simulator:
    def __init__(self, Nslots, Utime, maxFreeSlots = None):
        self.Nslots = Nslots
        self.counters = [0]*self.Nslots
        self.budget = 1
        self.Utime = Utime
        if maxFreeSlots is None:
            self.maxFreeSlots = self.Nslots
        else:
            self.maxFreeSlots = maxFreeSlots
        self.logs = [0]*self.Nslots
    
    
    def add(self, to_Add, idx):
        freeslots = [i for i in range(self.Nslots) if self.counters[i] == 0]
        is_added = False



        if to_Add and len(freeslots) > 0:
            i = freeslots[0]
            self.logs[i] = idx
            self.counters[i] = self.Utime
            is_added =  True
        return is_added
            
    def inc(self):
        endLogs = [log for c, log in zip(self.counters, self.logs) if c == 1]
        self.counters = [max(0, c-1) for c in self.counters]
        freeslots = [c for c in self.counters if c == 0]
        self.budget = len(freeslots)/self.Nslots
        self.budget = min(1, self.budget)
        return endLogs
        
    def get_budget(self):
        return self.budget



class Predictor(BaseEstimator, RegressorMixin):
    def __init__(self, seed = None):
        self.all_acc_scores = []
        self.coupling_acc_scores = []
        self.pred_acc_scores = []
    
        self.n_ldm = 20
        self.ldms = []
        self.idxs = []
        self.simulator = Simulator(10, 40)
        self.n_retrain = 50
        self.last_train = 0

        self.coupling_exp_correction = []
        self.k_coupling = 1
        self.alpha = 0.99

        self.squared = 0
        self.raw_scores = []

        if seed is None:
            self.randomstate = np.random.RandomState()
        else:
            self.randomstate = np.random.RandomState(seed)


    def predict(self, idx):
        return np.zeros((47,))

    def update_scores(self, is_added, y_pred, y_true):
        s = Score(y_pred, y_true)
        self.squared += s**2
        self.raw_scores.append(s)

        if len(self.all_acc_scores) == 0:
            self.all_acc_scores.append(s)
        else:
            k = len(self.all_acc_scores)
            self.all_acc_scores.append((self.all_acc_scores[-1]*k + s)/(k+1))

        if not is_added:
            if len(self.coupling_acc_scores) == 0:
                self.coupling_acc_scores.append(s)
            else:
                k = len(self.coupling_acc_scores)
                self.coupling_acc_scores.append((self.coupling_acc_scores[-1]*k + s)/(k+1))

            if len(self.pred_acc_scores) == 0:
                self.pred_acc_scores.append(s)
            else:
                k = self.k_coupling
                self.pred_acc_scores.append((self.pred_acc_scores[-1]*k + s)/(k+1))
                self.k_coupling += 1
        else:
            if len(self.coupling_acc_scores) == 0:
                self.coupling_acc_scores.append(s)
            else:
                k = len(self.coupling_acc_scores)
                self.coupling_acc_scores.append(self.coupling_acc_scores[-1]*k/(k+1))

            if len(self.pred_acc_scores) == 0:
                self.pred_acc_scores.append(s)
            else:
                self.pred_acc_scores.append(self.pred_acc_scores[-1])


    def get_scores(self):
        #all_exp_scores = [s/(1-(1-self.alpha)**(i+1)) for i, s in enumerate(self.all_exp_scores)]
        #coupling_exp_scores = [s/np.sum(self.coupling_exp_correction[:(i+1)]) for i, s in enumerate(self.coupling_exp_scores)]
        return self.all_acc_scores, self.coupling_acc_scores, self.pred_acc_scores

    def get_var(self):
        v = self.squared/len(self.all_acc_scores) - self.all_acc_scores[-1]**2
        return v

    