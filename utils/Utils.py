import sys


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import pickle as pkl
from .shared_func import *
from sklearn.utils import shuffle
from ..Coupling.PrecompSimul.master import MasterBddRF

SAVE = True
UTIME = 40
STREAM_RATE = 1
BUDGET = 0.2
N_TREE = 500
NSLOT = 10
N_RETRAIN = np.inf
INIT = 500

def run(X, Y):
    X, Y = shuffle(X, Y)

    """
    init models
    """


    GSM_model = MasterBddRF(X, Y, 
                            init = INIT,
                            budget = BUDGET, 
                            n_tree = N_TREE,
                            arrivalRate = STREAM_RATE,
                            simulationRate = UTIME,
                            nSlots = 1,
                            n_retrain = N_RETRAIN, 
                            strategy = "1inN")


    GSMx_model = MasterBddRF(X, Y, 
                            init = INIT,
                            budget = BUDGET, 
                            n_tree = N_TREE,
                            arrivalRate = STREAM_RATE,
                            simulationRate = 1/UTIME,
                            nSlots = NSLOT,
                            n_retrain = N_RETRAIN,
                            strategy = "max")


    Amb_model = MasterBddRF(X, Y, 
                            init = INIT,
                            budget = BUDGET, 
                            n_tree = N_TREE,
                            arrivalRate = STREAM_RATE,
                            simulationRate = 1/UTIME,
                            nSlots = NSLOT,
                            n_retrain = N_RETRAIN,
                            strategy = "disagreement")

    Smp_model = MasterBddRF(X, Y, 
                            init = INIT,
                            budget = BUDGET, 
                            n_tree = N_TREE,
                            arrivalRate = STREAM_RATE,
                            simulationRate = 1/UTIME,
                            nSlots = NSLOT,
                            n_retrain = N_RETRAIN,
                            strategy = "sampling")

    Dist_model = MasterBddRF(X, Y, 
                            init = INIT,
                            budget = BUDGET, 
                            n_tree = N_TREE,
                            arrivalRate = STREAM_RATE,
                            simulationRate = 1/UTIME,
                            nSlots = NSLOT,
                            n_retrain = N_RETRAIN,
                            strategy = "diversity")

    Con_model = MasterBddRF(X, Y, 
                            init = INIT,
                            budget = BUDGET, 
                            n_tree = N_TREE,
                            arrivalRate = STREAM_RATE,
                            simulationRate = 1/UTIME,
                            nSlots = NSLOT,
                            n_retrain = N_RETRAIN,
                            strategy = "consistency")


    GSM_model.run()
    res = GSM_model.evaluate(Print = False)
    GSM_ML_error = res["ML error"]
    GSM_ML_regret = res["ML regret"]
    GSM_Couple_regret = res["Couple regret"]
    GSM_b = res["simulator usage"]


    GSMx_model.run()
    res = GSMx_model.evaluate(Print = False)
    GSMx_ML_error = res["ML error"]
    GSMx_ML_regret = res["ML regret"]
    GSMx_Couple_regret = res["Couple regret"]
    GSMx_b = res["simulator usage"]


    Amb_model.run()
    res = Amb_model.evaluate(Print = False)
    Amb_ML_error = res["ML error"]
    Amb_ML_regret = res["ML regret"]
    Amb_Couple_regret = res["Couple regret"]
    Amb_b = res["simulator usage"]


    Smp_model.run()
    res = Smp_model.evaluate(Print = False)
    Smp_ML_error = res["ML error"]
    Smp_ML_regret = res["ML regret"]
    Smp_Couple_regret = res["Couple regret"]
    Smp_b = res["simulator usage"]


    Dist_model.run()
    res = Dist_model.evaluate(Print = False)
    Dist_ML_error = res["ML error"]
    Dist_ML_regret = res["ML regret"]
    Dist_Couple_regret = res["Couple regret"]
    Dist_b = res["simulator usage"]


    Con_model.run()
    res = Con_model.evaluate(Print = False)
    Con_ML_error = res["ML error"]
    Con_ML_regret = res["ML regret"]
    Con_Couple_regret = res["Couple regret"]
    Con_b = res["simulator usage"]


 
    return (GSM_ML_error, 
            GSM_ML_regret, 
            GSM_Couple_regret,
            GSM_b,
            GSMx_ML_error, 
            GSMx_ML_regret, 
            GSMx_Couple_regret,
            GSMx_b,
            Amb_ML_error, 
            Amb_ML_regret, 
            Amb_Couple_regret,
            Amb_b,
            Smp_ML_error, 
            Smp_ML_regret, 
            Smp_Couple_regret,
            Smp_b,
            Dist_ML_error, 
            Dist_ML_regret, 
            Dist_Couple_regret,
            Dist_b,
            Con_ML_error, 
            Con_ML_regret, 
            Con_Couple_regret,
            Con_b)

def repeatRun(X, Y, nRepeat, n_jobs = 1, PRINT = True):
    rep_GSM_ML_error = []
    rep_GSM_ML_regret = []
    rep_GSM_coupling_regret = []
    rep_GSM_b = []
    rep_GSMx_ML_error = []
    rep_GSMx_ML_regret = []
    rep_GSMx_coupling_regret = []
    rep_GSMx_b = []
    rep_Amb_ML_error = []
    rep_Amb_ML_regret = []
    rep_Amb_coupling_regret = []
    rep_Amb_b = []
    rep_Smp_ML_error = []
    rep_Smp_ML_regret = []
    rep_Smp_coupling_regret = []
    rep_Smp_b = []
    rep_Dist_ML_error = []
    rep_Dist_ML_regret = []
    rep_Dist_coupling_regret = []
    rep_Dist_b = []
    rep_Con_ML_error = []
    rep_Con_ML_regret = []
    rep_Con_coupling_regret = []
    rep_Con_b = []
    

    results = Parallel(n_jobs = n_jobs)(delayed(run)(X, Y) for _ in range(nRepeat))

    for i in range(nRepeat):

        (GSM_ML_error, 
            GSM_ML_regret, 
            GSM_Couple_regret,
            GSM_b,
            GSMx_ML_error, 
            GSMx_ML_regret, 
            GSMx_Couple_regret,
            GSMx_b,
            Amb_ML_error, 
            Amb_ML_regret, 
            Amb_Couple_regret,
            Amb_b,
            Smp_ML_error, 
            Smp_ML_regret, 
            Smp_Couple_regret,
            Smp_b,
            Dist_ML_error, 
            Dist_ML_regret, 
            Dist_Couple_regret,
            Dist_b,
            Con_ML_error, 
            Con_ML_regret, 
            Con_Couple_regret,
            Con_b) = results[i]

        rep_GSM_ML_error.append(GSM_ML_error)
        rep_GSM_ML_regret.append(GSM_ML_regret)
        rep_GSM_coupling_regret.append(GSM_Couple_regret)
        rep_GSM_b.append(GSM_b)
        rep_GSMx_ML_error.append(GSMx_ML_error)
        rep_GSMx_ML_regret.append(GSMx_ML_regret)
        rep_GSMx_coupling_regret.append(GSMx_Couple_regret)
        rep_GSMx_b.append(GSMx_b)
        rep_Amb_ML_error.append(Amb_ML_error)
        rep_Amb_ML_regret.append(Amb_ML_regret)
        rep_Amb_coupling_regret.append(Amb_Couple_regret)
        rep_Amb_b.append(Amb_b)
        rep_Smp_ML_error.append(Smp_ML_error)
        rep_Smp_ML_regret.append(Smp_ML_regret)
        rep_Smp_coupling_regret.append(Smp_Couple_regret)
        rep_Smp_b.append(Smp_b)
        rep_Dist_ML_error.append(Dist_ML_error)
        rep_Dist_ML_regret.append(Dist_ML_regret)
        rep_Dist_coupling_regret.append(Dist_Couple_regret)
        rep_Dist_b.append(Dist_b)
        rep_Con_ML_error.append(Con_ML_error)
        rep_Con_ML_regret.append(Con_ML_regret)
        rep_Con_coupling_regret.append(Con_Couple_regret)
        rep_Con_b.append(Con_b)


    if PRINT:
        print("greedy mean")
        print("all preds : ", np.mean(np.array(rep_GSM_ML_error)), np.std(np.array(rep_GSM_ML_error)))
        print("ML regret : ", np.mean(np.array(rep_GSM_ML_regret)), np.std(np.array(rep_GSM_ML_regret)))
        print("coupling regret : ", np.mean(np.array(rep_GSM_coupling_regret)), np.std(np.array(rep_GSM_coupling_regret)))
        print("b : ", np.mean(rep_GSM_b), np.std(rep_GSM_b))

        print("greedy max")
        print("all preds : ", np.mean(np.array(rep_GSMx_ML_error)), np.std(np.array(rep_GSMx_ML_error)))
        print("ML regret : ", np.mean(np.array(rep_GSMx_ML_regret)), np.std(np.array(rep_GSMx_ML_regret)))
        print("coupling regret : ", np.mean(np.array(rep_GSMx_coupling_regret)), np.std(np.array(rep_GSMx_coupling_regret)))
        print("b : ", np.mean(rep_GSMx_b), np.std(rep_GSMx_b))

        print("AL ambiguity")
        print("all preds : ", np.mean(np.array(rep_Amb_ML_error)), np.std(np.array(rep_Amb_ML_error)))
        print("ML regret : ", np.mean(np.array(rep_Amb_ML_regret)), np.std(np.array(rep_Amb_ML_regret)))
        print("coupling regret : ", np.mean(np.array(rep_Amb_coupling_regret)), np.std(np.array(rep_Amb_coupling_regret)))
        print("b : ", np.mean(rep_Amb_b), np.std(rep_Amb_b))

        print("AL sampling")
        print("all preds : ", np.mean(np.array(rep_Smp_ML_error)), np.std(np.array(rep_Smp_ML_error)))
        print("ML regret : ", np.mean(np.array(rep_Smp_ML_regret)), np.std(np.array(rep_Smp_ML_regret)))
        print("coupling regret : ", np.mean(np.array(rep_Smp_coupling_regret)), np.std(np.array(rep_Smp_coupling_regret)))
        print("b : ", np.mean(rep_Smp_b), np.std(rep_Smp_b))

        print("AL diss")
        print("all preds : ", np.mean(np.array(rep_Dist_ML_error)), np.std(np.array(rep_Dist_ML_error)))
        print("ML regret : ", np.mean(np.array(rep_Dist_ML_regret)), np.std(np.array(rep_Dist_ML_regret)))
        print("coupling regret : ", np.mean(np.array(rep_Dist_coupling_regret)), np.std(np.array(rep_Dist_coupling_regret)))
        print("b : ", np.mean(rep_Dist_b), np.std(rep_Dist_b))

        print("AL Con")
        print("all preds : ", np.mean(np.array(rep_Con_ML_error)), np.std(np.array(rep_Con_ML_error)))
        print("ML regret : ", np.mean(np.array(rep_Con_ML_regret)), np.std(np.array(rep_Con_ML_regret)))
        print("coupling regret : ", np.mean(np.array(rep_Con_coupling_regret)), np.std(np.array(rep_Con_coupling_regret)))
        print("b : ", np.mean(rep_Con_b), np.std(rep_Con_b))

    return [rep_GSM_ML_error,
                    rep_GSM_ML_regret,
                    rep_GSM_coupling_regret,
                    rep_GSM_b,
                    rep_GSMx_ML_error,
                    rep_GSMx_ML_regret,
                    rep_GSMx_coupling_regret,
                    rep_GSMx_b,
                    rep_Amb_ML_error,
                    rep_Amb_ML_regret,
                    rep_Amb_coupling_regret,
                    rep_Amb_b,
                    rep_Smp_ML_error,
                    rep_Smp_ML_regret,
                    rep_Smp_coupling_regret,
                    rep_Smp_b,
                    rep_Dist_ML_error,
                    rep_Dist_ML_regret,
                    rep_Dist_coupling_regret,
                    rep_Dist_b,
                    rep_Con_ML_error,
                    rep_Con_ML_regret,
                    rep_Con_coupling_regret,
                    rep_Con_b]