import os
import sys
mdpath = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
data_path = os.path.join(mdpath, 'CouplingDT\\data')
sys.path.append(mdpath)

from typing import DefaultDict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import pickle as pkl
from CouplingDT.utils.Utils import repeatRun
from CouplingDT.utils.loaders import *


"""
global variables
"""

N_EXP = 50

"""
load and process dataset
"""
df = loadMusic(data_path)

if __name__ == "__main__":

    [rep_GSM_ML_error,
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
        rep_Con_b] = repeatRun(X, Y, nRepeat = N_EXP, n_jobs = 7)

    if SAVE:
        pkl.dump([rep_GSM_ML_error,
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
                    rep_Con_b],
                    open("C:/Users/Sylvain Chabanet/OneDrive - Universite de Lorraine/Documents/these/experimental/python/Coupling/multi_dataset/results/GOMT40_2.pkl", "wb")
                )
