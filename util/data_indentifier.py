from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class InstanceData:
    J:              int
    I:              int
    X_ijk:          np.ndarray
    S_ij:           np.ndarray
    C_ij:           np.ndarray
    C_j :           np.ndarray
    p_ijk:          np.ndarray
    h_ijk:          np.ndarray
    d_j:            np.ndarray
    n_j:            np.ndarray
    MC_ji:          list
    n_MC_ji:        list
    OperationPool:  pd.DataFrame
    new_job_indices:list

@dataclass
class ScenarioData:
    JA_event:       list
    MB_event:       list 

@dataclass
class MasterData:
    p_ik:           np.ndarray
    h_ik:           np.ndarray
    n_:             np.ndarray
    MC_i:           list
    n_MC_i:         list
    duration:       int