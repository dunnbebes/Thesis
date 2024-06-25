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

@dataclass
class ScenarioData:
    JA_event:       list
    MB_event:       list 