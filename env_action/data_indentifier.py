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


instances = {
    "instance_1": InstanceData(X_ijk=np.random.rand(5, 5, 5), S_ij=np.random.rand(5, 5), C_ij=np.random.rand(5, 5), J=5, I=5),
    "instance_2": InstanceData(X_ijk=np.random.rand(6, 6, 6), S_ij=np.random.rand(6, 6), C_ij=np.random.rand(6, 6), J=6, I=6),
    # Add more instances as needed
}
    