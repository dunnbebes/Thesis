import numpy  as np
import copy
import random
from time                   import time


def action_space(J, K, p_ijk, h_ijk, d_j, n_j, 
                 S_k, S_j, MC_ji, n_ops_left_j, 
                 JSet, T_cur, Tard_job):
    method = Method(J, K, p_ijk, h_ijk, d_j, n_j, 
                    S_k, S_j, MC_ji, n_ops_left_j, 
                    JSet, T_cur, Tard_job)
    return [
          method.CDR1
        , method.CDR2
        , method.CDR3
        , method.CDR4
        , method.CDR5
        , method.CDR6
    ]


class Method:
    def __init__(self, J, K, p_ijk, h_ijk, d_j, n_j, 
                 S_k, S_j, MC_ji, n_ops_left_j, 
                 JSet, T_cur, Tard_job):
        
        self.J     			= copy.deepcopy(J)
        self.K  	   		= copy.deepcopy(K)
        self.p_ijk          = copy.deepcopy(p_ijk)
        self.h_ijk          = copy.deepcopy(h_ijk)
        self.d_j            = copy.deepcopy(d_j)
        self.n_j            = copy.deepcopy(n_j)
        self.S_k            = copy.deepcopy(S_k)
        self.S_j            = copy.deepcopy(S_j)
        self.MC_ji          = copy.deepcopy(MC_ji)
        self.n_ops_left_j   = copy.deepcopy(n_ops_left_j)
        self.JSet           = copy.deepcopy(JSet)
        self.T_cur          = copy.deepcopy(T_cur)
        self.Tard_job       = copy.deepcopy(Tard_job)


    def CDR1(self):
        p_mean              = np.mean(self.p_ijk*self.h_ijk, axis= 2)

        # Check if there is Tard_job
        if self.Tard_job:
            OP = self.n_j - self.n_ops_left_j
            estimated_tardiness = np.full(self.J, -np.inf)
            for j in self.Tard_job:
                estimated_tardiness[j] = self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
            j = np.argmax(estimated_tardiness)
        else:
            average_slack = np.full(self.J, np.inf)
            average_slack[self.JSet] = (self.d_j[self.JSet] - self.T_cur)/self.n_ops_left_j[self.JSet]
            j = np.argmin(average_slack)

        i                   = self.n_j[j] - self.n_ops_left_j[j]
        # Find earliest available machine
        available           = np.maximum(self.S_j[j], self.S_k)
        mask                = self.h_ijk[i, j, :] == 1
        filtered_available  = available[mask]
        min_value           = np.min(filtered_available)
        min_indices         = np.where(filtered_available == min_value)[0]
        filtered_index      = np.random.choice(min_indices)
        k                   = np.arange(len(available))[mask][filtered_index]

        return i, j, k


    def CDR2(self):
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        OP               = self.n_j - self.n_ops_left_j
        # Check if there is Tard_job
        if self.Tard_job:
            estimated_tardiness = np.full(self.J, -np.inf)
            for j in self.Tard_job:
                estimated_tardiness[j] = self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
            j = np.argmax(estimated_tardiness)
        else:
            critical_ratio = np.full(self.J, np.inf)
            for j in self.JSet:
                critical_ratio[j] = (self.d_j[j] - self.T_cur)/np.sum(p_mean[OP[j]:self.n_j[j], j])
            j = np.argmin(critical_ratio)

        i                   = self.n_j[j] - self.n_ops_left_j[j]
        # Find earliest available machine
        available           = np.maximum(self.S_j[j], self.S_k)
        mask                = self.h_ijk[i, j, :] == 1
        filtered_available  = available[mask]
        min_value           = np.min(filtered_available)
        min_indices         = np.where(filtered_available == min_value)[0]
        filtered_index      = np.random.choice(min_indices)
        k                   = np.arange(len(available))[mask][filtered_index]
        return i, j, k
       
    
    def CDR3(self):
        p_mean              = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        OP                  = self.n_j - self.n_ops_left_j

        # Select job with largest estimated tardiness
        estimated_tardiness = np.full(self.J, -np.inf)
        for j in self.JSet:
            estimated_tardiness[j] = self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
        j = np.argmax(estimated_tardiness)
        i = self.n_j[j] - self.n_ops_left_j[j]
        # Select machine
        r = random.random()
        workload = np.full(self.K, np.inf)
        for k in self.MC_ji[j][i]:
            workload[k] = np.sum(self.p_ijk[ope, job, k] * self.X_ijk[ope, job, k] for job in range(self.J) for ope in range(int(self.n_j[job] - self.n_ops_left_j[job])) )
        utilization = workload / self.T_cur

        mask = self.h_ijk[i, j, :] == 1
        if r < 0.5:           
            filtered_available  = utilization[mask]
            min_value           = np.min(filtered_available)
            min_indices         = np.where(filtered_available == min_value)[0]
            filtered_index      = np.random.choice(min_indices)
            k                   = np.arange(len(utilization))[mask][filtered_index]
        else:
            filtered_available  = workload[mask]
            min_value           = np.min(filtered_available)
            min_indices         = np.where(filtered_available == min_value)[0]
            filtered_index      = np.random.choice(min_indices)
            k                   = np.arange(len(workload))[mask][filtered_index]
        
        return i, j, k
            

    def CDR4(self):
        # Randomly select a job
        j                   = random.choice(self.JSet)
        i                   = self.n_j[j] - self.n_ops_left_j[j]
        # Find earliest available machine
        available           = np.maximum(self.S_j[j], self.S_k)
        mask                = self.h_ijk[i, j, :] == 1
        filtered_available  = available[mask]
        min_value           = np.min(filtered_available)
        min_indices         = np.where(filtered_available == min_value)[0]
        filtered_index      = np.random.choice(min_indices)
        k                   = np.arange(len(available))[mask][filtered_index]
        return i, j, k

    
    def CDR5(self):
        p_mean              = np.mean(self.p_ijk*self.h_ijk, axis= 2)        
        OP                  = self.n_j - self.n_ops_left_j

        # Check if there is Tard_job
        if self.Tard_job:
            InversedCompletionRate_tardiness = np.full(self.J, -np.inf)
            for j in self.Tard_job:
                InversedCompletionRate_tardiness[j] = 0 if OP[j] == 0 else self.n_j[j]/OP[j] * (self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j])
            j = np.argmax(InversedCompletionRate_tardiness)
        else:
            CompletionRate_slack = np.full(self.J, np.inf)
            for j in self.JSet:
                CompletionRate_slack[j] = OP[j]/self.n_j[j] * (self.d_j[j] - self.T_cur)
            j = np.argmin(CompletionRate_slack)

        i                   = self.n_j[j] - self.n_ops_left_j[j]
        # Find earliest available machine
        available           = np.maximum(self.S_j[j], self.S_k)
        mask                = self.h_ijk[i, j, :] == 1
        filtered_available  = available[mask]
        min_value           = np.min(filtered_available)
        min_indices         = np.where(filtered_available == min_value)[0]
        filtered_index      = np.random.choice(min_indices)
        k                   = np.arange(len(available))[mask][filtered_index]   
        return i, j, k

    def CDR6(self):
        p_mean              = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        OP                  = self.n_j - self.n_ops_left_j

        estimated_tardiness = np.full(self.J, -np.inf)
        for j in self.JSet:
            estimated_tardiness[j] = self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
        j                   = np.argmax(estimated_tardiness)
        i                   = self.n_j[j] - self.n_ops_left_j[j]
        # Find earliest available machine
        available           = np.maximum(self.S_j[j], self.S_k)
        mask                = self.h_ijk[i, j, :] == 1
        filtered_available  = available[mask]
        
        min_value           = np.min(filtered_available)
        min_indices         = np.where(filtered_available == min_value)[0]
        filtered_index      = np.random.choice(min_indices)
        k                   = np.arange(len(available))[mask][filtered_index]
        return i, j, k
           