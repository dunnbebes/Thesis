import pandas as pd
import numpy as np
import copy
import random
from util.util_action import find_Mch_seq

def find_indices(X, k):
    indices = np.argwhere(X[:, :, k] == 1)
    return [(idx[0], idx[1]) for idx in indices]

def find_S_j (t, JSet, ODSet, C_ij, J):
    S_j = np.zeros(J)
    # Create a boolean mask for jobs with empty ODSet
    empty_mask = np.array([len(ODSet[j]) == 0 for j in JSet])

    # Handle the non-empty and empty cases
    non_empty_jobs = np.array(JSet)[~empty_mask]
    if non_empty_jobs.size > 0:
        last_operations = np.array([ODSet[j][-1] for j in non_empty_jobs])
        max_C_ij = np.max(C_ij[last_operations, non_empty_jobs], axis=0)
        S_j[non_empty_jobs] = np.maximum(max_C_ij, t)

    # Set S_j values for jobs with empty ODSet
    S_j[np.array(JSet)[empty_mask]] = t

    return S_j


def change_dataset(p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, j, i, processed, I, K, org_p_ijk, org_h_ijk):
    """p_ijk, h_ijk, X_ijk, S_ij, C_ij"""
    slice_p_ijk   = p_ijk[:, j, :].copy()
    slice_h_ijk   = h_ijk[:, j, :].copy()
    slice_X_ijk   = X_ijk[:, j, :].copy()
    slice_S_ij    = S_ij [:, j]   .copy()
    slice_C_ij    = C_ij [:, j]   .copy()

    # Check if need modify shape
   
    if n_j[j] < slice_p_ijk.shape[0]:
        # Move rows down
        dummy_p = slice_p_ijk.copy()
        dummy_h = slice_h_ijk.copy()
        dummy_X = slice_X_ijk.copy()
        dummy_S = slice_S_ij .copy()
        dummy_C = slice_C_ij .copy()

        for operation in range(i, int(n_j[j])):
            dummy_p[operation + 1] = copy.deepcopy(slice_p_ijk[operation])
            dummy_h[operation + 1] = copy.deepcopy(slice_h_ijk[operation])
            dummy_X[operation + 1] = copy.deepcopy(slice_X_ijk[operation])
            dummy_S[operation + 1] = copy.deepcopy(slice_S_ij [operation])
            dummy_C[operation + 1] = copy.deepcopy(slice_C_ij [operation])

        slice_p_ijk = copy.deepcopy(dummy_p)
        slice_h_ijk = copy.deepcopy(dummy_h)
        slice_X_ijk = copy.deepcopy(dummy_X)  
        slice_S_ij  = copy.deepcopy(dummy_S)
        slice_C_ij  = copy.deepcopy(dummy_C)

        # Update row i+1 with max(0, row index i - processed)
        slice_p_ijk[i+1] = np.maximum(0, slice_p_ijk[i] - processed)
        slice_S_ij [i+1] = slice_S_ij[i] + processed
        slice_C_ij [i]   = slice_S_ij[i] + processed

        p_ijk[:, j, :] = slice_p_ijk.copy()
        h_ijk[:, j, :] = slice_h_ijk.copy()
        X_ijk[:, j, :] = slice_X_ijk.copy()
        S_ij [:, j]    = slice_S_ij .copy()
        C_ij [:, j]    = slice_C_ij .copy()
    else:
        # Insert a new row after row index i
        new_row_p      = np.maximum(0, slice_p_ijk[i] - processed)
        new_row_h      = slice_h_ijk  [i].copy()
        new_row_X      = slice_X_ijk  [i].copy()
        new_ele_S      = slice_S_ij   [i] + processed
        cur_ele_C      = slice_S_ij   [i] + processed

        slice_p_ijk    = np.insert(slice_p_ijk,   i+1, new_row_p, axis=0)
        slice_h_ijk    = np.insert(slice_h_ijk,   i+1, new_row_h, axis=0)
        slice_X_ijk    = np.insert(slice_X_ijk,   i+1, new_row_X, axis=0)
        slice_S_ij     = np.insert(slice_S_ij,    i+1, new_ele_S, axis=0)
        slice_C_ij     = np.insert(slice_C_ij,    i,   cur_ele_C, axis=0)

        p_ijk          = np.pad(p_ijk,     ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        h_ijk          = np.pad(h_ijk,     ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        X_ijk          = np.pad(X_ijk,     ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        S_ij           = np.pad(S_ij,      ((0, 1), (0, 0)),         mode='constant', constant_values=0)
        C_ij           = np.pad(C_ij,      ((0, 1), (0, 0)),         mode='constant', constant_values=0)

        p_ijk[:, j, :] = slice_p_ijk  .copy()
        h_ijk[:, j, :] = slice_h_ijk  .copy()
        X_ijk[:, j, :] = slice_X_ijk  .copy()
        S_ij [:, j]    = slice_S_ij   .copy()
        C_ij [:, j]    = slice_C_ij   .copy()

        org_p_ijk      = np.pad(org_p_ijk, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
        org_h_ijk      = np.pad(org_h_ijk, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)

        I += 1

    for k in range(K):
        p_ijk[i,j,k] = processed if p_ijk[i,j,k] != 999 else 999

    """MC_ji and n_MC_ji"""
    sublist_to_copy1   = copy.deepcopy(MC_ji[j][i])
    sublist_to_copy2   = copy.deepcopy(n_MC_ji[j][i])
    MC_ji  [j].insert(i + 1, sublist_to_copy1)
    n_MC_ji[j].insert(i + 1, sublist_to_copy2)

    """n_j"""
    n_j[j] += 1
    
    return p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, I, org_p_ijk, org_h_ijk

def random_events(t, J, K, X_ijk, S_ij, C_ij, C_j, JA_event, MB_event, MB_record):
    events          = {}
    have_event      = False
    for job, deadline, description in JA_event:
        have_event  = True
        time_occur  = copy.deepcopy(C_j[job])
        if time_occur not in events:
            events[time_occur] = []
        events[time_occur].append(("JA", job, deadline, description))

    for k in range(K):
        if MB_event[k]:
            have_event  = True
            time_occur, repair, description = MB_event[k][0]
            if time_occur not in events:
                events[time_occur] = []
            events[time_occur].append(("MB", k, repair, description))
    
    re            = np.zeros((K)) 
    if have_event == True:
        found         = None
        while found == None:
            if events:
                affected_Oij    = {}
                new_t           = copy.deepcopy(int(min(events.keys())))
                triggered_event = copy.deepcopy(events[new_t])
                if new_t <= t:
                    new_time = t+300
                else:
                    new_time = copy.deepcopy(new_t)
                need_modify     = 0
                for uncertain_type, k, time_event, description in triggered_event:
                    if uncertain_type == "MB":
                        X_mask =  X_ijk.astype(bool)
                        re[k]  =  time_event
                        """Find affected operation"""
                        # Use the boolean mask to find the indices where overlap occurs
                        overlap_mask = np.logical_or(
                            np.logical_and(S_ij >= new_time        , C_ij <= new_time + re[k]),       # in
                            np.logical_and(S_ij <= new_time        , C_ij >  new_time        ),       # left
                            np.logical_and(S_ij <  new_time + re[k], C_ij >= new_time + re[k]))       # right
                        indices = np.argwhere(X_mask[:, :, k] & overlap_mask[:, :])
                        # Append the indices to the affected_Oij dictionary
                        if len(indices) > 0:
                            # adjusted_indices = [[i, j] for i, j in indices] # Due to segmentize the operation
                            affected_Oij[k] = indices.tolist()
                        if k not in affected_Oij:
                            event = (uncertain_type, k, time_event, description)
                            # Remove current time
                            events[new_t].remove(event)
                            if len (events[new_t]) == 0:
                                events.pop(new_t)
                            # Adjust to new time (shift_time)
                            mask          = np.logical_and(S_ij > new_time, X_ijk[:, :, k] == 1)
                            filtered_S_ij = S_ij[mask]

                            if filtered_S_ij.size > 0:
                                shift_time    = np.min(filtered_S_ij)
                                if shift_time not in events:
                                    events[shift_time] = []
                                events[shift_time].append(event)

                            need_modify += 1
                        
                if need_modify == 0:
                    found = True
            else:
                break

        if events: # After While loop, If have events
            for uncertain_type, partID, time_event, description in triggered_event:
                if uncertain_type == "JA":
                    JA_event = [JA for JA in JA_event if JA[0] != partID]
                else:
                    if partID not in MB_record:
                        MB_record[partID] = []
                    record = (new_time, new_time+re[partID])
                    MB_record[partID].append(record)
                    MB_event[partID].pop(0)
            # for key, value in events.items():
            #     if key < new_time:
            #         new_key = new_time + 60
            #         if new_key in new_data:
            #             events[new_key].extend(value)
            #         else:
            #             events[new_key] = value
            if not triggered_event:
                new_time = np.max(C_j)
                triggered_event = None     
        else:
            new_time = np.max(C_j)
            triggered_event = None 

    else:
        new_time = np.max(C_j)
        triggered_event = None  
    
    return JA_event, MB_event, new_time, triggered_event, re, MB_record

    
def snapshot(t, triggered_event, MC_ji, n_MC_ji,                 \
             d_j, n_j, p_ijk, h_ijk, J, I, K, X_ijk, S_ij, C_ij, \
             OperationPool, re, S_k,                             \
             org_J, org_p_ijk, org_h_ijk, org_n_j,               \
             org_MC_ji, org_n_MC_ji, C_j                         ):
    
    # Set ----------------------------------------------------------------
    ## Job and operation still need to be scheduled
    LS_j  = np.max(S_ij,axis=0)
    JSet  = np.where(LS_j >= t)[0].tolist()
    OJSet = [[] for _ in range(J)]
    for j in JSet:
        OJSet[j] = np.where(S_ij[:int(n_j[j]), j] >= t)[0].tolist()


    ## Job and operation has been run
    ES_j  = np.min(S_ij, axis=0)
    DSet  = np.where(ES_j < t)[0].tolist()
    ODSet = [[] for _ in range(J)]
    for j in DSet:
        ODSet[j] = np.where(S_ij[:int(n_j[j]), j] < t)[0].tolist()

    #------------------------------------------------------------------------
    JA                  = []
    MB                  = [[] for _ in range(K)]
    Oij_on_machine      = [[] for _ in range(K)]
    JA_urgent_boolean   = 0
    JA_long_boolean     = 0
    MB_critical_boolean = 0
    NewJobList          = []
    MBList              = []
    if triggered_event is not None:
        for uncertain_type, partID, time_event, description in triggered_event:
            if uncertain_type == "MB": 
                MB[partID].append((time_event, description))    # if Machine break down
                if description == "critical": 
                    MB_critical_boolean = 1
                    print('MB: critical')
            else:            
                JA.append((partID, time_event, description))   # if Job arrival

        # Operation on machine ------------------------------------------------
        for k in range(K):
            i, j = np.where((X_ijk[:, :, k] == 1) & (S_ij < t) & (C_ij > t))
            if len(i) != 0 and len(j) != 0:
                operation = (i,j)
                # Oij_on_machine[k].append(operation)
                Oij_on_machine[k] = copy.deepcopy(operation)

        # Soonest start time ------------------------------------------------------
        idle     = np.zeros((K)) # boolean, =1 if machine idle at time t, 0 otherwise (busy processing or repair)
        bu       = np.zeros((K)) # boolean, =1 if machine busy processing, = 0 if machine is repaired
        av       = np.zeros((K)) # remaining processing time 

        for k in range(K):
            if not MB[k]:
                operation = Oij_on_machine[k]
                if len(operation) != 0:
                    i = copy.deepcopy(operation[0][0])
                    j = copy.deepcopy(operation[1][0])
                    av[k] = C_ij[i, j] - t
                    bu[k] = 1
                else:   
                    idle[k] = 1
            else:
                MBList.append(k)
                operation = Oij_on_machine[k]
                if len(operation) != 0:
                    i = copy.deepcopy(operation[0][0])
                    j = copy.deepcopy(operation[1][0])
                    processed  = t - S_ij[i, j]
                
                    """Break the operation into  2 segments"""
                    p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, I, org_p_ijk, org_h_ijk = change_dataset(p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, j, i, processed, I, K, org_p_ijk, org_h_ijk)
                    OJSet[j].append(n_j[j]-1)                    
                    if j not in JSet:
                        JSet.append(j)

        S_k = np.maximum(t + (1-idle)*(bu*av + (1-bu)*re), S_k)
        
        # Check if JA -----------------------------------------------------------
        if JA: 
            for jobresemble, deadline, description in JA: 
                """Adjust the dataset"""              
                J += 1
                NewJobList.append((J-1, description))
                # num operation of new job
                n_newjob                   = copy.deepcopy(org_n_j[jobresemble])
                n_j                        = np.append(n_j, n_newjob)
                # processing time
                p_newjob                   = copy.deepcopy(org_p_ijk[:, jobresemble, :])
                p_newjob_reshape           = copy.deepcopy(p_newjob[:, np.newaxis, :])
                p_ijk                      = np.concatenate((p_ijk, p_newjob_reshape), axis= 1)
                
                # capable machine            
                h_newjob                   = copy.deepcopy(org_h_ijk[:, jobresemble, :])
                h_newjob_reshape           = copy.deepcopy(h_newjob[:, np.newaxis, :])
                h_ijk                      = np.concatenate((h_ijk, h_newjob_reshape), axis= 1)

                MC_newjob                  = copy.deepcopy(org_MC_ji[jobresemble])
                n_MC_newjob                = copy.deepcopy(org_n_MC_ji[jobresemble])
                MC_ji  .append(MC_newjob)
                n_MC_ji.append(n_MC_newjob)

                # Adjust the org
                org_n_j                    = np.append(org_n_j, n_newjob)
                org_p_ijk                  = np.concatenate((org_p_ijk, p_newjob_reshape), axis= 1)
                org_h_ijk                  = np.concatenate((org_h_ijk, h_newjob_reshape), axis= 1)
                org_MC_ji.append(MC_newjob)
                org_n_MC_ji.append(n_MC_newjob)

                new_row_values                        = [J-1, n_j[J-1]]
                OperationPool.loc[len(OperationPool)] = new_row_values

                JSet.append(J-1)
                OJSet.append(list(range(int(n_newjob))))
                ODSet.append([])

                TPT                        = np.sum(np.sum(p_newjob * h_newjob, axis= 1)/ np.maximum(np.sum(h_newjob, axis=1),1))
                if TPT > 1000:
                    JA_long_boolean = 1
                    print("JA: long TPT")
                # deadline
                if description == "urgent":
                    d_newjob = TPT*deadline
                    JA_urgent_boolean = 1
                    print("JA: urgent")
                else:
                    d_newjob = deadline
                d_j                        = np.append(d_j, d_newjob)

                # X, S, C
                X_ijk          = np.pad(X_ijk,     ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
                S_ij           = np.pad(S_ij,      ((0, 0), (0, 1)),         mode='constant', constant_values=0)
                C_ij           = np.pad(C_ij,      ((0, 0), (0, 1)),         mode='constant', constant_values=0)
                C_j            = np.pad(C_j,       ((0, 1)),                 mode='constant', constant_values=0)
                """"Check urgency"""
                h_excluded     = h_newjob[1:int(n_newjob -1)]
                p_excluded     = p_newjob[1:int(n_newjob -1)]
                LC_firstope    = d_newjob - np.sum(np.sum(p_excluded*h_excluded, axis=1)/np.maximum(np.sum(h_excluded, axis=1),1))

                Acceptable     = [k for k in MC_newjob[0] if S_k[k] + p_newjob[0][k] <= LC_firstope]
                if len(Acceptable) == 0:
                    # Find LC of operation currently on machine k
                    LC_operation_on_k = np.full(K, np.inf)
                    for k in MC_newjob[0]:
                        operation = copy.deepcopy(Oij_on_machine[k])

                        if len(operation) != 0:
                            i = copy.deepcopy(operation[0][0])
                            j = copy.deepcopy(operation[1][0])

                            # Calculate LC of the operation
                            mean_p = np.sum(p_ijk[:,j,:]*h_ijk[:,j,:], axis=1)/np.maximum(np.sum(h_ijk[:,j,:], axis=1), 1)
                            LC_operation_on_k[k] = d_j[j] - np.sum(mean_p[i+1 : int(n_j[j]-1)])

                    # Consider to remove current operation on machine
                    Considered_list = [k for k in MC_newjob[0] if LC_firstope < LC_operation_on_k[k] and LC_operation_on_k[k] != np.inf]
                    if len(Considered_list) != 0:
                        sums        = S_k[Considered_list] + p_newjob[0, Considered_list]
                        k_min_index = np.argmin(sums)
                        k           = copy.deepcopy(Considered_list[k_min_index])
                        if Oij_on_machine[k]:
                            operation  = copy.deepcopy(Oij_on_machine[k]) 
                            i = copy.deepcopy(operation[0][0])
                            j = copy.deepcopy(operation[1][0])
                            processed  = t - S_ij[i, j]

                            """Break the operation into  2 segments"""
                            p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, I, org_p_ijk, org_h_ijk = change_dataset(p_ijk, h_ijk, X_ijk, S_ij, C_ij, MC_ji, n_MC_ji, n_j, j, i, processed, I, K, org_p_ijk, org_h_ijk)
                            OJSet[j].append(n_j[j]-1)                    # need to schedule the remaining of affected operation
                            if j not in JSet:
                                JSet.append(j)
    S_j = np.zeros((J))
    if triggered_event is not None:
        for j in JSet:
            if len(ODSet[j]) == 0:
                S_j[j] = t
            else: 
                last_operation = ODSet[j][-1]
                S_j[j] = max(C_ij[last_operation][j], t)

    # Number of operation of job j that have been run ---------------------------------
    n_ops_left_j = np.array([len(sublist) for sublist in OJSet])
    OperationPool['Num operation left'] = n_ops_left_j.astype(int)

    # Observation --------------------------------------------------------------------    
    JA_boolean = 1 if JA else 0
    MB_boolean = 1 if any(breakdown for breakdown in MB) else 0
    sum_re     = np.sum(re)/(60*60*3)
    print('sum_re:', sum_re)
    # Find completion time of last operation assigned to machine k at rescheduling point t
    CT_k = np.zeros(K)
    for k in range(K):  
        indices = np.argwhere((X_ijk[:, :, k] == 1) & (S_ij < t))  
        if len(indices) > 0:
            max_index = np.argmax(S_ij[indices[:, 0], indices[:, 1]])
            i, j      = indices[max_index]
            CT_k[k]   = C_ij[i, j] 


    T_cur  = np.zeros((J))
    OP_cur = (n_j - n_ops_left_j).astype(int)
    for j in JSet:
        i = OP_cur[j]
        T_cur[j] = np.mean(S_k[h_ijk[i,j] == 1])

    Tard_job = [j for j in JSet if d_j[j] < T_cur[j]]

    return  S_k, S_j, J, I, JSet, OJSet, DSet, ODSet, OperationPool, \
            n_ops_left_j, MC_ji, n_MC_ji, d_j, n_j, p_ijk, h_ijk,    \
            org_p_ijk, org_h_ijk, org_n_j, org_MC_ji, org_n_MC_ji,   \
            X_ijk, S_ij, C_ij, C_j,                                  \
            JA_boolean, JA_long_boolean, JA_urgent_boolean,          \
            MB_boolean, MB_critical_boolean, sum_re,                 \
            CT_k, T_cur, Tard_job, NewJobList, MBList


def store_schedule(X_ijk, S_ij, C_ij):
    X_previous = X_ijk.copy()
    S_previous = S_ij .copy()
    C_previous = C_ij .copy()
    return X_previous, S_previous, C_previous


def update_schedule(DSet, ODSet, t, X_ijk, S_ij, C_ij, X_previous, S_previous, C_previous):
    # for j in DSet:
    #     for i in ODSet[j]:
    #         X_ijk[i, j] = copy.deepcopy(X_previous[i, j])
    #         S_ij [i, j] = copy.deepcopy(S_previous[i, j])
    #         C_ij [i, j] = copy.deepcopy(C_previous[i, j])
    S_mask = S_ij < t
    S_mask_expand = S_mask[:, :, np.newaxis]
    X_ijk = np.where(S_mask_expand, X_previous, X_ijk)
    S_ij  = np.where(S_mask, S_previous, S_ij)
    C_ij  = np.where(S_mask, C_previous, C_ij)
    
    C_j = np.max(C_ij, axis=0)
    return X_ijk, S_ij, C_ij, C_j

def generate_random_event (J, K, planning_horizon, WeibullDistribution, critical_machines, ReworkProbability):
    import ast
    import math
    from scipy.stats import weibull_min

    JA_event = []
    MB_event = [[] for _ in range(K)]
    # Determine description based on scenario
    description = random.choice(['urgent', 'normal', 'loose'])

    # Set deadline based on description
    if description == 'urgent':
        Deadline = np.random.randint(1, 11, size=J)
    elif description == 'normal':
        Deadline = np.full(J, planning_horizon)
    elif description == 'loose':
        Deadline = np.full(J, 2 * planning_horizon)

    # Generate defected jobs with 3% probability
    defected_jobs_indices = np.where(np.random.uniform(size=J) < ReworkProbability)[0]  # One-indexed
    for job_id in defected_jobs_indices:
        JA_event.append((job_id, Deadline[job_id], description)) # Adjusting for one-indexing

    # Iterate over each row in the DataFrame
    for _, row in WeibullDistribution.iterrows():
        machine_id                       = row['MachineID']
        shape_up, loc_up, scale_up       = ast.literal_eval(row['ParameterUpTime'])
        shape_down, loc_down, scale_down = ast.literal_eval(row['ParameterDownTime'])
        description                      = "critical" if machine_id in critical_machines else "normal"
        # Calculate uptime and downtime
        t = -604800*2
        downtime = 0
        while t < planning_horizon:
            skip     = False
            t       += downtime
            uptime   = math.ceil(weibull_min.rvs(c=shape_up, loc=0, scale=scale_up))*60 # change to second
            count    = 0
            while uptime < 300:
                uptime   = math.ceil(weibull_min.rvs(c=shape_up, loc=0, scale=scale_up))*60 # change to second
                count += 1
                if count == 10:
                    skip = True
                    break
            downtime = math.ceil(weibull_min.rvs(c=shape_down, loc=0, scale=scale_down))*60 # change to second
            t += uptime
            if skip == True:
                break
            else:
                if t > 0 and t < planning_horizon:
                    MB_event[machine_id - 1].append((t, downtime, description))
    
    return JA_event, MB_event

def generate_JA_event (J, planning_horizon, ReworkProbability):
    JA_event = {}
    # Determine description based on scenario
    description = random.choice(['urgent', 'normal', 'loose'])

    # Set deadline based on description
    if description == 'urgent':
        Deadline = np.random.randint(1, 11, size=J)
    elif description == 'normal':
        Deadline = np.full(J, planning_horizon)
    elif description == 'loose':
        Deadline = np.full(J, 1.5 * planning_horizon)

    # Generate defected jobs with 3% probability
    defected_jobs_indices = np.where(np.random.uniform(size=J) < ReworkProbability)[0]  

    for job_id in defected_jobs_indices:
        JA_event[job_id] = (Deadline[job_id],description)

    return JA_event