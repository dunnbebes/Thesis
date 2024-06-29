import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy  as np
import copy
import random
import time  
from time                   import time  
from pulp                   import *                           
from env_action.metaheu     import random_population, GeneticAlgorithm, encode_schedule, generate_neighborhood, TabuSearch
from util.util_action       import find_Mch_seq, evaluate_LocalCost, RightShift
from scipy.optimize         import linear_sum_assignment


def action_space(J, I, K, p_ijk, h_ijk, d_j, n_j, 
                 MC_ji, n_MC_ji, n_ops_left_j, OperationPool, S_k, S_j, 
                 JSet, OJSet, t, X_ijk, S_ij, C_ij, C_j, CT_k, T_cur, Tard_job,
                 NewJobList, MBList, PopSize, maxtime, re):
    method = Method(J, I, K, p_ijk, h_ijk, d_j, n_j, 
                    MC_ji, n_MC_ji, n_ops_left_j, OperationPool, S_k, S_j, 
                    JSet, OJSet, t, X_ijk, S_ij, C_ij, C_j, CT_k, T_cur, Tard_job,
                    NewJobList, MBList, PopSize, maxtime, re)
    return [
        method.GA
        , method.TS
        , method.LFOH
        , method.LAPH
        , method.LAP_LFO
        , method.LFOH_TS
        , method.LAPH_TS
        , method.LFOH_GA
        , method.LAPH_GA
        , method.CDR1
        , method.CDR2
        , method.CDR3
        , method.CDR4
        , method.CDR5
        , method.CDR6
        , method.RouteChange_RightShift
    ]


class Method:
    def __init__(self, J, I, K, p_ijk, h_ijk, d_j, n_j, 
                 MC_ji, n_MC_ji, n_ops_left_j, OperationPool, S_k, S_j, 
                 JSet, OJSet, t, X_ijk, S_ij, C_ij, C_j, CT_k, T_cur, Tard_job,
                 NewJobList, MBList, PopSize, maxtime, re):
        
        self.J     			= copy.deepcopy(J)
        self.I 	    		= copy.deepcopy(I)
        self.K  	   		= copy.deepcopy(K)
        self.p_ijk          = copy.deepcopy(p_ijk)
        self.h_ijk          = copy.deepcopy(h_ijk)
        self.d_j            = copy.deepcopy(d_j)
        self.n_j            = copy.deepcopy(n_j)
        self.MC_ji          = copy.deepcopy(MC_ji)
        self.n_MC_ji        = copy.deepcopy(n_MC_ji)
        self.n_ops_left_j   = copy.deepcopy(n_ops_left_j)
        self.OperationPool  = copy.deepcopy(OperationPool)
        self.S_k            = copy.deepcopy(S_k)
        self.S_j            = copy.deepcopy(S_j)
        self.JSet           = copy.deepcopy(JSet)
        self.OJSet          = copy.deepcopy(OJSet)
        self.X_ijk          = copy.deepcopy(X_ijk)
        self.S_ij           = copy.deepcopy(S_ij)
        self.C_ij           = copy.deepcopy(C_ij)
        self.C_j            = copy.deepcopy(C_j)
        self.CT_k           = copy.deepcopy(CT_k)
        self.T_cur          = copy.deepcopy(T_cur)
        self.Tard_job       = copy.deepcopy(Tard_job)
        self.t              = copy.deepcopy(t)
        self.NewJobList     = copy.deepcopy(NewJobList)
        self.MBList         = copy.deepcopy(MBList)
        self.PopSize        = copy.deepcopy(PopSize)
        self.maxtime        = copy.deepcopy(maxtime)
        self.re             = copy.deepcopy(re)


    # def exact(self):
    #     warnings.filterwarnings('ignore', 'Spaces are not permitted in the name')        
    #     StartTime = time()
    #     M = 10000

    #     # Create the LP problem
    #     problem = LpProblem("Flexible Job Shop Problem", LpMinimize)

    #     class TimeoutException(Exception):
    #         pass

    #     try:
    #         # Decision variables
    #         X_ijk= [[[LpVariable(f"X_{i}_{j}_{k}", cat='Binary') for k in range(self.K)] for j in range(self.J)] for i in range(self.I)]
    #         Y_ijab= [[[[LpVariable(f"Y_{i}_{j}_{a}_{b}", cat='Binary') for b in range(self.J)] for a in range(self.I)] for j in range(self.J)] for i in range(self.I)]
            
    #         print("after Y_ijab", time())
    #         if time() - StartTime > self.maxtime/5:
    #             raise TimeoutException("Time limit exceeded")
            
    #         S_ij= [[LpVariable(f"S_{i}_{j}", lowBound=0, cat='Integer') for j in range(self.J)] for i in range(self.I)]
    #         C_ij= [[LpVariable(f"C_{i}_{j}", lowBound=0, cat='Integer') for j in range(self.J)] for i in range(self.I)]
    #         C_j= [LpVariable(f"C_{j}", lowBound=0, cat='Integer') for j in range(self.J)]
    #         T_j= [LpVariable(f"T_{j}", lowBound=0, cat='Integer') for j in range(self.J)]  

    #         # Model
    #         for j in self.JSet:
    #             if time() - StartTime > self.maxtime/3:
    #                 raise TimeoutException("Time limit exceeded")
                
    #             for i in self.OJSet[j]:                    
    #                 problem += lpSum(X_ijk[i][j][k] for k in range(self.K)) == 1
    #                 for k in range(self.K):
    #                     problem += X_ijk[i][j][k] <= self.h_ijk[i, j, k]
    #                 problem += S_ij[i][j] >= lpSum(X_ijk[i][j][k]*self.S_k[k] for k in self.MC_ji[j][i])
    #                 problem += S_ij[i][j] >= self.S_j[j]

    #                 if i != int(self.n_j[j]) - 1:
    #                     problem += S_ij[i][j] + lpSum(X_ijk[i][j][k] * self.p_ijk[i, j, k] for k in self.MC_ji[j][i]) <= S_ij[i+1][j]

    #                 for b in self.JSet:                        
    #                     if j < b:
    #                         for a in self.OJSet[b]:
    #                             MC_common = set(self.MC_ji[j][i]).intersection(set(self.MC_ji[b][a]))
    #                             for k in MC_common:
    #                                 problem += S_ij[i][j] >= S_ij[a][b] + self.p_ijk[a, b, k] - M * (2 - X_ijk[i][j][k] - X_ijk[a][b][k] + Y_ijab[i][j][a][b])
    #                                 problem += S_ij[a][b] >= S_ij[i][j] + self.p_ijk[i, j, k] - M * (3 - X_ijk[i][j][k] - X_ijk[a][b][k] - Y_ijab[i][j][a][b])        

    #                 problem += C_ij[i][j] == S_ij[i][j] + lpSum(X_ijk[i][j][k] * self.p_ijk[i, j, k] for k in self.MC_ji[j][i])
    #             problem += C_j[j] == C_ij[int(self.n_j[j])-1][j]
    #             problem += T_j[j] >= C_j[j] - self.d_j[j]
            
    #         problem.setObjective(lpSum([T_j[j] for j in self.JSet]))
            
    #         if time() - StartTime > self.maxtime/3:
    #             raise TimeoutException("Time limit exceeded")
            
    #         solver = PULP_CBC_CMD(timeLimit=self.maxtime - (time() - StartTime))
    #         problem.solve(solver)

    #         print(LpStatus[problem.status])
    #         if LpStatus[problem.status] != "Optimal" or time() - StartTime > self.maxtime:
    #             print(time() - StartTime, self.maxtime)
    #             raise TimeoutException("Time limit exceeded")
    #         else:
    #             # Retrieve the objective value
    #             objective_value = value(problem.objective)

    #             X_values  = [[[value(X_ijk[i][j][k]) for k in range(self.K)] for j in range(self.J)] for i in range(self.I)]
    #             S_values  = [[value(S_ij[i][j]) for j in range(self.J)] for i in range(self.I)]
    #             C_values  = np.array([[value(C_ij[i][j]) for j in range(self.J)] for i in range(self.I)])

    #             X_values  = np.array(X_values)
    #             S_values  = np.array(S_values)
    #             C_values  = np.array(C_values)

    #     except TimeoutException:
    #         print("punishment", time())
    #         objective_value, X_values, S_values, C_values, Cj_values = self.CDR4()
    #         objective_value = 10**10  # Punishment

    #     return objective_value, X_values, S_values, C_values, Cj_values
   

    # def exact(self):
    #     import threading
    #     import warnings

    #     class TimeoutException(Exception):
    #         pass
        
    #     self.d_j = self.d_j.astype(int)
    #     self.n_j = self.n_j.astype(int)
    #     self.S_k = self.S_k.astype(int)
    #     self.S_j = self.S_j.astype(int)
    #     self.p_ijk = self.p_ijk.astype(int)

    #     warnings.filterwarnings('ignore', 'Spaces are not permitted in the name')
        
    #     StartTime = time()
        
    #     M = 10000
    #     timeout_event = threading.Event()
    #     problem = LpProblem("Flexible Job Shop Problem", LpMinimize)
    #     def interruptable_work(problem):
    #         nonlocal timeout_event
    #         global X_ijk, Y_ijab, S_ij, C_ij, C_j, T_j
    #         try:
    #             # Decision variables
    #             X_ijk = [[[LpVariable(f"X_{i}_{j}_{k}", cat='Binary') for k in range(self.K)] for j in range(self.J)] for i in range(self.I)]
    #             Y_ijab = [[[[LpVariable(f"Y_{i}_{j}_{a}_{b}", cat='Binary') for b in range(self.J)] for a in range(self.I)] for j in range(self.J)] for i in range(self.I)]
    #             if time() - StartTime > self.maxtime:
    #                 print(True)
    #                 timeout_event.set()
    #                 return
    #             S_ij = [[LpVariable(f"S_{i}_{j}", lowBound=0, cat='Integer') for j in range(self.J)] for i in range(self.I)]
    #             C_ij = [[LpVariable(f"C_{i}_{j}", lowBound=0, cat='Integer') for j in range(self.J)] for i in range(self.I)]
    #             C_j = [LpVariable(f"C_{j}", lowBound=0, cat='Integer') for j in range(self.J)]
    #             T_j = [LpVariable(f"T_{j}", lowBound=0, cat='Integer') for j in range(self.J)]

    #             # Model constraints
    #             for j in self.JSet:
    #                 if time() - StartTime > self.maxtime:
    #                     print(True)
    #                     timeout_event.set()
    #                     return
    #                 for i in self.OJSet[j]:                          
    #                     problem += lpSum(X_ijk[i][j][k] for k in range(self.K)) == 1
    #                     for k in range(self.K):
    #                         problem += X_ijk[i][j][k] <= self.h_ijk[i, j, k]
    #                     problem += -S_ij[i][j] + lpSum(X_ijk[i][j][k] * self.S_k[k] for k in self.MC_ji[j][i]) <= 0
    #                     problem += -S_ij[i][j] <= -self.S_j[j]

    #                     if i != self.n_j[j] - 1:
    #                         problem += S_ij[i][j] + lpSum(X_ijk[i][j][k] * self.p_ijk[i, j, k] for k in self.MC_ji[j][i]) - S_ij[i+1][j] <= 0

    #                     for b in self.JSet:
    #                         if j < b:
    #                             for a in self.OJSet[b]:
    #                                 MC_common = set(self.MC_ji[j][i]).intersection(set(self.MC_ji[b][a]))
    #                                 for k in MC_common:
    #                                     problem += - S_ij[i][j] + S_ij[a][b] - M * (2 - X_ijk[i][j][k] - X_ijk[a][b][k] + Y_ijab[i][j][a][b]) <=  -self.p_ijk[a, b, k]
    #                                     problem += - S_ij[a][b] + S_ij[i][j] - M * (3 - X_ijk[i][j][k] - X_ijk[a][b][k] - Y_ijab[i][j][a][b]) <=  -self.p_ijk[i, j, k] 
    #                     problem += C_ij[i][j] == S_ij[i][j] + lpSum(X_ijk[i][j][k] * self.p_ijk[i, j, k] for k in self.MC_ji[j][i])
    #                 problem += C_j[j] - C_ij[int(self.n_j[j]-1)][j] == 0
    #                 problem += C_j[j] - T_j[j] <= self.d_j[j]

    #         except Exception as e:
    #             timeout_event.set()
    #             raise e

    #     try:
    #         # Start the thread
    #         worker_thread = threading.Thread(target=interruptable_work(problem))
    #         worker_thread.start()

    #         # Wait for the worker thread with a timeout
    #         worker_thread.join(timeout=self.maxtime)
    #         if worker_thread.is_alive():
    #             timeout_event.set()
    #         if timeout_event.is_set():
    #             raise TimeoutException("Time limit exceeded for variable creation and constraints setting")

    #         # Set objective
    #         problem.setObjective(lpSum([T_j[j] for j in self.JSet]))

    #         # Calculate remaining time for solving
    #         remaining_time = self.maxtime - (time() - StartTime)
    #         if remaining_time <= 0:
    #             raise TimeoutException("Time limit exceeded before solving")

    #         # Start a timer for solving the problem
    #         solver_timer = threading.Timer(remaining_time, lambda: timeout_event.set())
    #         solver_timer.start()

    #         # Solve the problem
    #         solver = PULP_CBC_CMD(msg=False, options=['sec', str(remaining_time)])
    #         problem.solve(solver)

    #         # Cancel the solver timer if solving is completed in time
    #         solver_timer.cancel()

    #         print(LpStatus[problem.status])
    #         # Check the result
    #         if LpStatus[problem.status] != "Optimal" or time() - StartTime > self.maxtime:
    #             raise TimeoutException("Time limit exceeded or problem not optimal")

    #         # Retrieve the objective value
    #         objective_value = value(problem.objective)

    #         X_values = np.array([[[value(X_ijk[i][j][k]) for k in range(self.K)] for j in range(self.J)] for i in range(self.I)])
    #         S_values = np.array([[value(S_ij[i][j]) for j in range(self.J)] for i in range(self.I)])
    #         C_values = np.array([[value(C_ij[i][j]) for j in range(self.J)] for i in range(self.I)])

    #         Cj_values = np.max(C_values, axis=0)

    #     except TimeoutException as e:
    #         print(str(e))
    #         print("punishment")
    #         objective_value, X_values, S_values, C_values, Cj_values = self.CDR4()
    #         objective_value = 10**10  # Punishment
    #     except Exception as e:
    #         print("Error:", str(e))

    #     return objective_value, X_values, S_values, C_values, Cj_values

    def GA(self):
        StartTime                     = time()
        population, chromosome_len    = random_population(self.OperationPool, self.PopSize)
        GBest, X_ijk, S_ij, C_ij, C_j = GeneticAlgorithm(self.S_k, self.S_j, self.JSet, self.OJSet, 
                                                        self.J, self.I, self.K, 
                                                        self.p_ijk, self.h_ijk, self.d_j, self.n_j, self.n_ops_left_j, 
                                                        self.MC_ji, self.n_MC_ji, self.OperationPool,
                                                        self.PopSize, population, chromosome_len,
                                                        StartTime, self.maxtime)
        return GBest, X_ijk, S_ij, C_ij, C_j
    
    def TS(self):
        StartTime              = time()
        OA, MS, chromosome_len = encode_schedule(self.J, self.I, self.n_j, self.X_ijk, self.S_ij, 
                                                self.MC_ji, self.n_MC_ji, self.n_ops_left_j, self.t)
        
        if self.NewJobList:
            for job, deadline in self.NewJobList:
                chromosome_len += self.n_j[job]
                random_numbers  = [random.randint(0, 1) for _ in range(self.n_j[job])]
                if deadline == "urgent":
                    OA = [job]*self.n_j[job] + OA
                    MS = random_numbers + MS                
                else:
                    OA = OA + [job]*self.n_j[job]
                    MS = MS + random_numbers

        GBest, X_ijk, S_ij, C_ij, C_j = TabuSearch (self.S_k, self.S_j, self.JSet, self.J, self.I, self.K, 
                                                    self.p_ijk, self.d_j, self.n_j, self.n_ops_left_j, self.MC_ji, self.n_MC_ji, 
                                                    OA, MS, chromosome_len, StartTime, self.maxtime)
        return GBest, X_ijk, S_ij, C_ij, C_j
    

    def LFOH (self):
        ORSet            = []
        ORDict           = {} # key: ready time, value: [i, j]
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        S_k              = copy.deepcopy(self.S_k)
        S_j              = copy.deepcopy(self.S_j)
        dummy_JSet       = copy.deepcopy(self.JSet)
        dummy_OJSet      = copy.deepcopy(self.OJSet)

        Reschedule_completion = False
        # Start
        for j in dummy_JSet:
            ready_time = copy.deepcopy(S_j[j])
            operation  = copy.deepcopy([dummy_OJSet[j][0], j])
            if ready_time not in ORDict:
                ORDict[ready_time] = []
            ORDict[ready_time].append(operation)

        ready_time       = min(ORDict)
        ORSet            = copy.deepcopy(ORDict[ready_time])
        sorted_operation = sorted(ORSet, key=lambda x: (self.n_MC_ji[x[1]][x[0]], self.d_j[x[1]], np.mean(self.p_ijk[x[0], x[1]])))


        while Reschedule_completion == False:
            for i, j in sorted_operation:
                # Select machine
                if self.n_MC_ji[j][i] == 1:
                    k = self.MC_ji[j][i]
                    X_ijk[i, j, k] = 1
                else:
                    available           = np.maximum(ready_time, S_k)
                    mask                = self.h_ijk[i, j, :] == 1
                    filtered_available  = available[mask]
                    min_value           = np.min(filtered_available)
                    min_indices         = np.where(filtered_available == min_value)[0]
                    filtered_index      = np.random.choice(min_indices)
                    k                   = np.arange(len(available))[mask][filtered_index]
                    X_ijk[i, j, k]  = 1  
                # Calculate Start time, Completion time, and set new S_k
                S_ij[i, j]          = max(ready_time, S_k[k])
                C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]
                S_k[k]              = copy.deepcopy(C_ij[i, j])
                # Adjust the set and append the dictionary (if any)
                dummy_OJSet[j].remove(i)
                if len(dummy_OJSet[j]) != 0:
                    new_ready_time = copy.deepcopy(C_ij[i, j])
                    operation      = copy.deepcopy([dummy_OJSet[j][0], j])
                    if new_ready_time not in ORDict:
                        ORDict[new_ready_time] = []
                    ORDict[new_ready_time].append(operation)
                else:
                    dummy_JSet.remove(j)
            # Process the dictionary, check flag
            ORDict.pop(ready_time)
            if ORDict:
                ready_time       = min(ORDict)
                ORSet            = copy.deepcopy(ORDict[ready_time])
                sorted_operation = sorted(ORSet, key=lambda x: (self.n_MC_ji[x[1]][x[0]], self.d_j[x[1]], np.mean(self.p_ijk[x[0], x[1]])))
            else: Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j


    def LAPH(self):
        ORSet            = []
        ORDict           = {} # key: ready time, value: [i, j]
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        S_k              = copy.deepcopy(self.S_k)
        dummy_JSet       = copy.deepcopy(self.JSet)
        dummy_OJSet      = copy.deepcopy(self.OJSet)

        Reschedule_completion = False
        # Start
        for j in dummy_JSet:
            ready_time = copy.deepcopy(self.S_j[j])
            operation  = copy.deepcopy([dummy_OJSet[j][0], j])
            if ready_time not in ORDict:
                ORDict[ready_time] = []
            ORDict[ready_time].append(operation)

        ready_time = min(ORDict)
        ORSet      = copy.deepcopy(ORDict[ready_time])

        while Reschedule_completion == False:
            # Initialize cost matrix with a high value
            num_operations = len(ORSet)
            cost_matrix    = np.full((num_operations, self.K), 999**3)

            # Populate the cost matrix based on constraints
            for op_idx, [i, j] in enumerate(ORSet):
                for k in range(self.K):
                    if self.h_ijk[i, j, k] == 1:
                        cost_matrix[op_idx, k] = S_k[k] + self.p_ijk[i, j, k]

            # Solve the assignment problem using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Output the assignment results
            assignments = [(ORSet[row], machine) for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999**3]
            new_availabletime = []
            ORDict.pop(ready_time)
            for (i, j), k in assignments:
                X_ijk[i, j, k] = 1
                S_ij[i, j]     = max(ready_time, S_k[k])
                C_ij[i, j]     = S_ij[i, j] + self.p_ijk[i, j, k]  
                S_k[k]         = copy.deepcopy(C_ij[i, j])
                new_availabletime.append(S_k[k])
                dummy_OJSet[j].remove(i) 
                
                if len(dummy_OJSet[j]) != 0:
                    new_ready_time = copy.deepcopy(C_ij[i, j])
                    operation      = copy.deepcopy([dummy_OJSet[j][0], j])
                    if new_ready_time not in ORDict:
                        ORDict[new_ready_time] = []
                    ORDict[new_ready_time].append(operation)
                else:
                    dummy_JSet.remove(j)

            assigned_operations = [ORSet[row] for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999**3]
            unassigned_operations = [op for op in ORSet if op not in assigned_operations]

            if unassigned_operations:
                new_ready_time = min(new_availabletime) ####
                if new_ready_time in ORDict:
                    ORDict[new_ready_time].extend(unassigned_operations)
                else:
                    ORDict[new_ready_time] = unassigned_operations
                    
            # Process the dictionary, check flag
            if ORDict:
                ready_time   = min(ORDict)
                ORSet        = copy.deepcopy(ORDict[ready_time])
               
            else: Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j

    def LAP_LFO(self):
        ORSet            = []
        ORDict           = {} # key: ready time, value: [i, j]
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        S_j              = copy.deepcopy(self.S_j)
        S_k              = copy.deepcopy(self.S_k)
        dummy_JSet       = copy.deepcopy(self.JSet)
        dummy_OJSet      = copy.deepcopy(self.OJSet)

        Reschedule_completion = False
 
        # Start
        for j in dummy_JSet:
            ready_time = copy.deepcopy(S_j[j])
            operation  = copy.deepcopy([dummy_OJSet[j][0], j])
            if ready_time not in ORDict:
                ORDict[ready_time] = []
            ORDict[ready_time].append(operation)

        ready_time = min(ORDict)
        ORSet      = copy.deepcopy(ORDict[ready_time])

        while Reschedule_completion == False:
            # Initialize cost matrix with a high value
            num_operations = len(ORSet)
            cost_matrix    = np.full((num_operations, self.K), 999**3)

            # Populate the cost matrix based on constraints
            for op_idx, (i, j) in enumerate(ORSet):
                for k in range(self.K):
                    if self.h_ijk[i, j, k]  == 1:
                        cost_matrix[op_idx, k] = S_k[k] + self.p_ijk[i, j, k]  

            # Solve the assignment problem using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Output the assignment results
            assignments = [(ORSet[row], machine) for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999**3]
            ORDict.pop(ready_time)
            for (i, j), k in assignments:
                X_ijk[i, j, k]      = 1
                S_ij[i, j]          = max(ready_time, S_k[k])
                C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
                S_k[k]              = copy.deepcopy(C_ij[i, j])
                dummy_OJSet[j].remove(i) 
                if len(dummy_OJSet[j]) != 0:
                    new_ready_time = copy.deepcopy(C_ij[i][j])
                    operation      = copy.deepcopy([dummy_OJSet[j][0], j])
                    if new_ready_time not in ORDict:
                        ORDict[new_ready_time] = []
                    ORDict[new_ready_time].append(operation)
                else:
                    dummy_JSet.remove(j)

            assigned_operations = [ORSet[row] for row, machine in zip(row_ind, col_ind) if cost_matrix[row, machine] < 999**3]
            unassigned_operations = [op for op in ORSet if op not in assigned_operations]

            if unassigned_operations:
                sorted_operation = sorted(unassigned_operations, key=lambda x: (self.n_MC_ji[x[1]][x[0]], self.d_j[x[1]], np.mean(self.p_ijk[x[0], x[1]])))
                for i, j in sorted_operation:
                    # Select machine
                    if self.n_MC_ji[j][i] == 1:
                        k = self.MC_ji[j][i]
                        X_ijk[i, j, k]  = 1
                    else:
                        available           = np.maximum(ready_time, S_k)
                        mask                = self.h_ijk[i, j, :] == 1
                        filtered_available  = available[mask]
                        min_value           = np.min(filtered_available)
                        min_indices         = np.where(filtered_available == min_value)[0]
                        filtered_index      = np.random.choice(min_indices)
                        k                   = np.arange(len(available))[mask][filtered_index]
                        X_ijk[i, j, k]  = 1  
                    # Calculate Start time, Completion time, and set new S_k
                    S_ij[i, j]          = max(ready_time, S_k[k])
                    C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
                    S_k[k]              = copy.deepcopy(C_ij[i, j])
                    # Adjust the set and append the dictionary (if any)
                    dummy_OJSet[j].remove(i)
                    if len(dummy_OJSet[j]) != 0:
                        new_ready_time = copy.deepcopy(C_ij[i, j])
                        operation      = copy.deepcopy([dummy_OJSet[j][0], j])
                        if new_ready_time not in ORDict:
                            ORDict[new_ready_time] = []
                        ORDict[new_ready_time].append(operation)
                    else:
                        dummy_JSet.remove(j)

            # Process the dictionary, check flag
            if ORDict:
                ready_time   = min(ORDict)
                ORSet        = copy.deepcopy(ORDict[ready_time])
            else: Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j
    
    def LFOH_TS(self):
        StartTime = time()
        GBest, X_ijk, S_ij, C_ij, C_j = self.LFOH()
        OA, MS, chromosome_len        = encode_schedule(self.J, self.I, self.n_j, 
                                                        X_ijk, S_ij, self.MC_ji, self.n_MC_ji, self.n_ops_left_j,
                                                        self.t)
        GBest, X_ijk, S_ij, C_ij, C_j = TabuSearch (self.S_k, self.S_j, self.JSet, self.J, self.I, self.K, 
                                                    self.p_ijk, self.d_j, self.n_j, self.n_ops_left_j, self.MC_ji, self.n_MC_ji, 
                                                    OA, MS, chromosome_len, StartTime, self.maxtime)
        return GBest, X_ijk, S_ij, C_ij, C_j
    
    def LAPH_TS(self):
        StartTime = time()
        GBest, X_ijk, S_ij, C_ij, C_j = self.LAPH()

        OA, MS, chromosome_len        = encode_schedule(self.J, self.I, self.n_j, 
                                                        X_ijk, S_ij, self.MC_ji, self.n_MC_ji, self.n_ops_left_j,
                                                        self.t)
        GBest, X_ijk, S_ij, C_ij, C_j = TabuSearch (self.S_k, self.S_j, self.JSet, self.J, self.I, self.K, 
                                                    self.p_ijk, self.d_j, self.n_j, self.n_ops_left_j, self.MC_ji, self.n_MC_ji, 
                                                    OA, MS, chromosome_len, StartTime, self.maxtime)

        return GBest, X_ijk, S_ij, C_ij, C_j
    
    def LFOH_GA(self):
        StartTime = time()
        GBest, X_ijk, S_ij, C_ij, C_j = self.LFOH()
        OA, MS, chromosome_len        = encode_schedule(self.J, self.I, self.n_j, 
                                                        X_ijk, S_ij, self.MC_ji, self.n_MC_ji, self.n_ops_left_j,
                                                        self.t)
        current_solution              = (OA, MS)
        population                    = generate_neighborhood(current_solution, self.PopSize, chromosome_len)
        GBest, X_ijk, S_ij, C_ij, C_j = GeneticAlgorithm(self.S_k, self.S_j, self.JSet, self.OJSet, 
                                                    self.J, self.I, self.K, 
                                                    self.p_ijk, self.h_ijk, self.d_j, self.n_j, self.n_ops_left_j, 
                                                    self.MC_ji, self.n_MC_ji, self.OperationPool,
                                                    self.PopSize, population, chromosome_len,
                                                    StartTime, self.maxtime)
        return GBest, X_ijk, S_ij, C_ij, C_j
    
    def LAPH_GA(self):
        StartTime = time()
        GBest, X_ijk, S_ij, C_ij, C_j = self.LAPH()
        OA, MS, chromosome_len        = encode_schedule(self.J, self.I, self.n_j, 
                                                        X_ijk, S_ij, self.MC_ji, self.n_MC_ji, self.n_ops_left_j,
                                                        self.t)
        current_solution              = (OA, MS)
        population                    = generate_neighborhood(current_solution, self.PopSize, chromosome_len)
        GBest, X_ijk, S_ij, C_ij, C_j = GeneticAlgorithm(self.S_k, self.S_j, self.JSet, self.OJSet, 
                                                    self.J, self.I, self.K, 
                                                    self.p_ijk, self.h_ijk, self.d_j, self.n_j, self.n_ops_left_j, 
                                                    self.MC_ji, self.n_MC_ji, self.OperationPool,
                                                    self.PopSize, population, chromosome_len,
                                                    StartTime, self.maxtime)
        return GBest, X_ijk, S_ij, C_ij, C_j
    
    def CDR1(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        dummy_JSet       = copy.deepcopy(self.JSet)
        Reschedule_completion = False

        while Reschedule_completion == False:
            # Check if there is Tard_job
            if self.Tard_job:
                OP = self.n_j - self.n_ops_left_j
                estimated_tardiness = np.full(self.J, -np.inf)
                for j in self.Tard_job:
                    estimated_tardiness[j] = self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
                j = np.argmax(estimated_tardiness)
            else:
                average_slack = np.full(self.J, np.inf)
                average_slack[dummy_JSet] = (self.d_j[dummy_JSet] - self.T_cur)/self.n_ops_left_j[dummy_JSet]
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
            X_ijk[i, j, k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i, j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
            self.S_k[k]         = copy.deepcopy(C_ij[i, j])
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j] = copy.deepcopy(C_ij[i, j])

            if not dummy_JSet:
                Reschedule_completion = True
            else:
                self.T_cur = np.mean(self.S_k)
                self.Tard_job = [j for j in dummy_JSet if self.d_j[j] < self.T_cur]

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j

    def CDR2(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        dummy_JSet       = copy.deepcopy(self.JSet)
        Reschedule_completion = False
        while Reschedule_completion == False:
            OP = self.n_j - self.n_ops_left_j

            # Check if there is Tard_job
            if self.Tard_job:
                estimated_tardiness = np.full(self.J, -np.inf)
                for j in self.Tard_job:
                    estimated_tardiness[j] = self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
                j = np.argmax(estimated_tardiness)
            else:
                critical_ratio = np.full(self.J, np.inf)
                for j in dummy_JSet:
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
            X_ijk[i, j, k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i, j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
            self.S_k[k]         = copy.deepcopy(C_ij[i, j])
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j] = copy.deepcopy(C_ij[i, j])

            if not dummy_JSet:
                Reschedule_completion = True
            else:
                self.T_cur = np.mean(self.S_k)
                self.Tard_job = [j for j in dummy_JSet if self.d_j[j] < self.T_cur]

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j
    
    def CDR3(self):
        for j in self.JSet:
            self.X_ijk[self.OJSet[j], j, :] = 0

        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        dummy_JSet       = copy.deepcopy(self.JSet)
        Reschedule_completion = False
        while Reschedule_completion == False:
            OP = self.n_j - self.n_ops_left_j

            # Select job with largest estimated tardiness
            estimated_tardiness = np.full(self.J, -np.inf)
            for j in dummy_JSet:
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
            self.X_ijk[i, j, k] = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i, j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
            self.S_k[k]         = copy.deepcopy(C_ij[i, j])

            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j] = copy.deepcopy(C_ij[i, j])

            if not dummy_JSet:
                Reschedule_completion = True
            else:
                self.T_cur = np.mean(self.S_k)

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, self.X_ijk, S_ij, C_ij, C_j

    def CDR4(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        dummy_JSet       = copy.deepcopy(self.JSet)

        Reschedule_completion = False

        while Reschedule_completion == False:
            # Randomly select a job
            j                   = random.choice(dummy_JSet)
            i                   = self.n_j[j] - self.n_ops_left_j[j]
            # Find earliest available machine
            available           = np.maximum(self.S_j[j], self.S_k)
            mask                = self.h_ijk[i, j, :] == 1
            filtered_available  = available[mask]
            min_value           = np.min(filtered_available)
            min_indices         = np.where(filtered_available == min_value)[0]
            filtered_index      = np.random.choice(min_indices)
            k                   = np.arange(len(available))[mask][filtered_index]
            X_ijk[i, j, k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i, j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
            self.S_k[k]         = copy.deepcopy(C_ij[i, j])
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j]      = copy.deepcopy(C_ij[i, j])
            if not dummy_JSet:
                Reschedule_completion = True

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)
        return GBest, X_ijk, S_ij, C_ij, C_j

    
    def CDR5(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        dummy_JSet       = copy.deepcopy(self.JSet)
        Reschedule_completion = False
        while Reschedule_completion == False:
            OP = self.n_j - self.n_ops_left_j

            # Check if there is Tard_job
            if self.Tard_job:
                InversedCompletionRate_tardiness = np.full(self.J, -np.inf)
                for j in self.Tard_job:
                    InversedCompletionRate_tardiness[j] = 0 if OP[j] == 0 else self.n_j[j]/OP[j] * (self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j])
                j = np.argmax(InversedCompletionRate_tardiness)
            else:
                CompletionRate_slack = np.full(self.J, np.inf)
                for j in dummy_JSet:
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
            X_ijk[i, j, k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i, j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
            self.S_k[k]         = copy.deepcopy(C_ij[i, j])
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j] = copy.deepcopy(C_ij[i, j])

            if not dummy_JSet:
                Reschedule_completion = True
            else:
                self.T_cur = np.mean(self.S_k)
                self.Tard_job = [j for j in dummy_JSet if self.d_j[j] < self.T_cur]

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j
    

    def CDR6(self):
        X_ijk            = np.zeros((self.I, self.J, self.K))
        S_ij             = np.zeros((self.I, self.J))
        C_ij             = np.zeros((self.I, self.J))
        p_mean           = np.mean(self.p_ijk*self.h_ijk, axis= 2)
        dummy_JSet       = copy.deepcopy(self.JSet)
        Reschedule_completion = False

        while Reschedule_completion == False:
            OP = self.n_j - self.n_ops_left_j

            estimated_tardiness = np.full(self.J, -np.inf)
            for j in dummy_JSet:
                estimated_tardiness[j] = self.T_cur + np.sum(p_mean[OP[j]:self.n_j[j], j]) - self.d_j[j]
            j = np.argmax(estimated_tardiness)
            i                   = self.n_j[j] - self.n_ops_left_j[j]
            # Find earliest available machine
            available           = np.maximum(self.S_j[j], self.S_k)
            mask                = self.h_ijk[i, j, :] == 1
            filtered_available  = available[mask]
            min_value           = np.min(filtered_available)
            min_indices         = np.where(filtered_available == min_value)[0]
            filtered_index      = np.random.choice(min_indices)
            k                   = np.arange(len(available))[mask][filtered_index]
            X_ijk[i, j, k]      = 1  
            # Calculate Start time, Completion time, and set new S_k
            S_ij[i, j]          = max(self.S_j[j], self.S_k[k])
            C_ij[i, j]          = S_ij[i, j] + self.p_ijk[i, j, k]  
            self.S_k[k]         = copy.deepcopy(C_ij[i, j])
            
            # Adjust the set
            self.OJSet[j].remove(i)
            self.n_ops_left_j[j] -= 1
            if len(self.OJSet[j]) == 0:
                dummy_JSet.remove(j)
            else:
                self.S_j[j] = copy.deepcopy(C_ij[i, j])

            if not dummy_JSet:
                Reschedule_completion = True
            else:
                self.T_cur = np.mean(self.S_k)

        # Calculate GBest
        GBest, C_j = evaluate_LocalCost(self.d_j, C_ij, self.JSet)

        return GBest, X_ijk, S_ij, C_ij, C_j
    

    def RouteChange_RightShift(self):
        Job_seq = copy.deepcopy(self.OJSet)
        Mch_seq = find_Mch_seq(self.K, self.X_ijk, self.C_ij, self.t)

        affected_Oij = {}
        for k in self.MBList:
            X_mask =  self.X_ijk.astype(bool)
            self.re[k]  =  self.t
            """Find affected operation"""
            # Use the boolean mask to find the indices where overlap occurs
            overlap_mask = np.logical_or(
                np.logical_and(self.S_ij >= self.t              , self.C_ij <= self.t + self.re[k]),       # in
                np.logical_and(self.S_ij <= self.t              , self.C_ij >  self.t             ),       # left
                np.logical_and(self.S_ij <  self.t + self.re[k] , self.C_ij >= self.t + self.re[k]))       # right
            indices = np.argwhere(X_mask[:, :, k] & overlap_mask[:, :])
            # Append the indices to the affected_Oij dictionary
            if len(indices) > 0:
                affected_Oij[k] = indices.tolist()

        for breakdown_MC in affected_Oij:
            for Oij in reversed(affected_Oij[breakdown_MC]):
                X_mask    =  self.X_ijk.astype(bool)
                S_ij_mask =  self.S_ij >= self.t

                indirect_interval    = []
                indirect_RouteChange = []
                cost_Modified        = []

                direct_RouteChange   = []
                i = Oij[0]
                j = Oij[1]

                ## Find start time of successor operation of Oij
                if i < int(self.n_j[j]-1):
                    S_Oij_suc = copy.deepcopy(self.S_ij[i+1][j])
                else:
                    S_Oij_suc = np.inf
                ## Find start time of soonest operation on the machine k'
                for k in self.MC_ji[j][i]:
                    if k != breakdown_MC: 
                        combined_mask = X_mask[:,:, k] & S_ij_mask
                        S_k_suc_filtered = self.S_ij[combined_mask]
                        if np.any(S_k_suc_filtered):
                            S_k_suc = np.min(S_k_suc_filtered)
                        else:
                            S_k_suc = np.inf
                        ## Check if satisfy equation
                        interval = min(S_Oij_suc, S_k_suc) - self.S_k[k]
                        
                        if self.p_ijk[i, j, k] < interval:
                            cost = self.S_k[k] + self.p_ijk[i, j, k]  
                            direct_RouteChange  .append(k)
                            cost_Modified       .append(cost)
                        else:
                            indirect_RouteChange.append(k)
                            indirect_interval   .append(interval)

                ### If satisfied
                if direct_RouteChange:
                    min_cost =  min(cost_Modified)
                    k_index  =  cost_Modified.index(min_cost)
                    new_k    =  direct_RouteChange[k_index]
                    self.X_ijk[i, j, new_k] = 1
                    self.X_ijk[i, j, breakdown_MC] = 0

                    self.S_ij[i, j] = self.S_k[new_k]
                    self.C_ij[i, j] = self.S_ij[i, j] + self.p_ijk[i, j, new_k]

                ### If not
                else:
                    X_modified       = []
                    S_modified       = []
                    C_modified       = []
                    Mch_seq_modified = []
                    id_ope_onMCh     = Mch_seq[breakdown_MC].index(Oij)

                    X_RightShift, S_RightShift, C_RightShift = RightShift(breakdown_MC, id_ope_onMCh, self.S_k[breakdown_MC], Job_seq, Mch_seq, self.X_ijk, self.S_ij, self.C_ij, self.p_ijk, self.n_j)
                    cost_RightShift, redundant = evaluate_LocalCost(self.d_j, C_RightShift, self.JSet)

                    X_modified      .append(X_RightShift)
                    S_modified      .append(S_RightShift)
                    C_modified      .append(C_RightShift)
                    cost_Modified   .append(cost_RightShift)
                    Mch_seq_modified.append(Mch_seq)

                    ## Route-Change
                    max_interval =  max(indirect_interval)
                    k_index      =  indirect_interval.index(max_interval)
                    new_k        =  indirect_RouteChange[k_index]

                    dummy_X_ijk  =  copy.deepcopy(self.X_ijk)
                    dummy_X_ijk[i, j, new_k] = 1
                    dummy_X_ijk[i, j, breakdown_MC] = 0

                    ## Change Mch seq
                    dummy_Mch_seq = copy.deepcopy(Mch_seq)
                    dummy_Mch_seq[breakdown_MC].remove(Oij)
                    dummy_Mch_seq[new_k]       .insert(0, Oij)

                    ## Right-Shift
                    X_RCRS, S_RCRS, C_RCRS = RightShift(breakdown_MC, id_ope_onMCh, self.S_k[new_k], Job_seq, Mch_seq, dummy_X_ijk, self.S_ij, self.C_ij, self.p_ijk, self.n_j)
                    cost_RCRS, redundant = evaluate_LocalCost(self.d_j, C_RCRS, self.JSet)

                    X_modified      .append(X_RCRS)
                    S_modified      .append(S_RCRS)
                    C_modified      .append(C_RCRS)
                    cost_Modified   .append(cost_RCRS)
                    Mch_seq_modified.append(dummy_Mch_seq)
  
                    for new_k in self.MC_ji[j][i]:
                        # machine breakdown
                        X_mask_new_k   = self.X_ijk[:, :, new_k].astype(bool)

                        if not X_mask_new_k.any():
                            C_LastPosition = self.S_k[new_k]
                            S_Oij = self.S_j[j]
                        else:
                            C_LastPosition = self.C_ij[X_mask_new_k].max()
                            S_Oij = copy.deepcopy(C_LastPosition)
                        
                        if i < int(self.n_j[j]) - 1:
                            S_Oij_suc = copy.deepcopy(self.S_ij[i+1][j])
                        else:
                            S_Oij_suc = np.inf
                        
                        C_Oij = S_Oij + self.p_ijk[i, j, new_k]

                        dummy_X_ijk   = self.X_ijk.copy()
                        dummy_S_ij    = self.S_ij .copy()
                        dummy_C_ij    = self.C_ij .copy()
                        dummy_Mch_seq = copy.deepcopy(Mch_seq)

                        if C_Oij <= S_Oij_suc:
                            dummy_X_ijk[i, j, breakdown_MC] = 0
                            dummy_X_ijk[i, j, new_k] = 1
                            dummy_S_ij[i, j] = copy.deepcopy(S_Oij)
                            dummy_C_ij[i, j] = copy.deepcopy(C_Oij)
                            cost_LastPos, redundant = evaluate_LocalCost(self.d_j, dummy_C_ij, self.JSet)
                            
                            dummy_Mch_seq[breakdown_MC].remove(Oij)
                            dummy_Mch_seq[new_k]       .append(Oij)
                        else:
                            cost_LastPos = np.inf
                        
                        X_modified      .append(dummy_X_ijk)
                        S_modified      .append(dummy_S_ij)
                        C_modified      .append(dummy_C_ij)
                        cost_Modified   .append(cost_LastPos)
                        Mch_seq_modified.append(dummy_Mch_seq)

                    min_cost      =  min(cost_Modified)
                    sched_id      =  cost_Modified.index(min_cost)
                    self.X_ijk    =  X_modified      [sched_id]
                    self.S_ij     =  S_modified      [sched_id]
                    self.C_ij     =  C_modified      [sched_id]
                    Mch_seq       =  Mch_seq_modified[sched_id]

        if self.NewJobList:
            for j, deadline in self.NewJobList:
                for i in range(self.n_j[j]):
                    available           = np.maximum(self.S_j[j], self.S_k)
                    mask                = self.h_ijk[i, j, :] == 1
                    filtered_available  = available[mask]
                    
                    min_value           = np.min(filtered_available)
                    min_indices         = np.where(filtered_available == min_value)[0]
                    filtered_index      = np.random.choice(min_indices)
                    k                   = np.arange(len(available))[mask][filtered_index]
                    self.X_ijk[i, j, k] = 1  
                    # Calculate Start time, Completion time, and set new S_k
                    self.S_ij[i, j]     = max(self.S_j[j], self.S_k[k])
                    self.C_ij[i, j]     = self.S_ij[i, j] + self.p_ijk[i, j, k]  
                    self.S_k[k]         = copy.deepcopy(self.C_ij[i, j])
                    self.S_j[j]         = copy.deepcopy(self.C_ij[i, j])

        GBest, C_j = evaluate_LocalCost(self.d_j, self.C_ij, self.JSet)

        return GBest, self.X_ijk, self.S_ij, self.C_ij, C_j