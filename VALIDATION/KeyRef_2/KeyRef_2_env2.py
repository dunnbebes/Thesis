import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import copy
import random
from gymnasium import spaces
from KeyRef_2.KeyRef_2_action_space   import Method
from util.util_load          import read_txt
from util.util_reschedule    import generate_JA_event, change_dataset
from util.util_action 		 import find_Mch_seq, RightShift


def random_events(t, K, X_ijk, S_ij, C_ij, S_j, JSet, MB_event, S_k, UsedMachine):
	foundMB  = False
	MBList   = []
	events   = {}
	re       = np.zeros((K)) 
	if JSet:
		new_time = np.min(S_j[JSet])
	
	# have_event      = False
    # for job, deadline, description in JA_event:
    #     have_event  = True
    #     time_occur  = copy.deepcopy(C_j[job])
    #     if time_occur not in events:
    #         events[time_occur] = []
    #     events[time_occur].append(("JA", job, deadline, description))
	
	for k in range(K):
		if MB_event[k]:
			# have_event  = True
			time_occur, repair, description = MB_event[k][0]
			if time_occur not in events:
				events[time_occur] = []
			events[time_occur].append(("MB", k, repair, description))
    
    
    # if have_event == True:
    #     found         = None
    #     while found == None:
	if events:
		affected_Oij    = {}
		event_t         = copy.deepcopy(int(min(events.keys())))
		triggered_event = copy.deepcopy(events[event_t])
		# need_modify     = 0
		for uncertain_type, k, repair_time, description in triggered_event:
			if uncertain_type == "MB":
				mask = X_ijk[:, :, k] == 1  # Boolean array where True indicates assigned to machine k
				start_times = S_ij[mask]
				if start_times.size > 0:
					max_start_time = np.max(start_times)

					if event_t <= t: 	
						new_time = max(max_start_time, event_t)
					else: 	
						new_time = copy.deepcopy(event_t)
					
					X_mask   =  X_ijk.astype(bool)
					re[k]    =  repair_time
					
					"""Find affected operation"""
					# Use the boolean mask to find the indices where overlap occurs
					overlap_mask = np.logical_or(
						np.logical_and(S_ij >= new_time        , C_ij <= new_time + re[k]),       # in
						np.logical_and(S_ij <= new_time        , C_ij >  new_time        ),       # left
						np.logical_and(S_ij <  new_time + re[k], C_ij >= new_time + re[k]))       # right
					indices = np.argwhere(X_mask[:, :, k] & overlap_mask[:, :])
					# Append the indices to the affected_Oij dictionary
					if len(indices) > 0 and new_time != 0:
						# adjusted_indices = [[i, j] for i, j in indices] # Due to segmentize the operation
						affected_Oij[k] = indices.tolist()
						MBList.append(k)
						foundMB = True
					
				# if k not in affected_Oij:
				# 	event = (uncertain_type, k, repair_time, description)
				# 	# Remove current time
				# 	events[new_t].remove(event)
				# 	if len (events[new_t]) == 0:
				# 		events.pop(new_t)
				# 	# Adjust to new time (shift_time)
				# 	mask          = np.logical_and(S_ij > new_time, X_ijk[:, :, k] == 1)
				# 	filtered_S_ij = S_ij[mask]

				# 	if filtered_S_ij.size > 0:
				# 		shift_time    = np.min(filtered_S_ij)
				# 		if shift_time not in events:
				# 			events[shift_time] = []
				# 		events[shift_time].append(event)

			# 		need_modify += 1
                        
            #     if need_modify == 0:
            #         found = True
            # else:
            #     break

    #     if events: # After While loop, If have events
    #         re = np.zeros((K))
    #         for uncertain_type, partID, time_event, description in triggered_event:
    #             if uncertain_type == "JA":
    #                 JA_event = [JA for JA in JA_event if JA[0] != partID]
    #             else:
    #                 if partID not in MB_record:
    #                     MB_record[partID] = []
    #                 record = (new_time, new_time+re[partID])
    #                 MB_record[partID].append(record)
    #                 MB_event[partID].pop(0)
    #                 re[partID] = copy.deepcopy(time_event)
    #         # for key, value in events.items():
    #         #     if key < new_time:
    #         #         new_key = new_time + 60
    #         #         if new_key in new_data:
    #         #             events[new_key].extend(value)
    #         #         else:
    #         #             events[new_key] = value
    #         if not triggered_event:
    #             new_time = np.max(C_j)
    #             triggered_event = None  
    #             re = np.zeros((K))   
    #     else:
    #         new_time = np.max(C_j)
    #         triggered_event = None 
    #         re = np.zeros((K))

    # else:
    #     new_time = np.max(C_j)
    #     triggered_event = None 
    #     re = np.zeros((K)) 
    
    # if new_time > np.max(C_j):
    #     new_time = np.max(C_j)
    #     triggered_event = None
    #     re = np.zeros((K))  
	
	if foundMB:
		T = np.min(S_k[UsedMachine])
		if JSet and T < new_time:
			t = np.min(S_j[JSet])
			
			MBList = []    
		else:
			for k in MBList:
				MB_event[k].pop(0)
			t = copy.deepcopy(new_time)

	else:
		if JSet:
			t = np.min(S_j[JSet])

	return MB_event, MBList, t, re
			# JA_event, 
			# MB_event, 
			# new_time, 
			# triggered_event, 
			# re, 
			# MB_record
			

class KeyRef2_JA_and_MB_env(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self, K, planning_horizon, ReworkProbability, scenarios):
		super(KeyRef2_JA_and_MB_env, self).__init__()

		self.K              	= copy.deepcopy(K)
		self.planning_horizon   = copy.deepcopy(planning_horizon)
		self.ReworkProbability	= copy.deepcopy(ReworkProbability)
		self.scenarios          = copy.deepcopy(scenarios)
		self.method_list 		= ["CDR1", "CDR2", "CDR3", "CDR4", "CDR5", "CDR6"]
		self.CaseList 			= ['_fixed_instance'] + [case+1 for case in range(1, 48)]
		self.action_space 		= spaces.Discrete(6)
		self.observation_space  = spaces.Box(low=0, high=2,
											 shape=(7,), dtype=np.float32)

	def seed(self, seed=None):
		random.seed(seed)
		np.random.seed(seed)

	def load_scenario(self, scenario_id):
		self.current_scenario 	= self.scenarios[scenario_id]
		self.JA_event 	        = self.current_scenario.JA_event
		self.MB_event 			= self.current_scenario.MB_event

	def calc_observation (self):
		# Find utilization of each machine
		S_mask 		 = self.S_ij < self.t
		S_expand 	 = S_mask[:, :, np.newaxis]
		X_filtered 	 = self.X_ijk * S_expand
		Usage     	 = np.sum(X_filtered * self.p_ijk, axis = (0, 1))
		mask      	 = self.S_k != 0
		filtered_U_k = Usage[mask] / self.S_k[mask]
		
		# Find completion rate of each job
		CR_j = 1 - self.n_ops_left_j/self.n_j

		# Calculate estimated tardiness rate
		Ne_tard = 0
		Ne_left = 0
		for j in range(self.J):
			if self.n_ops_left_j[j] > 0:
				Ne_left += self.n_ops_left_j[j]
				T_left = 0
				for i in range(self.n_j[j] - self.n_ops_left_j[j], self.n_j[j]): 
					t_mean_ij = np.sum(self.p_ijk[i, j]*self.h_ijk[i,j])/np.maximum(np.sum(self.h_ijk[i,j]),1)
					T_left   += t_mean_ij
					if self.T_cur[j] + T_left > self.d_j[j]:
						Ne_tard += (self.n_j[j] - i) 
						break
		# Calculate actual tardiness rate 
		Na_tard = 0
		Na_left = 0
		for j in range(self.J):
			if self.n_ops_left_j[j] > 0:
				Na_left += self.n_ops_left_j[j]
				i = int(self.n_j[j] - self.n_ops_left_j[j]) - 1
				if self.C_ij[i, j] > self.d_j[j]:
					Na_tard += self.n_ops_left_j[j]

		Ne_left = 1 if Ne_left == 0 else Ne_left
		Na_left = 1 if Na_left == 0 else Na_left

		# Enviroment status	features	
		self.U_ave  = np.mean(filtered_U_k)									# 1. Average machine utilization
		U_std       = np.std(filtered_U_k) 									# 2. Std of machine utilization
		C_all       = 1 - np.sum(self.n_ops_left_j)/np.sum(self.n_j)	    # 3. Completion rate of all operation
		C_ave       = np.mean(CR_j)											# 4. Average completion rate
		C_std       = np.std(CR_j)											# 5. Std of completion rate
		self.Tard_e = Ne_tard/Ne_left 	    								# 6. Estimate tardiness rate
		self.Tard_a = Na_tard/Na_left		    							# 7. Actual tardiness rate

		observation = 	[self.U_ave, U_std, C_all, C_ave, C_std, self.Tard_e, self.Tard_a]
		self.observation = np.array(observation, dtype=np.float32)  

	def calc_reward(self):
		if self.Tard_a < self.pre_Tard_a:
			self.reward = 1
		else:
			if self.Tard_a > self.pre_Tard_a:
				self.reward = -1
			else:
				if self.Tard_e < self.pre_Tard_e:
					self.reward = 1
				else:
					if self.Tard_e > self.pre_Tard_e:
						self.reward = -1
					else:
						if self.U_ave > self.pre_U_ave:
							self.reward = 1
						else:
							if self.U_ave > self.pre_U_ave*0.95:
								self.reward = 0
							else:
								self.reward = -1

		
		self.pre_Tard_e = copy.deepcopy(self.Tard_e)
		self.pre_Tard_a = copy.deepcopy(self.Tard_a)
		self.pre_U_ave 	= copy.deepcopy(self.U_ave)
		
	def calc_tardiness(self):
		C_j = np.max(self.C_ij, axis = 0)
		T_j = np.maximum(C_j - self.d_j, 0)
		self.tardiness = np.sum(T_j)
		return self.tardiness

	def perform_action(self):
		method = Method(self.J, self.K, self.p_ijk, self.h_ijk, self.d_j,
				  self.n_j, self.S_k, self.S_j, self.MC_ji, self.n_ops_left_j, self.JSet, self.T_cur, self.X_ijk)
		return [
          method.CDR1
        , method.CDR2
        , method.CDR3
        , method.CDR4
        , method.CDR5
        , method.CDR6
    	]
	
	"""################################################ S T E P ###################################################"""
	def step(self, action):
		# ----------------------------------------------Action------------------------------------------------
		action_method                   = self.perform_action()					    
		operation_machine_selection     = action_method[action]
		i, j, k                         = operation_machine_selection()
		self.X_ijk[i, j, k]             = 1
		self.S_ij[i, j]                 = max(self.S_j[j], self.S_k[k])
		self.C_ij[i, j]                 = self.S_ij[i, j] + self.p_ijk[i, j, k]  
		self.S_k[k]                     = copy.deepcopy(self.C_ij[i, j])
		
		self.n_ops_left_j[j] -= 1
		if self.n_ops_left_j[j] > 0:
			self.S_j[j]   = copy.deepcopy(self.C_ij[i, j])
			self.T_cur[j] = np.mean(self.S_k[self.h_ijk[i+1,j] == 1])
            
		else:
			self.JSet.remove(j)
			self.count += 1
			if self.count >= 70:
				print("There are", len(self.JSet), "jobs left")
				self.count = 0

			if j in self.JA_event:
				print("----------- Rework job ", j)
				self.J += 1
				self.JSet.append(self.J-1)

				# num operation of new job
				n_newjob                   = copy.deepcopy(self.org_n_j[j])
				self.n_j                   = np.append(self.n_j, n_newjob)
				self.n_ops_left_j          = np.append(self.n_ops_left_j, n_newjob)
				# processing time
				p_newjob                   = copy.deepcopy(self.org_p_ijk[:, j, :])
				p_newjob_reshape           = copy.deepcopy(p_newjob[:, np.newaxis, :])
				self.p_ijk                 = np.concatenate((self.p_ijk, p_newjob_reshape), axis= 1)

				# capable machine            
				h_newjob                   = copy.deepcopy(self.org_h_ijk[:, j, :])
				h_newjob_reshape           = copy.deepcopy(h_newjob[:, np.newaxis, :])
				self.h_ijk                 = np.concatenate((self.h_ijk, h_newjob_reshape), axis= 1)

				MC_newjob                  = copy.deepcopy(self.org_MC_ji[j])
				n_MC_newjob                = copy.deepcopy(self.org_n_MC_ji[j])

				self.MC_ji.append(MC_newjob)
				self.n_MC_ji.append(n_MC_newjob)

                # Adjust the org
				
				self.org_n_j               = np.append(self.org_n_j, n_newjob)
				self.org_p_ijk             = np.concatenate((self.org_p_ijk, p_newjob_reshape), axis= 1)
				self.org_h_ijk             = np.concatenate((self.org_h_ijk, h_newjob_reshape), axis= 1)
				self.org_MC_ji.append(MC_newjob)
				self.org_n_MC_ji.append(n_MC_newjob)

				self.X_ijk          		= np.pad(self.X_ijk,((0, 0), (0, 1), (0, 0)), 	mode='constant', constant_values=0)
				self.S_ij                   = np.pad(self.S_ij, ((0, 0), (0, 1)), 		 	mode='constant', constant_values=0)
				self.C_ij                   = np.pad(self.C_ij, ((0, 0), (0, 1)),  			mode='constant', constant_values=0)

				self.S_j                    = np.append(self.S_j, 0)

				T_cur_newjob			    = np.mean(self.S_k[h_newjob[0] == 1])
				self.T_cur                  = np.append(self.T_cur, T_cur_newjob)

				deadline, description = self.JA_event[j]
				if description == "urgent":
					d_newjob = np.sum(np.sum(p_newjob*h_newjob, axis=1)/np.maximum(np.sum(h_newjob, axis=1),1)) *deadline
				else:
					d_newjob = deadline
				self.d_j                   = np.append(self.d_j, d_newjob)

		
		# --------------------------------- Terminated, Reward,  Observation  ------------------------------------
		"""
		1. Retrieve new event
		2. If there is MB => monitor the earliest available time of that machine
		"""	

		# Retrieve new event
		self.MB_event, self.MBList, self.t, self.re     = random_events(self.t, self.K, self.X_ijk, self.S_ij, self.C_ij, self.S_j, self.JSet, self.MB_event, self.S_k, self.UsedMachine)
	
		# Handle mannually if uncertain event is a machine breakdown
		if self.MBList:				
			OJSet = [[] for _ in range(self.J)]
			for j in self.JSet:
				OJSet[j] = np.where(self.S_ij[:int(self.n_j[j]), j] >= self.t)[0].tolist()

			Job_seq = copy.deepcopy(OJSet)
			Mch_seq = find_Mch_seq(self.K, self.X_ijk, self.C_ij, self.t)


			lateststarttime = 0
			operation		= None

			for breakdown_MC in self.MBList:
				Oij_assigned_to_machine = np.argwhere(self.X_ijk[:, :, breakdown_MC])
				for i, j in Oij_assigned_to_machine:
					if self.S_ij[i, j] >= lateststarttime and self.C_ij[i, j] > self.t:
						operation = [i, j]
						lateststarttime = copy.deepcopy(self.S_ij[i, j])

				if len(operation) != 0:
					print("----------- Random MB at time", self.t, "on machine", self.MBList, "at Ope", operation)
					i = copy.deepcopy(operation[0])
					j = copy.deepcopy(operation[1])
					processed  = self.t - self.S_ij[i, j]

					if processed > 0:
						"""Break the operation into  2 segments"""
						self.p_ijk, self.h_ijk, self.X_ijk, self.S_ij, self.C_ij, \
						self.MC_ji, self.n_MC_ji, self.n_j, self.I, self.org_p_ijk, self.org_h_ijk	= change_dataset(self.p_ijk, self.h_ijk, self.X_ijk, self.S_ij, self.C_ij, \
																										self.MC_ji, self.n_MC_ji, self.n_j, j, i, processed, self.I, self.K, self.org_p_ijk, self.org_h_ijk)
						
					# if j not in self.JSet:
					# 	self.JSet.append(j)

					# self.n_ops_left_j[j] += 1

				self.S_k[breakdown_MC] = self.t + self.re[breakdown_MC]

				id_ope_onMCh     = Mch_seq[breakdown_MC].index(operation)
				self.X_ijk, self.S_ij, self.C_ij = RightShift(breakdown_MC, id_ope_onMCh, self.S_k[breakdown_MC], Job_seq, Mch_seq, self.X_ijk, self.S_ij, self.C_ij, self.p_ijk, self.n_j)
				
				self.S_k = np.zeros(self.K)
				for k in range(self.K):
					indices = np.where(self.X_ijk[:, :, k] == 1)
					completion_times = self.C_ij[indices]
					self.S_k[k] = np.max(completion_times) if len(completion_times) > 0 else 0

				self.S_j = np.max(self.C_ij, axis=0)

		if not self.JSet:
			print("====== Done ======")
			self.done = True
			self.tardiness = self.calc_tardiness()
		
		self.calc_observation()
		self.calc_reward()

		return self.observation, self.reward, self.done, False, {}
	

	"""############################################### R E S E T ##################################################"""

	def reset(self, seed=None, test=None, datatest=None, scenariotest=None):
		if seed is not None:
			self.seed(seed)

		super().reset(seed=seed)
		self.count      = 0

		self.done       = False
		self.t          = 0
		self.reward     = 0
		self.pre_Tard_e = 0
		self.pre_Tard_a = 0
		self.pre_U_ave  = 0
		
		if test is None:

			CaseID = random.choice(self.CaseList)
			# CaseID                  = "_fixed_instance"
			data_path               = f"DATA/SMALL/Case{CaseID}_480.txt"

			self.J, self.I, self.K, self.p_ijk, self.h_ijk,\
			self.d_j, self.n_j, self.MC_ji, self.n_MC_ji,  \
			self.OperationPool      = read_txt(data_path)

			self.JA_event   		= generate_JA_event (self.J, self.planning_horizon, self.ReworkProbability)
			self.MB_event			= [[] for _ in range(self.K)]

		else:
			print("Testing")

			data_path               = f"VALIDATION/SMALL/Case{datatest}_480.txt"
			self.J, self.I, self.K, self.p_ijk, self.h_ijk,\
			self.d_j, self.n_j, self.MC_ji, self.n_MC_ji,  \
			self.OperationPool      = read_txt(data_path)
			
			# self.JA_event           = self.scenarios[datatest + scenariotest]
			
			scenariotest = str(datatest) + scenariotest
			self.load_scenario(scenariotest)

			self.JA_event = {item[0]: (item[1], item[2]) for item in self.JA_event}

			print ("MB \n", self.MB_event)

		self.S_j                = np.zeros((self.J))
		self.S_k                = np.zeros((self.K))
		self.X_ijk              = np.zeros((self.I, self.J, self.K))
		self.S_ij               = np.zeros((self.I, self.J))
		self.C_ij               = np.zeros((self.I, self.J))
		self.JSet               = list(range(self.J))
		self.n_ops_left_j       = copy.deepcopy(self.n_j)
		self.T_cur 				= np.zeros((self.J))

		self.org_p_ijk          = copy.deepcopy(self.p_ijk)
		self.org_h_ijk			= copy.deepcopy(self.h_ijk)
		self.org_n_j            = copy.deepcopy(self.n_j)
		self.org_MC_ji          = copy.deepcopy(self.MC_ji)
		self.org_n_MC_ji        = copy.deepcopy(self.n_MC_ji)

		self.UsedMachine        = np.any(self.h_ijk == 1, axis=(0, 1))

		# ---------------------------------------------Observation--------------------------------------------
		self.observation = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
	
		return self.observation, {}
	