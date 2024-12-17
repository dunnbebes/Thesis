import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import copy
import random
import pickle
from gymnasium import spaces
from KeyRef_2.KeyRef_2_action_space   import Method
from util.util_load          import read_txt
from util.util_reschedule    import generate_random_event, generate_JA_event, change_dataset
from util.util_action 		 import find_Mch_seq, RightShift


def random_events(t, K, X_ijk, S_ij, C_ij, S_j, JSet, JA_event, MB_event, S_k, UsedMachine):
	MBList   		= []
	events   		= {}
	re       		= np.zeros((K)) 
	
	if all(isinstance(t[2], str) for t in JA_event):
        # Loose duedate setting (New jobs = Rework)
		for job, deadline, description in JA_event:           
			time_occur  = copy.deepcopy(np.maximum(C_ij[:, job]))
			if time_occur not in events:
				events[time_occur] = []
			events[time_occur].append(("JA", job, deadline, description))

	elif all(isinstance(t[2], (int, float)) for t in JA_event):
        # Tight duedate setting (New jobs = Completely new jobs)
		for component, arrival_time, deadline in JA_event:           
			time_occur  = copy.deepcopy(arrival_time)
			if time_occur not in events:
				events[time_occur] = []
			events[time_occur].append(("JA", component, arrival_time, deadline))

	for k in range(K):
		if MB_event[k]:
			time_occur, repair, description = MB_event[k][0]
			if time_occur not in events:
				events[time_occur] = []
			events[time_occur].append(("MB", k, repair, description))
    
	if events:
		original_time   = copy.deepcopy(int(min(events.keys())))
		triggered_event = copy.deepcopy(events[original_time])
		for uncertain_type, k, repair_time, description in triggered_event:
			if uncertain_type == "MB":
				mask = X_ijk[:, :, k] == 1  # Boolean array where True indicates assigned to machine k
				start_times = S_ij[mask]
				if start_times.size > 0:
					max_start_time = np.max(start_times)

					if original_time <= t: 	
						new_time = max(max_start_time, original_time)
					else: 	
						new_time = copy.deepcopy(original_time)
					
					X_mask   =  X_ijk.astype(bool)
					
					"""Find affected operation"""
					# Use the boolean mask to find the indices where overlap occurs
					overlap_mask = np.logical_or(
						np.logical_and(S_ij >= new_time        		 , C_ij <= new_time + repair_time),       # in
						np.logical_and(S_ij <= new_time        		 , C_ij >  new_time        ),       # left
						np.logical_and(S_ij <  new_time + repair_time, C_ij >= new_time + repair_time))       # right
					indices = np.argwhere(X_mask[:, :, k] & overlap_mask[:, :])

					event = (uncertain_type, k, repair_time, description)
					events[original_time].remove(event)
					if len(indices) > 0:
						events.setdefault(new_time, []).append(event)

	events = {key: value for key, value in events.items() if value != []}

	if events:
		# print("find events")
		new_time 		= copy.deepcopy(int(min(events.keys())))
		triggered_event = copy.deepcopy(events[new_time])
		T 				= np.min(S_k[UsedMachine])
		if len(JSet)> 0 and T < new_time:
			# print("but prioritize assigning jobs", len(JSet))
			# prioritize assign jobs to machines
			new_time = np.min(S_j[JSet])
			MBList = []    
			triggered_event =[]
		else:
			# uncertain events occur
			# print("and really have events")
			for uncertain_type, partID, time_event, description in triggered_event:
				if uncertain_type == "JA":
					# Update remaining JA_event
					JA_event.remove((partID, time_event, description))
				else:
					MBList.append(partID)
					MB_event[partID].pop(0)
					re[partID] = copy.deepcopy(time_event)
	else:
		print("find no events")
		if JSet:
			new_time = np.min(S_j[JSet])
			triggered_event = []
		else:
			new_time = np.max(S_j)
			triggered_event = []

	return JA_event, MB_event, new_time, triggered_event, re, MBList
			

class Luo_DDQN_env(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self, K, planning_horizon, ReworkProbability, scenarios, WeibullDistribution, critical_machines,
			  	 master, tight_duedate_setting, JA_only_setting, directory):
		super(Luo_DDQN_env, self).__init__()

		self.K              		= copy.deepcopy(K)
		self.planning_horizon   	= copy.deepcopy(planning_horizon)
		self.ReworkProbability		= copy.deepcopy(ReworkProbability)
		self.scenarios          	= copy.deepcopy(scenarios)
		self.WeibullDistribution    = copy.deepcopy(WeibullDistribution)
		self.critical_machines 		= copy.deepcopy(critical_machines)
		self.master					= copy.deepcopy(master)
		self.tight_duedate_setting	= copy.deepcopy(tight_duedate_setting)
		self.JA_only_setting		= copy.deepcopy(JA_only_setting)
		
		self.directory				= copy.deepcopy(directory)


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
		self.JA_event 	        = copy.deepcopy(self.current_scenario.JA_event)
		self.MB_event 			= copy.deepcopy(self.current_scenario.MB_event)

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

		Ne_left = max(Ne_left,1)
		Na_left = max(Na_left,1)

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
	
	def handle_machine_breakdown (self):
		
		# Handle mannually if uncertain event is a machine breakdown
		if self.MBList:				
			OJSet = [[] for _ in range(self.J)]
			for j in self.JSet:
				OJSet[j] = np.where(self.S_ij[:int(self.n_j[j]), j] >= self.t)[0].tolist()

			Job_seq = copy.deepcopy(OJSet)
			Mch_seq = find_Mch_seq(self.K, self.X_ijk, self.C_ij, self.t)


			for breakdown_MC in self.MBList:
				operation		= None
				lateststarttime = 0
				Oij_assigned_to_machine = np.argwhere(self.X_ijk[:, :, breakdown_MC])
				findingoperation = False
				for i, j in Oij_assigned_to_machine:
					if findingoperation == True:
						break
					if self.S_ij[i, j] >= lateststarttime:
						
						if self.C_ij[i, j] > self.t:
							operation = [i, j]
							findingoperation = True
						else:
							lateststarttime = copy.deepcopy(self.S_ij[i, j])
						


				if operation is not None:
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
							

				self.S_k[breakdown_MC] = self.t + self.re[breakdown_MC]

				id_ope_onMCh     = Mch_seq[breakdown_MC].index(operation)
				self.X_ijk, self.S_ij, self.C_ij = RightShift(breakdown_MC, id_ope_onMCh, self.S_k[breakdown_MC], Job_seq, Mch_seq, self.X_ijk, self.S_ij, self.C_ij, self.p_ijk, self.n_j)
				
				self.S_k = np.zeros(self.K)
				for k in range(self.K):
					indices = np.where(self.X_ijk[:, :, k] == 1)
					completion_times = self.C_ij[indices]
					self.S_k[k] = np.max(completion_times) if len(completion_times) > 0 else 0

				self.S_j = np.max(self.C_ij, axis=0)

		return 
	
	def handle_job_arrival (self):
		JA = []
		if self.triggered_event is not None:
			for uncertain_type, partID, time_event, description in self.triggered_event:
				if uncertain_type == "JA":            
					JA.append((partID, time_event, description))   # if Job arrival


		for job_info, info1, info2 in JA: 
			"""Adjust the dataset"""              
			self.J += 1
			self.count +=1
			if all(isinstance(t[2], str) for t in JA):
				#Loose duedate setting
				jobresemble = copy.deepcopy(job_info)
				deadline    = copy.deepcopy(info1)
				description = copy.deepcopy(info2)

				n_newjob    = copy.deepcopy(self.org_n_j[jobresemble])
				p_newjob    = copy.deepcopy(self.org_p_ijk[:, jobresemble, :])
				h_newjob    = copy.deepcopy(self.org_h_ijk[:, job_info, :])
				MC_newjob   = copy.deepcopy(self.org_MC_ji[job_info])
				n_MC_newjob = copy.deepcopy(self.org_n_MC_ji[job_info])

				TPT         = np.sum(np.sum(p_newjob * h_newjob, axis= 1)/ np.maximum(np.sum(h_newjob, axis=1),1))

				# deadline
				if description == "urgent":
					d_newjob = TPT*deadline
				else:
					d_newjob = deadline

			elif all(isinstance(t[2], (int, float)) for t in JA):
				#Tight duedate setting
				jobinstance  = copy.deepcopy(job_info)
				d_newjob     = copy.deepcopy(info2)

				job_profile  = self.master[jobinstance]
				n_newjob     = copy.deepcopy(job_profile.n_)
				p_newjob     = copy.deepcopy(job_profile.p_ik[:self.I])
				h_newjob     = copy.deepcopy(job_profile.h_ik[:self.I])
				MC_newjob    = copy.deepcopy(job_profile.MC_i[:self.I])
				n_MC_newjob  = copy.deepcopy(job_profile.n_MC_i[:self.I])

			else:
				raise ValueError("Error: Mixed duedate setting")
	
			
			# num operation of new job
			self.n_j                   = np.append(self.n_j, n_newjob)
			self.n_ops_left_j		   = np.append(self.n_ops_left_j, n_newjob)
			# processing time
			p_newjob_reshape           = copy.deepcopy(p_newjob[:, np.newaxis, :])
			self.p_ijk                 = np.concatenate((self.p_ijk, p_newjob_reshape), axis= 1)
			# deadline
			self.d_j                   = np.append(self.d_j, d_newjob)
			# capable machine            
			h_newjob_reshape           = copy.deepcopy(h_newjob[:, np.newaxis, :])
			self.h_ijk                 = np.concatenate((self.h_ijk, h_newjob_reshape), axis= 1)

			self.MC_ji  .append(MC_newjob)
			self.n_MC_ji.append(n_MC_newjob)

			# Adjust the org
			self.org_n_j               = np.append(self.org_n_j, n_newjob)
			self.org_p_ijk             = np.concatenate((self.org_p_ijk, p_newjob_reshape), axis= 1)
			self.org_h_ijk             = np.concatenate((self.org_h_ijk, h_newjob_reshape), axis= 1)
			self.org_MC_ji.append(MC_newjob)
			self.org_n_MC_ji.append(n_MC_newjob)

			self.JSet.append(self.J-1)
			self.S_j                    = np.append(self.S_j, 0)
			T_cur_newjob			    = np.mean(self.S_k[h_newjob[0] == 1])
			self.T_cur                  = np.append(self.T_cur, T_cur_newjob)

			# X, S, C
			self.X_ijk          = np.pad(self.X_ijk,     ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
			self.S_ij           = np.pad(self.S_ij,      ((0, 0), (0, 1)),         mode='constant', constant_values=0)
			self.C_ij           = np.pad(self.C_ij,      ((0, 0), (0, 1)),         mode='constant', constant_values=0)
		
		return
	
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
		if self.n_ops_left_j[j] >= 0.9:
			self.S_j[j]   = copy.deepcopy(self.C_ij[i, j])
			self.T_cur[j] = np.mean(self.S_k[self.h_ijk[i+1,j] == 1])
            
		else:
			self.JSet.remove(j)

		if j == 0:
			print(self.n_ops_left_j[0], 0 in self.JSet)
		
		finding = False

		while finding == False:
			# Retrieve new event
			self.JA_event, self.MB_event, self.new_time, \
			self.triggered_event, self.re, self.MBList   = random_events(self.t, self.K, self.X_ijk, self.S_ij, self.C_ij, self.S_j, self.JSet, 
																		self.JA_event, self.MB_event, self.S_k, self.UsedMachine)
			
			# Handle mannually if uncertain event is a machine breakdown
			self.handle_machine_breakdown()

			# Job Arrival
			self.handle_job_arrival()

			if not(len(self.JSet) == 0 and len(self.JA_event)> 0):
				finding = True
			else:
				print(len(self.JSet), len(self.JA_event), self.t, self.triggered_event)
		
		# --------------------------------- Terminated, Reward,  Observation  ------------------------------------
		
		
		if len(self.JSet) == 0 and len(self.JA_event) == 0:
			print("====== Done ======")
			self.done = True
			self.tardiness = self.calc_tardiness()
		else: 
			if self.count >= 70:
				self.count = 0
				print (self.count, "There are", len(self.JA_event), "jobs left")
		
		if len(self.JSet) < 1: print("JSet contains", len(self.JSet), "elements while JA_event contains", len(self.JA_event), self.done)
		self.calc_observation()
		self.calc_reward()

		return self.observation, self.reward, self.done, False, {}
	

	"""############################################### R E S E T ##################################################"""

	def reset(self, seed=None, test=None, datatest=None, scenariotest=None):
		if seed is not None:
			self.seed(seed)

		super().reset(seed=seed)
		self.done       = False
		self.t          = 0
		self.reward     = 0
		self.pre_Tard_e = 0
		self.pre_Tard_a = 0
		self.pre_U_ave  = 0
		
		if test is None:

			CaseID 						 = random.choice(self.CaseList)
			data_path               	 = f"{self.directory}/Case{CaseID}_480.txt"
			self.J, self.I, self.K, self.p_ijk, self.h_ijk,\
			self.d_j, self.n_j, self.MC_ji, self.n_MC_ji,  \
			self.OperationPool      	 = read_txt(data_path)

			remaining_info_file = f'{self.directory}/Case{CaseID}_{self.planning_horizon // 60}_InfoNewJob.pkl'
			with open(remaining_info_file, 'rb') as f:
				remaining_batches = pickle.load(f)

			new_job_indices = [comp_id for comp_id, qty in remaining_batches.items() for _ in range(qty)]
			self.JA_event, self.MB_event = generate_random_event(self.J, self.K, self.planning_horizon, self.WeibullDistribution, 
																	self.critical_machines, self.ReworkProbability, 
																	self.master, new_job_indices)
			# self.JA_event   		= generate_JA_event (self.J, self.planning_horizon, self.ReworkProbability)
			# self.MB_event			= [[] for _ in range(self.K)]

		else:
			data_path               	= f"{self.directory}_VALIDATION/Case{datatest}_480.txt"
			self.J, self.I, self.K, self.p_ijk, self.h_ijk,\
			self.d_j, self.n_j, self.MC_ji, self.n_MC_ji,  \
			self.OperationPool      	= read_txt(data_path)
			
			# self.JA_event           = self.scenarios[datatest + scenariotest]
			
			scenariotest = str(datatest) + scenariotest
			self.load_scenario(scenariotest)

			# self.JA_event = {item[0]: (item[1], item[2]) for item in self.JA_event}

		if self.JA_only_setting == True:
			self.MB_event = [[] for _ in range(self.K)]

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

		self.count = 0
		return self.observation, {}
	