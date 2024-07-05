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
from util.util_reschedule    import generate_JA_event


class KeyRef2_Env(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self, K, planning_horizon, ReworkProbability, scenarios):
		super(KeyRef2_Env, self).__init__()

		self.K              	= copy.deepcopy(K)
		self.planning_horizon   = copy.deepcopy(planning_horizon)
		self.ReworkProbability	= copy.deepcopy(ReworkProbability)
		self.scenarios          = copy.deepcopy(scenarios)
		self.method_list 		= ["CDR1", "CDR2", "CDR3", "CDR4", "CDR5", "CDR6"]
		self.CaseList = ['_fixed_instance'] + [case+1 for case in range(1, 48)]
		self.action_space = spaces.Discrete(6)
		self.observation_space = spaces.Box(low=0, high=2,
											shape=(7,), dtype=np.float32)

	def seed(self, seed=None):
		random.seed(seed)
		np.random.seed(seed)


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
		Ne_left = 0 if self.done == False else 1
		for j in range(self.J):
			if self.n_ops_left_j[j] > 0:
				Ne_left += self.n_ops_left_j[j]
				T_left = 0
				for i in range(self.n_j[j] - self.n_ops_left_j[j], self.n_j[j]): 
					t_mean_ij = np.mean(self.p_ijk[i][j] * self.h_ijk[i][j])
					T_left   += t_mean_ij
					if self.T_cur + T_left > self.d_j[j]:
						Ne_tard += (self.n_j[j] - i) 
						break

		# Calculate actual tardiness rate 
		Na_tard = 0
		Na_left = 0 if self.done == False else 1
		for j in range(self.J):
			if self.n_ops_left_j[j] > 0:
				Na_left += self.n_ops_left_j[j] > 0 
				i = int(self.n_j[j] - self.n_ops_left_j[j])
				if self.C_ij[i, j] > self.d_j[j]:
					Na_tard += self.n_ops_left_j[j]

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
		method = self.method_list[action]
		# print("-------------------------------------------------")
		# print(f'Method selection:                    {method}')
		
		action_method                   = self.perform_action()					    
		operation_machine_selection     = action_method[action]
		i, j, k                         = operation_machine_selection()
		self.X_ijk[i, j, k]             = 1
		self.S_ij[i, j]                 = max(self.S_j[j], self.S_k[k])
		self.C_ij[i, j]                 = self.S_ij[i, j] + self.p_ijk[i, j, k]  
		self.S_k[k]                     = copy.deepcopy(self.C_ij[i, j])
		self.T_cur                      = np.mean(self.S_k)

		self.n_ops_left_j[j] -= 1
		if self.n_ops_left_j[j] != 0:
			self.S_j[j] = copy.deepcopy(self.C_ij[i, j])
            
		else:
			self.JSet.remove(j)
			if j in self.JA_event:
				self.J += 1
				self.JSet.append(self.J-1)

				# num operation of new job
				n_newjob                   = copy.deepcopy(self.n_j[j])
				self.n_j                   = np.append(self.n_j, n_newjob)
				self.n_ops_left_j          = np.append(self.n_ops_left_j, n_newjob)
				# processing time
				p_newjob                   = copy.deepcopy(self.p_ijk[:, j, :])
				p_newjob_reshape           = copy.deepcopy(p_newjob[:, np.newaxis, :])
				self.p_ijk                 = np.concatenate((self.p_ijk, p_newjob_reshape), axis= 1)

				deadline, description = self.JA_event[j]
				if description == "urgent":
					d_newjob = np.sum(np.mean(p_newjob, axis=0)) *deadline
				else:
					d_newjob = deadline
				self.d_j                   = np.append(self.d_j, d_newjob)

				# capable machine            
				h_newjob                   = copy.deepcopy(self.h_ijk[:, j, :])
				h_newjob_reshape           = copy.deepcopy(h_newjob[:, np.newaxis, :])
				self.h_ijk                 = np.concatenate((self.h_ijk, h_newjob_reshape), axis= 1)

				MC_newjob                  = copy.deepcopy(self.MC_ji[j])
				self.MC_ji.append(MC_newjob)

				self.X_ijk          		= np.pad(self.X_ijk,((0, 0), (0, 1), (0, 0)), 	mode='constant', constant_values=0)
				self.S_ij                   = np.pad(self.S_ij, ((0, 0), (0, 1)), 		 	mode='constant', constant_values=0)
				self.C_ij                   = np.pad(self.C_ij, ((0, 0), (0, 1)),  			mode='constant', constant_values=0)

				self.S_j                    = np.append(self.S_j, 0)


		# --------------------------------- Terminated, Reward,  Observation  ------------------------------------
		if self.JSet: 
			self.t = np.min(self.S_j[self.JSet])
		else:
			self.done = True
		
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
			CaseID = random.choice(self.CaseList)
			data_path               = f"DATA/SMALL/Case{CaseID}_480.txt"
			self.J, self.I, self.K, self.p_ijk, self.h_ijk,\
			self.d_j, self.n_j, self.MC_ji, self.n_MC_ji,  \
			self.OperationPool      = read_txt(data_path)

			self.JA_event   		= generate_JA_event (self.J, self.planning_horizon, self.ReworkProbability)

		else:
			data_path               = f"VALIDATION/SMALL/Case{datatest}_480.txt"
			self.J, self.I, self.K, self.p_ijk, self.h_ijk,\
			self.d_j, self.n_j, self.MC_ji, self.n_MC_ji,  \
			self.OperationPool      = read_txt(data_path)
			
			self.JA_event = self.scenarios[datatest + scenariotest]
			

		self.S_j                = np.zeros((self.J))
		self.S_k                = np.zeros((self.K))
		self.X_ijk              = np.zeros((self.I, self.J, self.K))
		self.S_ij               = np.zeros((self.I, self.J))
		self.C_ij               = np.zeros((self.I, self.J))
		self.JSet               = list(range(self.J))
		self.n_ops_left_j       = copy.deepcopy(self.n_j)
		self.T_cur              = 0

		# ---------------------------------------------Observation--------------------------------------------
		self.observation = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

		return self.observation, {}
