import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import copy
import random
from gymnasium import spaces
from env_action.action_space import Method
from util.util_reschedule    import store_schedule, update_schedule, random_events, snapshot, generate_random_event


class FJSP_under_uncertainties_Env(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self, fixed_instance, fixed_scenario, instances, scenarios, K, WeibullDistribution, 
			  critical_machines, ReworkProbability, planning_horizon, PopSize, maxtime, maxJob, maxOpe, reward_ratio):
		super(FJSP_under_uncertainties_Env, self).__init__()

		self.K              		 	= copy.deepcopy(K)
		self.PopSize				 	= copy.deepcopy(PopSize)
		self.maxtime                	= copy.deepcopy(maxtime)
		self.maxJob  					= copy.deepcopy(maxJob)
		self.maxOpe            			= copy.deepcopy(maxOpe)
		self.reward_ratio               = copy.deepcopy(reward_ratio)
		self.WeibullDistribution        = copy.deepcopy(WeibullDistribution)
		self.critical_machines			= copy.deepcopy(critical_machines)
		self.ReworkProbability			= copy.deepcopy(ReworkProbability)
		self.planning_horizon	     	= copy.deepcopy(planning_horizon)
		self.fixed_instance			 	= copy.deepcopy(fixed_instance)
		self.fixed_scenario			 	= copy.deepcopy(fixed_scenario)
		self.instances 					= copy.deepcopy(instances)
		self.scenarios			     	= copy.deepcopy(scenarios)
		self.list_instances				= list(self.instances.keys())
		self.list_scenarios 			= list(self.scenarios.keys())
		self.num_scenario_per_instance 	= 10

		self.method_list 			 = ["GA", "TS", "LFOH", "LAPH", "LAP_LFO", 
						  				"LFOH-TS", "LAPH-TS", "LFOH-GA", "LAPH-GA",
						  				"CDR1", "CDR2", "CDR3", "CDR5", "CDR6",
										"RCRS"]

		self.action_space = spaces.Discrete(15)
		self.observation_space = spaces.Box(low=0, high=2,
											shape=(16,), dtype=np.float32)

	def load_instance(self, instance_id):
		self.current_instance 	= self.instances[instance_id]
		self.J 					= copy.deepcopy(self.current_instance.J)
		self.I 					= copy.deepcopy(self.current_instance.I)
		self.X_ijk 				= copy.deepcopy(self.current_instance.X_ijk)
		self.S_ij 				= copy.deepcopy(self.current_instance.S_ij)
		self.C_ij 				= copy.deepcopy(self.current_instance.C_ij)
		self.C_j 				= copy.deepcopy(self.current_instance.C_j)
		self.p_ijk 				= copy.deepcopy(self.current_instance.p_ijk)
		self.h_ijk 				= copy.deepcopy(self.current_instance.h_ijk)
		self.d_j 				= copy.deepcopy(self.current_instance.d_j)
		self.n_j 				= copy.deepcopy(self.current_instance.n_j)
		self.MC_ji				= copy.deepcopy(self.current_instance.MC_ji)
		self.n_MC_ji 			= copy.deepcopy(self.current_instance.n_MC_ji)
		self.OperationPool 		= copy.deepcopy(self.current_instance.OperationPool)
		self.n_Mch              = np.sum(np.max(self.h_ijk, axis= (0, 1)))/self.K
		self.org_J              = copy.deepcopy(self.J)
		self.org_p_ijk      	= copy.deepcopy(self.p_ijk)
		self.org_h_ijk      	= copy.deepcopy(self.h_ijk)
		self.org_n_j        	= copy.deepcopy(self.n_j)
		self.org_MC_ji      	= copy.deepcopy(self.MC_ji)
		self.org_n_MC_ji    	= copy.deepcopy(self.n_MC_ji)

	def load_scenario(self, scenario_id):
		self.current_scenario 	= self.scenarios[scenario_id]
		self.JA_event 	        = self.current_scenario.JA_event
		self.MB_event 			= self.current_scenario.MB_event
		
	def seed(self, seed=None):
		random.seed(seed)
		np.random.seed(seed)

	def calc_observation (self):
		# Find utilization of each machine
		S_mask 		 = self.S_ij < self.t
		S_expand 	 = S_mask[:, :, np.newaxis]
		X_filtered 	 = self.X_ijk * S_expand
		Usage     	 = np.sum(X_filtered * self.p_ijk, axis = (0, 1))
		mask      	 = self.CT_k != 0
		filtered_U_k = Usage[mask] / self.CT_k[mask]
		
		# Find completion rate of each job
		CR_j = 1 - self.n_ops_left_j/self.n_j

		# Calculate estimated tardiness rate
		Ne_tard = 0
		Ne_left = 0 if self.t < np.max(self.C_ij) else 1
		for j in range(self.J):
			if self.n_ops_left_j[j] > 0:
				Ne_left += self.n_ops_left_j[j]
				T_left = 0
				for i in self.OJSet[j]: 
					t_mean_ij = np.sum(self.p_ijk[i, j]*self.h_ijk[i,j])/np.maximum(np.sum(self.h_ijk[i,j]),1)
					T_left   += t_mean_ij
					if self.T_cur[j] + T_left > self.d_j[j]:
						Ne_tard += (self.n_j[j] - i) 
						break

		# Calculate actual tardiness rate 
		Na_tard = 0
		Na_left = 0 if self.t < np.max(self.C_ij) else 1
		for j in range(self.J):
			if self.n_ops_left_j[j] > 0:
				Na_left += self.n_ops_left_j[j] 
				i = int(self.n_j[j] - self.n_ops_left_j[j]) -1
				if self.C_ij[i, j] > self.d_j[j]:
					Na_tard += self.n_ops_left_j[j]
		
		
		# Problem size features
		n_Job  = len(self.JSet)/self.maxJob    				# 1. Number of job left 		- normalize
		n_Ops  = sum(map(len, self.OJSet))/self.maxOpe		# 2. Number of operation left 	- normalize
															# 3. Number of machine 			- normalize
		# Scenario features (taken from snapshot)
		# Enviroment status	features	
		U_ave  = np.mean(filtered_U_k)						# 1. Average machine utilization
		U_std  = min(np.std(filtered_U_k), 2)				# 2. Std of machine utilization
		C_all  = 1 - n_Ops/np.sum(self.n_j)					# 3. Completion rate of all operation
		C_ave  = np.mean(CR_j)								# 4. Average completion rate
		C_std  = min(np.std(CR_j), 2)						# 5. Std of completion rate
		Tard_e = Ne_tard/Ne_left 	    					# 6. Estimate tardiness rate
		Tard_a = Na_tard/Na_left		    				# 7. Actual tardiness rate
		observation = 	[n_Job, n_Ops, self.n_Mch, 
				 		self.JA_boolean, self.JA_long_boolean, self.JA_urgent_boolean,
        				self.MB_boolean, self.MB_critical_boolean, self.sum_re,
					   	U_ave, U_std, C_all, C_ave, C_std, Tard_e, Tard_a]
		self.observation = np.array(observation, dtype=np.float32)  

	def calc_reward(self):
		ConsideredJob = [job for job in self.pre_JSet if job not in self.JSet]
		self.C_j      = np.max(self.C_ij, axis = 0)
		filtered_Tard = np.sum(np.maximum(self.C_j[ConsideredJob] - self.d_j[ConsideredJob], 0))
		self.all_Tard = np.sum(np.maximum(self.C_j - self.d_j, 0))

		self.reward   = -(self.all_Tard*(1-self.reward_ratio) + filtered_Tard*self.reward_ratio)/self.planning_horizon
		self.pre_JSet = copy.deepcopy(self.JSet)

	def calc_tardiness(self):
		return self.all_Tard	

	def perform_action(self):
		method = Method(self.J, self.I, self.K, self.p_ijk, self.h_ijk, self.d_j, self.n_j, \
						self.MC_ji, self.n_MC_ji, self.n_ops_left_j, \
						self.OperationPool, self.S_k, self.S_j, self.JSet, self.OJSet, self.t, \
						self.X_ijk, self.S_ij, self.C_ij, self.C_j, self.CT_k, self.T_cur, self.Tard_job, \
						self.NewJobList, self.MBList, self.PopSize, self.maxtime, self.re)
		
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
        # , method.CDR4
        , method.CDR5
        , method.CDR6
        , method.RouteChange_RightShift
    	]
	
	"""################################################ S T E P ###################################################"""
	def step(self, action):
		# Store previous schedule
		self.X_previous, self.S_previous, self.C_previous = store_schedule(self.X_ijk, self.S_ij, self.C_ij)

		# ----------------------------------------------Action------------------------------------------------
		method = self.method_list[action]
		# print("-------------------------------------------------")
		print(f'Method selection:                    {method}')
		
		action_method                                = self.perform_action()					    
		reschedule							         = action_method[action]
		self.GBest, \
		self.X_ijk, self.S_ij, self.C_ij, self.C_j   = reschedule()
		self.X_ijk, self.S_ij, self.C_ij, self.C_j   = update_schedule(self.DSet, self.ODSet, self.t, self.X_ijk, self.S_ij, self.C_ij,\
																       self.X_previous, self.S_previous, self.C_previous)

		# ---------------------------------------- State transition ------------------------------------------
		self.JA_event, self.MB_event, self.t, self.triggered_event, \
		self.re, self.MB_record                                         = random_events(self.t, self.J, self.K, self.X_ijk, self.S_ij, self.C_ij, self.C_j, \
																					   	self.JA_event, self.MB_event, self.MB_record)
		# Snapshot
		self.S_k, self.S_j, self.J, self.I, self.JSet, self.OJSet, self.DSet,    \
		self.ODSet, self.OperationPool, self.n_ops_left_j, self.MC_ji,           \
        self.n_MC_ji, self.d_j, self.n_j, self.p_ijk, self.h_ijk,                \
        self.org_p_ijk, self.org_h_ijk, self.org_n_j, self.org_MC_ji,            \
        self.org_n_MC_ji, self.X_ijk, self.S_ij, self.C_ij, \
		self.C_j, self.JA_boolean, self.JA_long_boolean, self.JA_urgent_boolean, \
        self.MB_boolean, self.MB_critical_boolean, self.sum_re,                  \
        self.CT_k, self.T_cur, self.Tard_job, self.NewJobList, self.MBList= snapshot(self.t, self.triggered_event, self.MC_ji, self.n_MC_ji,                 \
																					self.d_j, self.n_j, self.p_ijk, self.h_ijk, self.J, self.I, self.K,  	\
																					self.X_ijk, self.S_ij, self.C_ij, self.OperationPool, self.re, self.S_k,\
																					self.org_J, self.org_p_ijk, self.org_h_ijk, self.org_n_j,               \
																					self.org_MC_ji, self.org_n_MC_ji, self.C_j                              )																				

		# --------------------------------- Terminated, Reward,  Observation  ------------------------------------
		if self.t >= np.max(self.C_ij) or self.triggered_event is None: 
			self.done = True
		
		self.calc_reward()
		self.calc_observation()

		return self.observation, self.reward, self.done, False, {}
	

	"""############################################### R E S E T ##################################################"""

	def reset(self, seed=None, test=None, datatest=None, scenariotest=None):
		if seed is not None:
			self.seed(seed)

		super().reset(seed=seed)
		# Reset input
		self.t                  	   = 0
		if test is None:
			if self.fixed_instance:
				self.load_instance('_fixed_instance')
				if self.fixed_scenario:
					self.load_scenario('_fixed_scenario')
				else:
					self.load_scenario(random.choice(self.list_scenarios))
			else:
				if self.num_scenario_per_instance >= 10:
					self.num_scenario_per_instance = 0
					self.instance_id 			   = random.choice(self.list_instances)
				self.num_scenario_per_instance +=1
				self.load_instance(self.instance_id)
				self.JA_event, self.MB_event = generate_random_event(self.J, self.K, self.planning_horizon, self.WeibullDistribution, 
																	self.critical_machines, self.ReworkProbability)
		else:
			scenariotest = str(datatest) + scenariotest
			self.load_instance(datatest)
			self.load_scenario(scenariotest)

		self.events         = {}
		self.S_k            = np.zeros((self.K))
		self.MB_record      = {}
		self.done 			= False
		

		# ------------------------------------------ State transition ------------------------------------------
		# Random event
		self.JA_event, self.MB_event, self.t, self.triggered_event, \
		self.re, self.MB_record                                         = random_events(self.t, self.J, self.K, self.X_ijk, self.S_ij, self.C_ij, self.C_j, \
																					   	self.JA_event, self.MB_event, self.MB_record)

		# Snapshot
		self.S_k, self.S_j, self.J, self.I, self.JSet, self.OJSet, self.DSet,    \
		self.ODSet, self.OperationPool, self.n_ops_left_j, self.MC_ji,           \
        self.n_MC_ji, self.d_j, self.n_j, self.p_ijk, self.h_ijk,                \
        self.org_p_ijk, self.org_h_ijk, self.org_n_j, self.org_MC_ji,            \
        self.org_n_MC_ji, self.X_ijk, self.S_ij, self.C_ij, \
		self.C_j, self.JA_boolean, self.JA_long_boolean, self.JA_urgent_boolean, \
        self.MB_boolean, self.MB_critical_boolean, self.sum_re,                  \
        self.CT_k, self.T_cur, self.Tard_job, self.NewJobList, self.MBList= snapshot(self.t, self.triggered_event, self.MC_ji, self.n_MC_ji,                 \
																					self.d_j, self.n_j, self.p_ijk, self.h_ijk, self.J, self.I, self.K,  	\
																					self.X_ijk, self.S_ij, self.C_ij, self.OperationPool, self.re, self.S_k,\
																					self.org_J, self.org_p_ijk, self.org_h_ijk, self.org_n_j,               \
																					self.org_MC_ji, self.org_n_MC_ji, self.C_j                              )															
		self.pre_JSet = copy.deepcopy(self.JSet)	

		# ---------------------------------------------Observation--------------------------------------------
		self.calc_observation()

		return self.observation, {}