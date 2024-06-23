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

	def __init__(self, fixed_instance, fixed_scenario, instances, scenarios, WeibullDistribution, 
			  critical_machines, ReworkProbability, planning_horizon, PopSize, maxtime):
		super(FJSP_under_uncertainties_Env, self).__init__()

		self.K              		 	= copy.deepcopy(K)
		self.PopSize				 	= copy.deepcopy(PopSize)
		self.maxtime                	= copy.deepcopy(maxtime)
		self.WeibullDistribution        = copy.deepcopy(WeibullDistribution)
		self.critical_machines			= copy.deepcopy(critical_machines)
		self.ReworkProbability			= copy.deepcopy(ReworkProbability)
		self.planning_horizon	     	= copy.deepcopy(planning_horizon)
		self.fixed_instance			 	= copy.deepcopy(fixed_instance)
		self.fixed_scenario			 	= copy.deepcopy(fixed_scenario)
		self.instances 					= copy.deepcopy(instances)
		self.scenario			     	= copy.deepcopy(scenario)
		self.num_scenario_per_instance 	= 10

		self.method_list 			 = ["exact", "GA", "LFOH", "LAPH", "LAP_LFO", 
						  				"LFOH-TS", "LAPH-TS", "LFOH-GA", "LAPH-GA",
						  				"CDR1", "CDR2", "CDR3", "CDR4", "CDR5", "CDR6",
										"RCRS"]

		self.action_space = spaces.Discrete(16)
		self.observation_space = spaces.Box(low=0, high=255,
											shape=(16,), dtype=np.float32)

	def load_instance(self, instance_id):
        self.current_instance 	= self.instances[instance_id]
        self.J 					= self.current_instance.J
        self.I 					= self.current_instance.I
        self.X_ijk 				= self.current_instance.X_ijk
        self.S_ij 				= self.current_instance.S_ij
        self.C_ij 				= self.current_instance.C_ij
        self.C_j 				= self.current_instance.C_j
        self.p_ijk 				= self.current_instance.p_ijk
        self.h_ijk 				= self.current_instance.h_ijk
        self.d_j 				= self.current_instance.d_j
        self.n_j 				= self.current_instance.n_j
        self.MC_ji				= self.current_instance.MC_ji
        self.n_MC_ji 			= self.current_instance.n_MC_ji
        self.OperationPool 		= self.current_instance.OperationPool
		self.LB_Cmax            = np.sum(np.mean(self.p_ijk*self.h_ijk, axis= 2))

    def load_scenario(self, scenario_id):
		self.current_scenario 	= self.scenarios[scenario_id]
		self.JA_event 	        = self.current_scenario.JA_event
		self.MB_event 			= self.current_scenario.MB_event
		
	def seed(self, seed=None):
		random.seed(seed)
		np.random.seed(seed)

	def calc_observation (self):
		# Find utilization of each machine
		U_k       = np.zeros((self.K))
		Usage     = np.sum(self.X_ijk * self.p_ijk * self.h_ijk, axis = (0, 1))
		mask      = self.CT_k != 0
		U_k[mask] = Usage[mask] / self.CT_k[mask]
		
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
					t_mean_ij = np.mean(self.p_ijk[i][j] * self.h_ijk[i][j])
					T_left   += t_mean_ij
					if self.T_cur + T_left > self.d_j[j]:
						Ne_tard += (self.n_j[j] - i) 
						break

		# Calculate actual tardiness rate 
		Na_tard = 0
		Na_left = 0 if self.t < np.max(self.C_ij) else 1
		for j in range(self.J):
			if self.n_ops_left_j[j] > 0:
				Na_left += self.n_ops_left_j[j] > 0 
				i = int(self.n_j[j] - self.n_ops_left_j[j])
				if self.C_ij[i, j] > self.d_j[j]:
					Na_tard += self.n_ops_left_j[j]
		
		
		# Problem size features
		n_Job  = len(self.JSet)				# 1. Number of job left
		n_Ops  = sum(map(len, self.OJSet))	# 2. Number of operation left
		n_Mch  = copy.deepcopy(self.K)		# 3. Number of machine
		# Scenario features (taken from snapshot)
		# Enviroment status	features	
		U_ave  = np.mean(U_k)				# 1. Average machine utilization
		U_std  = np.std(U_k) 				# 2. Std of machine utilization
		C_all  = n_Ops/np.sum(self.n_j)		# 3. Completion rate of all operation
		C_ave  = np.mean(CR_j)				# 4. Average completion rate
		C_std  = np.std(CR_j)				# 5. Std of completion rate
		Tard_e = Ne_tard/Ne_left 	    	# 6. Estimate tardiness rate
		Tard_a = Na_tard/Na_left		    # 7. Actual tardiness rate

		observation = 	[n_Job, n_Ops, n_Mch, 
				 		self.JA_boolean, self.JA_long_boolean, self.JA_urgent_boolean,
        				self.MB_boolean, self.MB_critical_boolean, self.sum_re,
					   	U_ave, U_std, C_all, C_ave, C_std, Tard_e, Tard_a]
		self.observation = np.array(observation, dtype=np.float32)  

	def calc_reward(self):
		ConsideredJob = [job for job in self.pre_JSet if job not in self.JSet]
		self.C_j      = np.max(self.C_ij, axis = 0)
		filtered_Tard = np.sum(np.maximum(self.C_j[ConsideredJob] - self.d_j[ConsideredJob], 0))
		self.all_Tard = np.sum(np.maximum(self.C_j - self.d_j, 0))

		self.reward   = -(self.all_Tard*0.01 + filtered_Tard*0.99)/self.LB_Cmax
		self.pre_JSet = copy.deepcopy(self.JSet)	

	def perform_action(self):
		method = Method(self.J, self.I, self.K, self.p_ijk, self.h_ijk, self.d_j, self.n_j, \
						self.MC_ji, self.n_MC_ji, self.n_ops_left_j, \
						self.OperationPool, self.S_k, self.S_j, self.JSet, self.OJSet, \
						self.Oij_on_machine, self.affected_Oij, self.t, \
						self.X_ijk, self.S_ij, self.C_ij, self.C_j, self.CT_k, self.T_cur, self.Tard_job, \
						self.NewJobList, self.PopSize, self.maxtime)
		
		return [
          method.exact
        , method.GA
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
	
	"""################################################ S T E P ###################################################"""
	def step(self, action):
		# Store previous schedule
		self.X_previous, self.S_previous, self.C_previous = store_schedule(self.X_ijk, self.S_ij, self.C_ij)

		# ----------------------------------------------Action------------------------------------------------
		method = self.method_list[action]
		print("-------------------------------------------------")
		print(f'Method selection:                    {method}')
		
		action_method                                = self.perform_action()					    
		reschedule							         = action_method[action]
		self.GBest, \
		self.X_ijk, self.S_ij, self.C_ij, self.C_j   = reschedule()
		self.X_ijk, self.S_ij, self.C_ij             = update_schedule(self.DSet, self.ODSet, self.X_ijk, self.S_ij, self.C_ij,\
																       self.X_previous, self.S_previous, self.C_previous)

		# ---------------------------------------- State transition ------------------------------------------
		self.JA_event, self.MB_event, self.t, self.triggered_event, \
		self.affected_Oij, self.re, self.MB_record                      = random_events(self.t, self.J, self.K, self.X_ijk, self.S_ij, self.C_ij, self.C_j, \
																					   	self.JA_event, self.MB_event, self.MB_record)
		# Snapshot
		self.S_k, self.S_j, self.J, self.I, self.JSet, self.OJSet, self.DSet,    \
		self.ODSet, self.OperationPool, self.n_ops_left_j, self.MC_ji,           \
        self.n_MC_ji, self.d_j, self.n_j, self.p_ijk, self.h_ijk,                \
        self.org_p_ijk, self.org_h_ijk, self.org_n_j, self.org_MC_ji,            \
        self.org_n_MC_ji, self.Oij_on_machine, self.X_ijk, self.S_ij, self.C_ij, \
		self.C_j, self.JA_boolean, self.JA_long_boolean, self.JA_urgent_boolean, \
        self.MB_boolean, self.MB_critical_boolean, self.sum_re,                  \
        self.CT_k, self.T_cur, self.Tard_job, self.NewJobList            = snapshot(self.t, self.triggered_event, self.MC_ji, self.n_MC_ji,                 \
																					self.d_j, self.n_j, self.p_ijk, self.h_ijk, self.J, self.I, self.K,  	\
																					self.X_ijk, self.S_ij, self.C_ij, self.OperationPool, self.re, self.S_k,\
																					self.org_J, self.org_p_ijk, self.org_h_ijk, self.org_n_j,               \
																					self.org_MC_ji, self.org_n_MC_ji, self.C_j                              )																				

		# --------------------------------- Terminated, Reward,  Observation  ------------------------------------
		if self.t >= np.max(self.C_ij): 
			self.done = True
		
		self.calc_reward()
		self.calc_observation()

		return self.observation, self.reward, self.done, False, {}
	

	"""############################################### R E S E T ##################################################"""

	def reset(self, seed=None):
		if seed is not None:
			self.seed(seed)

		super().reset(seed=seed)
		# Reset input
		self.t                  	   = 0
		if self.fixed_instance == True:
			self.load_instance('fixed_instance')
			if self.fixed_scenario == True:
				self.load_scenario('fixed_scenario')
			else:
				self.load_scenario(random.choice(self.list_scenario))
		else:
			if self.num_scenario_per_instance >= 10:
				self.num_scenario_per_instance = 0
				self.instance_id 			   = random.choice(self.list_instance)
			self.num_scenario_per_instance +=1
			self.load_instance(self.instance_id)
			self.JA_event, self.MB_event = generate_random_event(self.J, self.K, self.planning_horizon, self.WeibullDistribution, 
																self.critical_machines, self.ReworkProbability)

		self.events         = {}
		self.S_k            = np.zeros((self.K))
		self.MB_record      = {}
		self.done 			= False
		

		# ------------------------------------------ State transition ------------------------------------------
		# Random event
		self.JA_event, self.MB_event, self.t, self.triggered_event, \
		self.affected_Oij, self.re, self.MB_record                      = random_events(self.t, self.J, self.K, self.X_ijk, self.S_ij, self.C_ij, self.C_j, \
																					   	self.JA_event, self.MB_event, self.MB_record)

		# Snapshot
		self.S_k, self.S_j, self.J, self.I, self.JSet, self.OJSet, self.DSet,    \
		self.ODSet, self.OperationPool, self.n_ops_left_j, self.MC_ji,           \
        self.n_MC_ji, self.d_j, self.n_j, self.p_ijk, self.h_ijk,                \
        self.org_p_ijk, self.org_h_ijk, self.org_n_j, self.org_MC_ji,            \
        self.org_n_MC_ji, self.Oij_on_machine, self.X_ijk, self.S_ij, self.C_ij, \
		self.C_j, self.JA_boolean, self.JA_long_boolean, self.JA_urgent_boolean, \
        self.MB_boolean, self.MB_critical_boolean, self.sum_re,                  \
        self.CT_k, self.T_cur, self.Tard_job, self.NewJobList            = snapshot(self.t, self.triggered_event, self.MC_ji, self.n_MC_ji,                 \
																					self.d_j, self.n_j, self.p_ijk, self.h_ijk, self.J, self.I, self.K,  	\
																					self.X_ijk, self.S_ij, self.C_ij, self.OperationPool, self.re, self.S_k,\
																					self.org_J, self.org_p_ijk, self.org_h_ijk, self.org_n_j,               \
																					self.org_MC_ji, self.org_n_MC_ji, self.C_j                              )															
		self.pre_JSet = copy.deepcopy(self.JSet)	

		# ---------------------------------------------Observation--------------------------------------------
		self.calc_observation()

		return self.observation, {}
	

	
		# self.original_J              = copy.deepcopy(J)
		# self.original_I              = copy.deepcopy(I)
		# self.original_X_ijk          = copy.deepcopy(X_ijk)
		# self.original_S_ij           = copy.deepcopy(S_ij)
		# self.original_C_ij           = copy.deepcopy(C_ij)
		# self.original_C_j            = copy.deepcopy(C_j)
		# self.original_p_ijk          = copy.deepcopy(p_ijk)
		# self.original_d_j            = copy.deepcopy(d_j)
		# self.original_n_j            = copy.deepcopy(n_j)
		# self.original_MC_ji          = copy.deepcopy(MC_ji)
		# self.original_n_MC_ji        = copy.deepcopy(n_MC_ji)
		# self.original_h_ijk          = copy.deepcopy(h_ijk)
		# self.original_OperationPool  = copy.deepcopy(OperationPool)
		# self.original_JA_event       = copy.deepcopy(JA_event)
		# self.original_MB_event       = copy.deepcopy(MB_event)
		
		# self.LB_Cmax                 = np.sum(np.mean(self.p_ijk*self.h_ijk, axis= 2))


		# if self.fixed_instance == True:
		# 	self.J 	  		    = copy.deepcopy(self.original_J)
		# 	self.I 			    = copy.deepcopy(self.original_I)
		# 	self.X_ijk 			= copy.deepcopy(self.original_X_ijk)
		# 	self.S_ij 			= copy.deepcopy(self.original_S_ij)
		# 	self.C_ij 			= copy.deepcopy(self.original_C_ij)
		# 	self.C_j            = copy.deepcopy(self.original_C_j)
		# 	self.p_ijk 			= copy.deepcopy(self.original_p_ijk)
		# 	self.d_j 			= copy.deepcopy(self.original_d_j)
		# 	self.n_j 			= copy.deepcopy(self.original_n_j)
		# 	self.MC_ji 			= copy.deepcopy(self.original_MC_ji)
		# 	self.n_MC_ji 		= copy.deepcopy(self.original_n_MC_ji)
		# 	self.h_ijk 			= copy.deepcopy(self.original_h_ijk)
		# 	self.OperationPool 	= copy.deepcopy(self.original_OperationPool)

		# 	self.org_J          = copy.deepcopy(self.original_J)
		# 	self.org_p_ijk      = copy.deepcopy(self.original_p_ijk)
		# 	self.org_h_ijk      = copy.deepcopy(self.original_h_ijk)
		# 	self.org_n_j        = copy.deepcopy(self.original_n_j)
		# 	self.org_MC_ji      = copy.deepcopy(self.original_MC_ji)
		# 	self.org_n_MC_ji    = copy.deepcopy(self.original_n_MC_ji)

		# 	if self.fixed_scenario == True:
		# 		self.JA_event   = copy.deepcopy(self.original_JA_event)
		# 		self.MB_event   = copy.deepcopy(self.original_MB_event)
		# 	else:
		# 		self.JA_event, self.MB_event = generate_random_event(self.J, self.planning_horizon)


		# else:

		# 	self.JA_event, self.MB_event = generate_random_event(self.J, self.planning_horizon)

		# J, I, K, X_ijk, S_ij, C_ij, C_j, 
		# 		p_ijk, d_j, n_j, MC_ji, n_MC_ji, h_ijk, OperationPool, 
		# 		JA_event, MB_event, 
