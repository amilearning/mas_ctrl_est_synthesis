import numpy as np
import math
from agent2 import Agent2
from utils import *
import control as ct
from synthetis import DistributedCommunication
import threading
import concurrent.futures
from eval import MASEval
enable_debug_mode = False

import numpy as np
import math
from agent2 import Agent2
from utils import *
import control as ct
import os 
import pickle

from scipy import linalg as la
from numpy.linalg import det

enable_debug_mode = False

class DistributedCommunication:
    # Constructor method (optional)
    def __init__(self, args):
  
        # Initialize instance variables here
        self.args = args
        self.Ts = args['Ts']
        self.N = args['N']
        self.w_std = args['w_std']
        self.v_std = args['v_std']
        self.L = args['L']
        self.N = args['N']
        self.n = args['n']
        self.p = args['p']
        self.Q = args['Q']
        self.R = args['R']
        self.adj = args['c']        
        self.F_filter_mtx = np.kron(self.adj,np.ones([self.p,self.n]))

        self.Q_tilde = np.kron(self.Q,np.eye(self.n))        
        self.R_tilde = np.kron(self.R, np.eye(self.p))             
        self.agents = []
        
        self.Mbar = []
        self.Atilde = []
        self.Btilde = []
        self.w_covs = []
        self.v_covs = []
        self.Hi = []
        self.C = []
        
        for i in range(self.N):
            tmp_args = args.copy()
            tmp_args['id'] = i
            tmp_agent = Agent2(tmp_args)               
            self.agents.append(tmp_agent)         
            self.Mbar.append(tmp_agent.Mi)
            self.Atilde.append(tmp_agent.A)
            self.Btilde.append(tmp_agent.B)
            self.w_covs.append(tmp_agent.w_cov)
            self.v_covs.append(tmp_agent.v_cov)
            self.C.append(tmp_agent.C[i])

        self.Hi = tmp_agent.Hi
        self.Atilde = block_diagonal_matrix(self.Atilde)        
        self.Btilde = block_diagonal_matrix(self.Btilde)        
        self.Bbar = np.kron(np.ones([1,self.N]),self.Btilde)
        self.Mbar = block_diagonal_matrix(self.Mbar)
        self.w_covs = block_diagonal_matrix(self.w_covs)
        self.v_covs = block_diagonal_matrix(self.v_covs)
        if enable_debug_mode:
            display_array_in_window(self.Atilde)
            display_array_in_window(self.Btilde)
            display_array_in_window(self.Mbar)        
            display_array_in_window(self.w_covs)
            display_array_in_window(self.v_covs)
        
                
        self.data = None
        data_load = self.load_gains()   
        if data_load:
            loaded_w_cov = self.data['w_covs']
            loaded_v_cov = self.data['v_covs']
            adj_ = self.data['adj_matrix'] 
            
                
        if data_load:
            self.lqr_gain = self.data['lqr_gain']
            self.est_gains = self.data['est_gains']            
        else:
            self.lqr_gain = self.compute_lpr_gain()
            # enforcing the gain is in the subspace
            self.lqr_gain = (self.lqr_gain * self.F_filter_mtx)
            self.est_gains = self.compute_est_gain()
            self.save_gains()
            
            
    
            
    def load_gains(self,file_name = None):             
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  # Create a 'data' directory in the same folder as your script
        if file_name is None:
            file_name = 'gains.pkl'
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.data = pickle.load(file)                
            # print(f"File '{file_path}' loaded.")
            return True
        else:
            return False
    
    
    def save_gains(self,file_name = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  # Create a 'data' directory in the same folder as your script
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data = {'lqr_gain': self.lqr_gain,
                'est_gains': self.est_gains,
                'w_covs': self.w_covs,
                'v_covs': self.v_covs,
                'adj_matrix' : self.adj,
                'args':self.args}
        if file_name is None:
            file_name = 'gains.pkl'
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
            print(f"File '{file_path}' Saved.")

        
    def compute_lpr_gain(self):        
        max_attempts = 100  # Maximum number of attempts
        K_ = None
        for attempt in range(max_attempts):
            try:
                K_, S_, E_ =ct.dlqr(self.Atilde, self.Btilde, self.Q_tilde, self.R_tilde)        
                break  # Exit the loop if successful
            except:            
                print(f"Attempt {attempt + 1}: Failed to solve DARE")                                
                self.R_tilde +=np.diagflat(np.random.rand(self.R_tilde.shape[0])*0.0001)                
                if attempt == max_attempts - 1:
                    print("Maximum attempts reached. Unable to solve DARE.")
                    
                  
        return -1*K_

    

    def compute_est_gain(self, dt=0.01, max_iterations=15000):
        opt_LL = []
        Q_inv = np.linalg.inv(np.eye(self.n * self.N))  # Assuming Q is identity as before
        
        for i in range(self.N):
            self.R_tildetilde = np.kron(self.R_tilde, np.eye(self.p))
            A = self.Atilde
            C = self.C[i]
            P = np.eye(self.n * self.N)
            
            for iteration in range(max_iterations):
                P_dot = A @ P + P @ A.T - P @ C.T @ Q_inv @ C @ P + self.R_tildetilde
                P = P + P_dot * dt
                
            opt_L = P @ C.T @ np.linalg.inv(self.R_tildetilde)
            opt_LL.append(opt_L)
        return opt_LL



    def set_MAS(self,info):
        self.N = info["N"]
        self.v_stds = info["v_stds"]

   

class MASsimulation:
    # Constructor method (optional)
    def __init__(self, args):
        # Initialize instance variables here
        self.args = args
        self.Ts = args['Ts']
        self.N = args['N']
        self.w_std = args['w_std']
        self.v_std = args['v_std']
        self.L = args['L']
        self.N = args['N']
        self.n = args['n']
        self.p = args['p']
        self.Q = args['Q']
        self.R = args['R']
        self.adj = args['c'] 
        self.gamma = 100
        sim_step = args['sim_n_step']
        self.offset = get_formation_offset_vector(self.N, self.n, dist = 4.0)
        self.synthesis = DistributedCommunication(args)
        self.eval = MASEval(args)
        self.stage_costs = []
        self.C =[]
        
        self.agents = self.synthesis.agents
        self.C = self.synthesis.C
        self.init_MAS()        
        self.X = np.zeros([self.N*self.n,1])
        self.run_sim(sim_step)
        self.eval_ready() 
        
        
    def agent_step(agent, shared_data):
        agent.step(shared_data)
        
    def compute_cost(self,x,u):
        return np.dot(np.dot(np.transpose(x-self.offset),self.synthesis.Q_tilde),x-self.offset) + np.dot(np.dot(np.transpose(u),self.synthesis.R_tilde),u)
        
    def run_sim(self, num_time_steps):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.N) as executor:
            for time_step in range(num_time_steps):

                ## set measurements 
                self.get_states()
                self.get_inputs()
                stage_cost = self.compute_cost(self.X, self.U)                
                self.eval.add_stage_cost(stage_cost)
                futures = []
                for agent in self.agents:
                    futures.append(executor.submit(agent.set_measurement,self.X))
                    
                # Wait for all agent step functions to complete
                concurrent.futures.wait(futures)
                if time_step ==0 :
                    for i in range(self.N):
                        self.agents[i].set_MAS_info(self.synthesis.Atilde,self.synthesis.Btilde, self.synthesis.w_covs, self.synthesis.v_covs)
                        self.agents[i].set_xhat(self.X)
                        
                
                ########### Estimation  ######################                                                  
                futures = []
                for agent in self.agents:
                    futures.append(executor.submit(agent.est_step))
                concurrent.futures.wait(futures)  
                self.Est = []
                for agent, future in zip(self.agents, futures):
                    agent.xhat = future.result()
                    self.Est.append(agent.xhat)
                
                self.est_consensus(self.Est.copy(), self.gamma)
                concurrent.futures.wait(futures)   
                futures = []
                for i, agent in enumerate(self.agents):
                    futures.append(executor.submit(agent.set_xhat,self.Est[i]))                                       
                concurrent.futures.wait(futures)                                                
                ###########   ######################                                                  
                
                ########### Dynamics prpogate ###########                                          
                futures = []
                for i, agent in enumerate(self.agents):
                    futures.append(executor.submit(agent.step))                                                    
                concurrent.futures.wait(futures)                                
                ########### ################## ##########
        
   
    def eval_ready(self):
        trajs = []
        est_trajs = []
        for i in range(self.N):
            tmp_traj = self.agents[i].get_traj()
            est_traj = self.agents[i].get_est_traj()
            trajs.append(tmp_traj)
            est_trajs.append(est_traj)
        self.eval.trajs = trajs
        self.eval.est_trajs = est_trajs
        
        # self.eval.eval_init()

    def get_inputs(self):
        input_vector = []
        for i in range(self.N):
            input_tmp = self.agents[i].get_input()
            input_vector.append(input_tmp)
        self.U = np.vstack(input_vector)  
        
        
    def get_states(self):
        gt_state_vector = []
        for i in range(self.N):
            gt_state = self.agents[i].get_x()
            gt_state_vector.append(gt_state)
        self.X = np.vstack(gt_state_vector)        

    def get_est(self):
        gt_est_vector = []
        for i in range(self.N):
            gt_est = self.agents[i].est_step(self.C)
            gt_est_vector.append(gt_est)
        self.Est = gt_est_vector   
        
    def est_consensus(self,current_est_agents,gamma):
        for _ in range(gamma):
            for i, agent_i in enumerate(current_est_agents):
                for j, agent_j in enumerate(current_est_agents):
                    if self.adj[i, j] == 1:
                        current_est_agents[i] = current_est_agents[i] + 0.3*(agent_j-agent_i)
        self.Est = current_est_agents.copy()
    
    def est_consensus1(self,current_est_agents, gamma):
        for _ in range(gamma):
            next_est_agents = current_est_agents.copy()
            for i, agent_i in enumerate(current_est_agents):    
                for j, agent_j in enumerate(current_est_agents):
                    if self.adj[i, j] == 1:
                        next_est_agents[i] = current_est_agents[i] + 0.1*(agent_j-agent_i)
            current_est_agents = next_est_agents.copy()
        self.Est = current_est_agents.copy()

    def est_consensus3(self, current_est_agents, gamma):
        for _ in range(gamma):
            next_est_agents = current_est_agents.copy()
            for i, agent_i in enumerate(current_est_agents):    
                for j, agent_j in enumerate(current_est_agents):
                    if self.adj[i, j] == 1:
                        next_est_agents[i] += 0.1 * (agent_j - agent_i)
            current_est_agents = next_est_agents.copy()
        self.Est = current_est_agents.copy()

    def est_consensus2(self, current_est_agents, gamma):
        for _ in range(gamma):
            next_est_agents = current_est_agents.copy()
            for i, agent_i in enumerate(current_est_agents):    
                for j, agent_j in enumerate(current_est_agents):
                    if self.adj[i, j] == 1:
                        difference = np.array(agent_j) - np.array(agent_i)
                        # Set values in the difference array to 0.0 if their absolute values are less than 0.01
                        difference[np.abs(difference) < 0.01] = 0.0
                        next_est_agents[i] += 0.3 * difference
            current_est_agents = next_est_agents.copy()
        self.Est = current_est_agents.copy()
 
    def init_MAS(self):
        for i in range(self.N):
            tmp_state = np.random.randn(self.n,1)
            self.agents[i].set_x(tmp_state)
            self.agents[i].set_gain(self.synthesis.lqr_gain)
            self.agents[i].set_est_gain(self.synthesis.est_gains[i])
            self.agents[i].set_offset(self.offset)



if __name__ == "__main__":
    args = {}
    args['Ts'] = 0.1
    N_agent = 5
    args['N'] = N_agent
    args['w_std'] = 0.1 # w std for each agent 
    args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
    # args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
    args['c'] = get_chain_adj_mtx(N_agent)
    #args['c'] = np.array([[1,1,1,0,0],
    #                      [1,1,0,1,0],
    #                      [1,0,1,0,1],
    #                      [0,1,0,1,0],
    #                      [0,0,1,0,1]])
    # args['c'] = np.array([[1,1,1,0,0,0,0,0,0],
    #                     [1,1,0,1,0,0,0,0,0],
    #                     [1,0,1,0,1,0,0,0,0],
    #                     [0,1,0,1,0,1,0,0,0],
    #                     [0,0,1,0,1,0,1,0,0],
    #                     [0,0,0,1,0,1,0,1,0],
    #                     [0,0,0,0,1,0,1,0,1],
    #                     [0,0,0,0,0,1,0,1,0],
    #                     [0,0,0,0,0,0,1,0,1]])
    #args['c'] = np.array([[1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1,1]])
    args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix     
    args['n'] = 4
    args['p'] = 2
    #args['Q'] = np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
    args['Q'] = get_laplacian_mtx(args['c'])
    args['R'] = np.eye(N_agent)
    args['sim_n_step'] = 400
    
    obj = MASsimulation(args)
    obj.eval.eval_init()
