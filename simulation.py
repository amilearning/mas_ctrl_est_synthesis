import numpy as np
import math
from agent import Agent
from utils import *
import control as ct
from synthetis import ControlEstimationSynthesis
import threading
import concurrent.futures
from eval import MASEval
enable_debug_mode = False


    
    
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
        sim_step = args['sim_n_step']
        self.offset = get_formation_offset_vector(self.N, self.n, dist = 4.0)
        self.synthesis = ControlEstimationSynthesis(args)
        self.eval = MASEval(args)
        self.stage_costs = []
        
        
        self.agents = self.synthesis.agents
        self.init_MAS()        
        self.X = np.zeros([self.N*self.n,1])
        self.run_sim(sim_step)
        self.eval_ready() 
        
        
        
        

    def agent_step(agent, shared_data):
        agent.step( shared_data)
        
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
                    futures.append(executor.submit(agent.set_measurement(self.X)))
                    
                # Wait for all agent step functions to complete
                concurrent.futures.wait(futures)
                if time_step ==0 :
                    for i in range(self.N):
                        self.agents[i].set_MAS_info(self.synthesis.Atilde,self.synthesis.Btilde, self.synthesis.w_covs, self.synthesis.v_covs)
                        self.agents[i].set_xhat(self.X)
                        
                
                ########### Estimation  ######################                                                  
                futures = []
                for agent in self.agents:
                    futures.append(executor.submit(agent.est_step()))                                                    
                concurrent.futures.wait(futures)                                                
                ###########   ######################                                                  
                
                ########### Dynamics prpogate ###########                                          
                futures = []
                for agent in self.agents:
                    futures.append(executor.submit(agent.step()))                                                    
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
        
        
    def init_MAS(self):
        for i in range(self.N):
            tmp_state = np.random.randn(4,1)
            self.agents[i].set_x(tmp_state)
            self.agents[i].set_gain(self.synthesis.lqr_gain)
            self.agents[i].set_est_gain(self.synthesis.est_gains[i])
            self.agents[i].set_offset(self.offset)

                
   
if __name__ == "__main__":
    args = {}
    args['Ts'] = 0.1
    N_agent = 20    
    args['N'] = N_agent
    args['w_std'] = 0.1 # w std for each agent 
    args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
    # args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
    args['c'] = get_chain_adj_mtx(N_agent) 
    # args['c'] = np.array([[1,1,0,0,0],
    #                       [1,1,1,0,0],
    #                       [0,1,1,1,0],
    #                       [0,0,1,1,1],
    #                       [0,0,0,1,1]])
    args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix     
    args['n'] = 4
    args['p'] = 2
    args['Q'] = np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
    args['R'] = np.eye(N_agent)
    args['sim_n_step'] = 100
    
    obj = MASsimulation(args)
    obj.eval.eval_init()
