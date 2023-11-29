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
        self.ctrl_type = args['ctrl_type']
        self.gamma = args['gamma']
        
        
        self.offset = get_formation_offset_vector(self.N, self.n, dist = 5.0)
        self.synthesis = ControlEstimationSynthesis(args)
     
        self.eval = MASEval(args)
        self.stage_costs = []
        
        
        self.agents = self.synthesis.agents
        self.init_MAS()        
        self.X = np.zeros([self.N*self.n,1])
        if self.ctrl_type == CtrlTypes.COMLQG:
            self.run_com_sim(sim_step)
        else:
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
                        
                if self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
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
                
   
                
    def run_com_sim(self,num_time_steps):
        delta = np.sort(np.linalg.eigvals(self.L))[-1] + 0.01       

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

                # est consensus 
                consensus_mtx = -1*(1/delta)*np.kron(self.L,np.eye(self.N*self.n))                
                roll_xhat = np.vstack([agent.xhat.copy() for agent in self.agents])
                for i in range(self.gamma):
                    tmp = consensus_mtx @ roll_xhat 
                    roll_xhat = roll_xhat + tmp.copy()
                for i in range(self.N):
                    consensus_est = roll_xhat[i*self.N*self.n:(i+1)*self.N*self.n]
                    self.agents[i].set_xhat(consensus_est.copy())
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
            if self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
                est_traj = self.agents[i].get_est_traj()    
                est_trajs.append(est_traj)
            trajs.append(tmp_traj)
            
        self.eval.trajs = trajs
        if self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
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
        self.X = np.vstack(gt_state_vector).copy()        
        
        
    def init_MAS(self):

        if self.ctrl_type == CtrlTypes.LQROutputFeedback:
            gain = self.synthesis.lqr_gain
        elif self.ctrl_type == CtrlTypes.SubOutpFeedback:
            gain = self.synthesis.sub_gain
        elif self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
            gain = self.synthesis.opt_gain
        elif self.ctrl_type == CtrlTypes.COMLQG:
            gain = self.synthesis.lqr_gain
    

        tmp_state = np.random.randn(self.n,1)*1e-3 
        for i in range(self.N):
            self.agents[i].set_x(tmp_state)
            self.agents[i].set_gain(gain)
            if self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
                self.agents[i].set_est_gain(self.synthesis.est_gains[i])
            elif self.ctrl_type == CtrlTypes.COMLQG:
                self.agents[i].set_est_gain(self.synthesis.kalman_gain[i])
            self.agents[i].set_offset(self.offset)

                
   
if __name__ == "__main__":
    args = {}
    args['Ts'] = 0.1
    N_agent = 5
    args['N'] = N_agent
    args['w_std'] = 0.1 # w std for each agent 
    args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
    # args['v_std'][0] = 1.0
    # args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
    args['c'] = get_chain_adj_mtx(N_agent) 
    # args['c'] = get_circular_adj_mtx(N_agent) 
    args['gamma'] = 1
    # args['c'] = np.array([[1,1,0,0,0],
    #                       [1,1,1,0,0],
    #                       [0,1,1,1,0],
    #                       [0,0,1,1,1],
    #                       [0,0,0,1,1]])
    args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix     
    args['n'] = 4
    args['p'] = 2
    args['Q'] = np.kron(args['L'], np.eye(args['n'])) #  np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
    # np.array([[1,0,0,0],
    #                        [0,0,0,0],
    #                        [0,0,1,0],
    #                        [0,0,0,0]])
    # args['Q'] = args['L']
    args['R'] = np.eye(N_agent)
    args['sim_n_step'] = 200
    args['gain_file_name'] = '22'

    # LQROutputFeedback = 0
    # SubOutpFeedback = 1 
    # CtrlEstFeedback = 2
    # COMLQG = 3
    args['ctrl_type'] = CtrlTypes.CtrlEstFeedback

    obj = MASsimulation(args)
    obj.eval.eval_init()
