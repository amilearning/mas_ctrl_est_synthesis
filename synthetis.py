import numpy as np
import math
from agent import Agent
from utils import *
import control as ct
import os 
import pickle

enable_debug_mode = False

class ControlEstimationSynthesis:
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
        self.Abar = []
        self.Bbar = []
        self.w_covs = []
        self.v_covs = []
        
        for i in range(self.N):
            tmp_args = args.copy()
            tmp_args['id'] = i
            tmp_agent = Agent(tmp_args)   
            self.agents.append(tmp_agent)         
            self.Mbar.append(tmp_agent.Mi)
            self.Abar.append(tmp_agent.A)
            self.Bbar.append(tmp_agent.B)
            self.w_covs.append(tmp_agent.w_cov)
            self.v_covs.append(tmp_agent.v_cov)
               
        self.Abar = block_diagonal_matrix(self.Abar)        
        self.Bbar = block_diagonal_matrix(self.Bbar)        
        self.Mbar = block_diagonal_matrix(self.Mbar)
        self.w_covs = block_diagonal_matrix(self.w_covs)
        self.v_covs = block_diagonal_matrix(self.v_covs)
        if enable_debug_mode:
            display_array_in_window(self.Abar)
            display_array_in_window(self.Bbar)
            display_array_in_window(self.Mbar)        
            display_array_in_window(self.w_covs)
            display_array_in_window(self.v_covs)
        
                
        self.gains = None
        if self.load_gains():
            self.lqr_gain = self.gains['lqr_gain']
        else:
            self.lqr_gain = self.compute_lpr_gain()
            # enforcing the gain is in the subspace
            self.lqr_gain = (self.lqr_gain * self.F_filter_mtx)
            self.save_gains()
    
            
    def load_gains(self,file_name = None):             
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  # Create a 'data' directory in the same folder as your script
        if file_name is None:
            file_name = 'gains.pkl'
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.gains = pickle.load(file)
            print(f"File '{file_path}' loaded.")
            return True
        else:
            return False
    
    
    def save_gains(self,file_name = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  # Create a 'data' directory in the same folder as your script
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data = {'lqr_gain': self.lqr_gain}
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
                K_, S_, E_ =ct.dlqr(self.Abar, self.Bbar, self.Q_tilde, self.R_tilde)        
                break  # Exit the loop if successful
            except:            
                print(f"Attempt {attempt + 1}: Failed to solve DARE")                                
                self.R_tilde +=np.diagflat(np.random.rand(self.R_tilde.shape[0])*0.0001)                
                if attempt == max_attempts - 1:
                    print("Maximum attempts reached. Unable to solve DARE.")
                    
                  
        return K_

 
        
    def set_MAS(self,info):
        self.N = info["N"]
        self.v_stds = info["v_stds"]
        
   
# if __name__ == "__main__":
#     args = {}
#     args['Ts'] = 0.1
#     N_agent = 5
#     args['N'] = N_agent
#     args['w_std'] = 0.1 # w std for each agent 
#     args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
#     # args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
#     args['c'] = np.array([[1,1,0,0,0],
#                           [1,1,1,0,0],
#                           [0,1,1,1,0],
#                           [0,0,1,1,1],
#                           [0,0,0,1,1]])
#     args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix     
#     args['n'] = 4
#     args['p'] = 2
#     args['Q'] = np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
#     args['R'] = np.eye(N_agent)
    
#     obj = ControlEstimationSynthesis(args)
