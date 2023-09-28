import numpy as np
import math
from agent import Agent
from utils import *
import control as ct
import os 
import pickle

from scipy import linalg as la
from numpy.linalg import det

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
        self.Atilde = []
        self.Btilde = []
        self.w_covs = []
        self.v_covs = []
        self.Hi = []
        
        for i in range(self.N):
            tmp_args = args.copy()
            tmp_args['id'] = i
            tmp_agent = Agent(tmp_args)   
            self.agents.append(tmp_agent)         
            self.Mbar.append(tmp_agent.Mi)
            self.Atilde.append(tmp_agent.A)
            self.Btilde.append(tmp_agent.B)
            self.w_covs.append(tmp_agent.w_cov)
            self.v_covs.append(tmp_agent.v_cov)
            self.Hi.append(tmp_agent.Hi)
            
               
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
            if np.allclose(loaded_v_cov, self.v_covs) and np.allclose(loaded_w_cov, self.w_covs) and np.allclose(adj_, self.adj):
                data_load = True        
            else:
                data_load = False
                
        if data_load:
            self.lqr_gain = self.data['lqr_gain']
            self.est_gains = self.data['est_gains']            
        else:
            self.lqr_gain = self.compute_lpr_gain()
            # enforcing the gain is in the subspace
            self.lqr_gain = (self.lqr_gain * self.F_filter_mtx)
            self.est_gains = self.compute_est_gain(self.lqr_gain)
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
                'adj_matrix' : self.adj}
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

    def compute_phi(self,F):
        Fbar = np.kron(np.eye(self.N), F)        
        
        tmp = np.kron(np.ones([self.N,1]),np.dot(np.dot(self.Bbar, self.Mbar), Fbar))           
        
        phi = np.kron(np.eye(self.N), (self.Atilde + np.dot(self.Btilde, F))) - tmp
        return phi
    def compute_LC(self,p_cov):
        S = []
        L = []
        # Calculate Kalman gains
        for i in range(self.N):
            cov_range = slice(self.N * self.n * i, self.N * self.n * (i + 1))
            S_i = self.Hi[i] @ (p_cov[cov_range, cov_range] + self.v_covs[cov_range, cov_range]) @ self.Hi[i].T
            slice_length = cov_range.stop - cov_range.start
            S_i_inv = la.solve(S_i, np.eye(S_i.shape[0]))
            L_i = p_cov[cov_range, cov_range] @ self.Hi[i].T @ S_i_inv
            S.append(S_i.copy())
            L.append(L_i.copy())
        LC_size = self.N * L[0].shape[0]
        # Initialize LC matrix with zeros
        LC = np.zeros((LC_size, LC_size))
        # Calculate LC matrix
        for i in range(self.N):
            LC_range = slice(i * self.n * self.N, (i + 1) * self.n * self.N)
            LC[LC_range, LC_range] = L[i] @ self.Hi[i]
        return LC, L
    
    def compute_est_gain(self,F):
        phi = self.compute_phi(F)        
        cov = np.eye(self.n*self.N*self.N)  
        max_iter = 200
        opt_L = None
        for i in range(max_iter):              
            p_cov = np.dot(np.dot(phi,cov),phi.T) + np.kron(np.ones([self.N,self.N]),self.w_covs)
            LC, L = self.compute_LC(p_cov)
            # Calculate e_cov_next
            eye_lc = np.eye(len(LC))
            jitter = 1e-6
            e_cov_next = (eye_lc - LC) @ p_cov @ (eye_lc - LC).T + LC @ self.v_covs @ LC.T+ jitter * eye_lc
            
            k = cov.shape[0]
            e_cov_next_inv = la.solve(e_cov_next, np.eye(e_cov_next.shape[0]))
            tmp = np.dot(e_cov_next_inv, cov)
            dKL = 0.5 * (np.trace(tmp) - k - np.log(det(tmp)))
            # print("iter i = " + str(i) + " , dKL = " + str(dKL))
            opt_L = L
            if dKL < 0.01:
                break 
            cov = e_cov_next.copy()
            
        return opt_L
        








        
        
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
