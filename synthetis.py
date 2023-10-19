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

        self.Mibar = [] ## set of filter matrix for inputs from the entire MAS \in R^{Np x Np}
        self.ci = [] ## set of filter matrix for states from the entire MAS \in R^{nN x nN }
    
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
            self.Mibar.append(tmp_agent.Mibar)
            self.ci.append(tmp_agent.ci)
            self.Mbar.append(tmp_agent.Mibar)            
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
        ############### hard FAlse
        ############### hard FAlse
        data_load = False
        ############### hard FAlse
        ############### hard FAlse

        if data_load:
            loaded_w_cov = self.data['w_covs']
            loaded_v_cov = self.data['v_covs']
            adj_ = self.data['adj_matrix']
            # if self.data['args'] == self.args and np.allclose(loaded_v_cov, self.v_covs) and np.allclose(loaded_w_cov, self.w_covs) and np.allclose(adj_, self.adj):                
            #     data_load = True        
            # else:
            #     data_load = False
            
                
        if data_load:
            self.lqr_gain = self.data['lqr_gain']
            self.est_gains = self.data['est_gains']  
            self.opt_gain = self.data['opt_gain']          
        else:
            self.lqr_gain = self.compute_lpr_gain()            
            self.opt_gain = self.ctrl_est_synthesis(init_gain_guess = self.lqr_gain)
            
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
                'opt_gain': self.opt_gain,
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

    def riccati_update(self,P,Q,R,F):
        a_bf = self.Atilde+np.dot(self.Btilde,F)        
        return Q+np.dot(F.T,np.dot(R,F)) + np.dot(a_bf.T,np.dot(P,a_bf))         
        
    def lqr_F_update(self,P):
        tmp = self.R_tilde + np.dot(np.dot(self.Btilde.T, P), self.Btilde)
        tmp_inv = la.solve(tmp, np.eye(tmp.shape[0]))
        sol = -1*np.dot(np.dot(np.dot(tmp_inv,self.Btilde.T),P),self.Atilde)
        return sol
    
    def prev_max_F_update(self,P):
        # solve via symvester equatoin
        #   AXB + CXD = E
        #  = -B^T P A Sig 
        # MRM F Sigij + 
        # R F Sigij +
        # B^TPB F Sigij 
        Atmp = []
        Btmp = []
        # F = matrixEquationSolver(A,B,C)
        for i in range(self.N):
            for j in range(self.N):
                Atmp_ = np.dot(np.dot(self.Mibar[i].T,(self.R_tilde + np.dot(np.dot(self.Btilde.T,P), self.Btilde))),self.Mibar[j])
                Atmp.append(Atmp_)
                # Btmp_ = np.eye(self.sigmas(i,j).shape[0])
                Btmp_ = self.sigmas(i,j)
                Btmp.append(Btmp_.T)

        for i in range(self.N):
            for j in range(self.N):
                Atmp_ = np.dot(np.dot(self.Mibar[i].T,(self.R_tilde + np.dot(np.dot(self.Btilde.T,P), self.Btilde))),self.Mibar[j])
                Atmp.append(Atmp_)
                Btmp_ = np.eye(self.sigmas(i,j).shape[0])
                Btmp.append(Btmp_.T)
        
        tmp = np.dot(np.dot(np.dot(self.Btilde.T, P),self.Atilde),self.sigmas(0,0))
        rrmtx = np.zeros(tmp.shape)
        for i in range(self.N):
            for j in range(self.N):                
                rrmtx +=  np.dot(np.dot(np.dot(self.Btilde.T, P),self.Atilde),self.sigmas(i,j))
        rrmtx = -1*np.dot(np.dot(self.Btilde.T, P),self.Atilde)    
        updated_F = matrixEquationSolver(Atmp,Btmp,rrmtx)
        return updated_F


    def ctrl_synthesis(self):
        roll_P = np.eye(self.Q_tilde.shape[0]) # self.Q_tilde
        roll_F = np.zeros(self.lqr_gain.shape)
        max_iter = 100
        for out_ier in range(max_iter):
            # roll_F =self.lqr_F_update(roll_P) ## original LQR derivative
            roll_F = self.prev_max_F_update(roll_P)
            next_P = self.riccati_update(roll_P,self.Q_tilde,self.R_tilde,roll_F)            
            opt_F = roll_F.copy()
            opt_P = next_P.copy()
            f_norm = np.linalg.norm((next_P-roll_P), 'fro')
            print("f_norm = " + str(f_norm))
            if f_norm < 1e-2:
                opt_F =self.lqr_F_update(next_P.copy()) ## original LQR derivative
                break 
            else:
                roll_P = next_P.copy()
        return opt_F

    def ctrl_est_synthesis(self,init_gain_guess = None):
        
        max_iter = 100
        roll_F = init_gain_guess.copy()
        for out_iter in range(max_iter):
            self.est_gains, self.est_covs = self.compute_est_gain(roll_F)
            self.sigmas = lambda i, j: self.est_covs[i * self.n*self.N: (i + 1) * self.n*self.N, j * self.n*self.N: (j + 1) * self.n*self.N]
            opt_F = self.ctrl_synthesis()
            out_f_norm = np.linalg.norm((opt_F-roll_F), 'fro')
            print("out iter f_norm = " + str(out_f_norm))
            if out_f_norm < 1e-1:
                break
            else:
                roll_F = opt_F.copy()
        
        self.est_gains, self.est_covs = self.compute_est_gain(opt_F)
        self.sigmas = lambda i, j: self.est_covs[i * self.n*self.N: (i + 1) * self.n*self.N, j * self.n*self.N: (j + 1) * self.n*self.N]
            
        return opt_F
    

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
                    
        K = -1*K_
        # enforcing the gain is in the subspace
        # K = (K* self.F_filter_mtx)
        return K

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
            
        return opt_L, cov
        








        
        
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
