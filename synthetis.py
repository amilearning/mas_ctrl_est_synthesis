import numpy as np
import math
from agent import Agent
from utils import *
import control as ct
import os 
import pickle

from scipy import linalg as la
from scipy.signal import cont2discrete



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
        self.ctrl_type = args['ctrl_type']    
        self.F_filter_mtx = np.kron(self.adj,np.ones([self.p,self.n]))

        self.Q_tilde = self.Q # np.kron(self.Q,np.eye(self.n))      #  np.kron(self.L, self.Q)   #
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
        file_name_ = None
        if 'gain_file_name' in args:
            file_name_ = args['gain_file_name']
         
        data_load = self.load_gains(file_name=file_name_)   
        ############### hard FAlse
        ############### hard FAlse
        # data_load = False
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
            self.sub_gain = self.data['sub_gain']
            if 'opt_stage_cost' in self.data:
                self.opt_stage_cost = self.data['opt_stage_cost']
        else:                       
            self.lqr_gain = self.compute_lpr_gain()              
            print('lqr solution found')     
            self.sub_gain = self.compute_suboptimal_gain()     
            print('suboptimal solution found')     
            if self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
                self.opt_gain = self.ctrl_est_synthesis(init_gain_guess = self.lqr_gain)
            else:
                self.opt_gain = np.zeros(self.lqr_gain.shape)
                self.est_gains = []
                self.opt_stage_cost = 0

                
            print('opt gains found')     
            
            self.save_gains(file_name=file_name_)
        
        if data_load is False:
            print("gains generated")
        
     
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
                'sub_gain': self.sub_gain,
                'w_covs': self.w_covs,
                'v_covs': self.v_covs,
                'adj_matrix' : self.adj,
                'args':self.args,
                'opt_stage_cost':self.opt_stage_cost}
        if file_name is None:
            file_name = 'gains.pkl'
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
            print(f"File '{file_path}' Saved.")

    def compute_suboptimal_gain(self):                
        ## assume the dynamics are the same for all agents
        ## assume the Q and R is identity matrix 
        Leigs = np.sort(np.linalg.eigvals(self.L))
        end_eig = Leigs[-1]
        second_eig = Leigs[1]
        c = 2/(second_eig+end_eig)
        Ac = np.array([[0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]])

        Bc = np.array([[0, 0],
                    [1, 0],
                    [0, 0],
                    [0, 1]])
        Q = np.eye(self.n)
        R = self.R[0:2,0:2]
        epsilon = 0.0001
        # solve  A.T P + PA + P * (c^2 * end_eig**2 - 2*c*end_eig) BR^-1 B.T P + end_eig Q + epsilon I_qn = 0
        
        # Qbar = end_eig Q + epsilon I_qn  
        Qbar = end_eig * Q + epsilon*np.eye(Q.shape[0])
        # Rbar = -1 *(c^2 * end_eig**2 - 2*c*end_eig) * R^-1
        Rbar = -1* (c**2 * end_eig**2 - 2* c * end_eig)* np.linalg.inv(R)
        P = la.solve_continuous_are(Ac,Bc,Qbar,Rbar)        
        K = -c * np.linalg.inv(R) @  Bc.T @ P
        
        A_tilde_c = Ac + Bc @ K
        sys = cont2discrete((A_tilde_c, np.zeros([4,1]), [],[]), self.Ts, method='zoh')
        A_tilde_d = sys[0]
        original_sys = cont2discrete((Ac, Bc, [],[]), self.Ts, method='zoh')
        Ad = original_sys[0]
        Bd = original_sys[1]
        Kd = np.dot(np.linalg.pinv(Bd), A_tilde_d - Ad)
        gain_d = np.kron(self.L, Kd)

        return gain_d
    
    def riccati_update(self,P,Q,R,F):
        a_bf = self.Atilde+np.dot(self.Btilde,F)        
      
        return Q+np.dot(F.T,np.dot(R,F)) + np.dot(a_bf.T,np.dot(P,a_bf)) 
        
    def lqr_F_update(self,P):
        tmp = self.R_tilde + np.dot(np.dot(self.Btilde.T, P), self.Btilde)
        tmp_inv = la.solve(tmp, np.eye(tmp.shape[0]))
        sol = -1*np.dot(np.dot(np.dot(tmp_inv,self.Btilde.T),P),self.Atilde)
        return sol
    
    def F_derivative_update(self,P,SigBarx):
        # solve via symvester equatoin
        #   AXB + CXD = E
        #  = -B^T P A SigBarx 
        # M(R+B^TPB)M F Sigij + 
        # R F SigBarx +
        # B^TPB F SigBarx 
        Atmp = []
        Btmp = []
        # F = matrixEquationSolver(A,B,C)
        for i in range(self.N):
            for j in range(self.N):
                Atmp_ = self.Mibar[i].T @ (self.R_tilde + self.Btilde.T @ P @  self.Btilde) @ self.Mibar[j]
                Atmp.append(Atmp_.copy())
                # Btmp_ = np.eye(self.sigmas(i,j).shape[0])
                Btmp_ = self.sigmas(i,j)
                Btmp.append(Btmp_.copy())

    
        Atmp_ = (self.R_tilde + self.Btilde.T @ P @  self.Btilde)
        Atmp.append(Atmp_.copy())
        Btmp_ = SigBarx
        Btmp.append(Btmp_.copy())
        
        rrmtx = -1 * self.Btilde.T @ P @ self.Atilde @ SigBarx     
        updated_F = matrixEquationSolver(Atmp,Btmp,rrmtx)
        
        return updated_F

    def update_estimation_gains(self,roll_F):
        self.est_gains, self.est_covs = self.compute_est_gain(roll_F)
        self.sigmas = lambda i, j: self.est_covs[i * self.n*self.N: (i + 1) * self.n*self.N, j * self.n*self.N: (j + 1) * self.n*self.N]
        self.sigmaBarx = self.compute_sigmaBarx(roll_F, self.est_covs)
        
    def ctrl_est_synthesis(self,init_gain_guess):        
        roll_P = np.eye(self.Q_tilde.shape[0]) # self.Q_tilde
        roll_F = init_gain_guess
        prev_F = roll_F.copy()
        out_max_iter = 30
        in_max_iter = 300
        out_check_count = 0
        stage_cost_list = []
        P_norm_diff_list = []
        opt_P_list = []
        opt_F_list = []
         
        self.update_estimation_gains(roll_F)   
        for out_ier in range(out_max_iter):
            print("outer iter = {}".format(out_ier))                       
            for in_ier in range(in_max_iter):                                  
                if self.ctrl_type == CtrlTypes.LQGFeedback:
                    roll_F =self.lqr_F_update(roll_P) ## original LQR derivativeW         
                else:
                    roll_F = self.F_derivative_update(roll_P, self.sigmaBarx)                                                                 
                next_P = self.riccati_update(roll_P,self.Q_tilde,self.R_tilde,roll_F)            
                opt_F, opt_P = roll_F.copy(), next_P.copy()                
                p_norm_diff = np.linalg.norm((next_P-roll_P), 'fro')
                # p_norm_diff = compute_mahalanobix_dist(next_P, roll_P)            
                P_norm_diff_list.append(p_norm_diff)
                # print("p_norm_diff = " + str(p_norm_diff))
                if p_norm_diff < 1e-3:             
                    break 
                else:
                    roll_P = next_P.copy()
            self.update_estimation_gains(opt_F)  
            stage_cost_tmp = self.compute_stage_cost(opt_P, opt_F, self.sigmaBarx)
            stage_cost_list.append(stage_cost_tmp)
            opt_F_list.append(opt_F)
            opt_P_list.append(opt_P)
            if len(stage_cost_list) > 1:
                stage_cost_diff = stage_cost_list[-2] - stage_cost_tmp
                if stage_cost_diff < 1e-2:                    
                    print("stage_cost_diff less then threshold")
                    print(stage_cost_list) 
                    out_check_count+=1
            if out_check_count > 1:    
                print("outer loop break")            
                break

       
            
        print(stage_cost_list)      
        min_idx = np.argmin(np.array(stage_cost_list))
        opt_F = opt_F_list[min_idx]
        opt_P = opt_P_list[min_idx]
        self.update_estimation_gains(opt_F)        
        self.opt_stage_cost = self.compute_stage_cost(opt_P, opt_F, self.sigmaBarx)
        print("optimized stage cost = {:.3f}".format(self.opt_stage_cost))
    

                  
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
        print("Trace(PW) of LQR = {}".format(np.trace(S_@self.w_covs)))
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
            jitter = 1e-4
            S_i_jitter = S_i + np.eye(len(S_i))*jitter
            S_i_inv = la.solve(S_i_jitter, np.eye(S_i_jitter.shape[0]))
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
            jitter = 1e-5
            e_cov_next = (eye_lc - LC) @ p_cov @ (eye_lc - LC).T + LC @ self.v_covs @ LC.T+ jitter * eye_lc
            # cov_diff = np.linalg.norm((e_cov_next-cov), 'fro')
            cov_diff = compute_mahalanobix_dist(e_cov_next, cov)            
            # print("cov_diff = {}".format(cov_diff))
            opt_L = L.copy()            
            cov = e_cov_next.copy()
            if cov_diff < 1e-2:                
                break 

        return opt_L, cov
        
    def compute_sigmaBarx(self,F,cov):
        steady_cov = cov.copy()
        roll_cov = np.eye(self.n*self.N)*1e-4
        max_iter = 200
        Fbar = np.kron(np.eye(self.N), F)                        
        abf = self.Atilde + self.Btilde@ F        
        fftilde = self.Btilde @ np.hstack(self.Mibar) @ Fbar        
        for i in range(max_iter):         
            eye_jitter = np.eye(len(self.w_covs))
            jitter = 1e-10
            next_cov = eye_jitter*jitter+abf @ roll_cov @ abf.T + fftilde @ steady_cov @ fftilde.T  +self.w_covs  
            # next_cov = abf @ roll_cov @ abf.T +self.w_covs
            # cov_diff = np.linalg.norm((next_cov-roll_cov), 'fro')
            cov_diff = compute_mahalanobix_dist(next_cov, roll_cov)            
            if cov_diff < 1e-2:
                break 
            roll_cov = next_cov.copy()
          
            
        return roll_cov
        
    def compute_stage_cost(self, P, F, SigmaXbar):
        '''
         stage_cost =    Sum_{ij} Trace{FMi(BPB)MjF}SigmaErrorij
                      + Trace(P Sigma_w)
        '''
        stage_cost = 0        
        for i in range(self.N):
            for j in range(self.N):
                agent_tmp = F.T @ self.Mibar[i].T @ (self.R_tilde+ self.Btilde.T @ P @ self.Btilde ) @ self.Mibar[j] @ F @  self.sigmas(i,j)
                stage_cost += np.trace(agent_tmp)
        stage_cost += np.trace(P @ self.w_covs)
        return stage_cost
    
    def set_MAS(self,info):
        self.N = info["N"]
        self.v_stds = info["v_stds"]
        
   
   