import numpy as np
import math
from utils import *
class Agent:
    # Constructor method (optional)
    def __init__(self, args):
        # Initialize instance variables here
        self.ctrl_type = args['ctrl_type']
        # self.ctrl_type = CtrlTypes.EstFeedback
        self.id = args['id'] # id of agents
        self.Ts = args['Ts']        
        self.n = args['n'] ## state dim         
        self.N = args['N']
        self.p = args['p'] ## input dim
        if self.n == 2:
            self.A = np.array([
                [1, 0],
                [0, 1]
            ])
            self.B = np.array([
                [self.Ts, 0],    
                [0,self.Ts]
            ])
        elif self.n ==4 :

            self.A = np.array([
                [1, self.Ts, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, self.Ts],
                [0, 0, 0, 1]
            ])
            self.B = np.array([
                [self.Ts**2 / 2, 0],
                [self.Ts, 0],
                [0, self.Ts**2 / 2],
                [0, self.Ts]
            ])
        else:
            assert 1==2
            
        
        self.x = np.zeros([self.n,1])
        self.xhat = np.zeros([self.n,1])
        self.xcov = np.eye(self.n*self.N)
        
        # input 
        self.u = np.zeros([self.p,1])
        
        self.N = args['N'] 
                
        self.w_std = args['w_std']# disturbance std
        self.w_cov = np.diagflat(np.kron( self.w_std**2,np.ones([self.n, 1])))        
        self.v_std = args['v_std']# disturbance std
        self.v_cov = np.diagflat(np.kron( self.v_std**2,np.ones([self.n, 1])))        
        self.L = args['L'] # laplacian matrix
        self.c = args['c'] # adgency matrix
        
        self.Ci = np.zeros([int(self.n * np.sum(self.c[self.id,:])), self.n * self.N])
        r_count = 0
        for j in np.where(self.c[self.id] == 1)[0]:
            self.Ci[r_count * self.n:(r_count + 1) * self.n, j * self.n:(j + 1) * self.n] = np.eye(self.n)
            r_count += 1
        
        self.x_mem = []
        self.xhat_mem = []        
        # self.est_gain = np.ones([self.Ci.shape[1], self.Ci.shape[0]])
        self.est_gain = self.Ci.transpose() 
            
        self.Hi = self.Ci ## trying to follow the paper notation
        
        self.z = np.zeros([self.Ci.shape[0],1])
        
        self.ci = np.zeros([self.n*self.N,self.n*self.N]) 
        self.ci[self.n * self.id:self.n * (self.id+1), self.n * self.id:self.n * (self.id+1)] = np.eye(self.n)
        self.Mi = np.zeros([self.p,self.N*self.p])
        self.Mi[:,self.p * self.id:self.p * (self.id+1)] = np.eye(self.p)
        self.Mibar = np.zeros([self.N*self.p,self.N*self.p])
        self.Mibar[self.p * self.id:self.p * (self.id+1), self.p * self.id:self.p * (self.id+1)] = np.eye(self.p)
        
        self.F = None # 
        self.offset = np.zeros([self.Ci.shape[0],1])

    def set_offset(self,offset):
        self.offset =offset.copy()
        
    def set_MAS_info(self, Atilde, Btilde,w_covs, v_covs ):
        self.Atilde = Atilde 
        self.Btilde = Btilde 
        self.w_covs = w_covs 
        self.v_covs = v_covs
    
    def set_xhat(self,xhat):
        self.xhat = xhat.copy() 
        
    def est_step(self):                        
        # x = Abar * x + B F xhat 
        input_tmp = np.dot(self.F, (self.xhat-self.offset)).copy()
        xhat_predict = (np.dot(self.Atilde, self.xhat) + np.dot(self.Btilde,input_tmp)).copy()
        # updated_xhat = xhat_predict+np.dot(self.est_gain, np.dot(self.Hi, (self.z - xhat_predict)))
        updated_xhat = xhat_predict+self.est_gain@ self.Hi@ (self.z - xhat_predict)
        self.xhat = updated_xhat.copy()
        self.xhat_mem.append(updated_xhat.copy())        
        return
    
    

    def set_est_gain(self,est_gain):
        self.est_gain = est_gain
        
    def set_gain(self,F):
        self.F = F
        
    def set_measurement(self,x_all):        
        noise_scale = np.diag(np.sqrt(self.v_cov)).reshape(-1, 1)
        # random normal takes std as scale, but v_cov is in variance
        noise = np.random.normal(loc=0, scale=noise_scale)
        self.z = x_all+noise
<<<<<<< HEAD
    def get_measurement(self):
        return (self.Hi @ self.z).copy()
=======
    
    def get_measurement(self):
        return (self.Hi @ self.z).copy()

>>>>>>> b68887d2504c2304def6f8bd2d4da7f4a83be4e7
    def get_x(self):
        return self.x.copy()
        
    def get_input(self):
        return self.u.copy()
    
    def set_input(self,input):
        self.u = input
    
    def set_x(self,state):
        self.x = state
        
    def step(self, u = None):        
        scale = np.diag(np.sqrt(self.w_cov)).reshape(1, self.n)
        # np.random.normal takes std, while w_cov is variance 
        disturbances = np.random.normal(loc=0, scale=scale, size=(1, self.n))        
        
        if self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback or self.ctrl_type == CtrlTypes.COMLQG:
            input = np.dot(np.dot(self.Mi,self.F),self.xhat-self.offset).copy()
        else:
            input = np.dot(np.dot(self.Mi,self.F),self.z-self.offset)
        if u is not None:               
            input = u
        
        self.u = input.copy()
        new_x = np.dot(self.A,self.x) + np.dot(self.B, self.u)+ disturbances.reshape(self.n,1)
        self.x = new_x.copy()
        self.x_mem.append(self.x.copy())
        
        return
        
    def get_traj(self):
        traj = None
        if len(self.x_mem) > 0:
            traj = np.array(self.x_mem)
        return traj.copy()

    def get_est_traj(self):
        est_traj = None
        if len(self.xhat_mem) > 0:
            est_traj = np.array(self.xhat_mem)
        return est_traj.copy()
        