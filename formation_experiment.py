import numpy as np
from mas_ws.utils import *
from mas_ws.synthetis import ControlEstimationSynthesis
import threading
from mas_ws.eval import MASEval
import time 

class CrazyFormation:
    def __init__(self,swarm,args):
        self.swarm = swarm
        args['is_sim'] = swarm.args.sim
        
        self.timeHelper = swarm.timeHelper
        self.cfs = swarm.allcfs.crazyflies
        args['gain_file_name'] = str(args['ctrl_type']).split('.')[-1]+'_'+str(args['Hz'])
        self.args = args
        self.Hz = args['Hz']
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
        self.distance = 0.5
          
        self.ctrl_type = args['ctrl_type']
        self.gamma = args['gamma']
        self.com_lqg_delta = np.sort(np.linalg.eigvals(self.L))[-1] + 0.01  
        self.consensus_mtx = -1*(1/self.com_lqg_delta)*np.kron(self.L,np.eye(self.N*self.n))                
        
        self.mission_duration = args['mission_duration']
        self.height = args['height']


        self.offset = get_formation_offset_vector(self.N, self.n, dist = self.distance)
        self.synthesis = ControlEstimationSynthesis(args)        
        self.eval = MASEval(args)
        self.agents = self.synthesis.agents
        self.X = np.zeros([self.N*self.n,1])
        self.Xhat = np.zeros([self.N*self.N*self.n,1])
        self.centralized_xhat = np.zeros([self.N*self.n,1])
        self.stage_costs = []
        
 


        self.init_hover()

        self.init_MAS()   
        self.init_operation()
        
        self.land()    
        self.eval_ready() 
        self.eval.eval_init()
    


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

    def land(self):
        for cf in self.cfs:
            cf.land(targetHeight=0.04, duration=2.5)
        self.timeHelper.sleep(3.0)

    def init_hover(self):
        for cf in self.cfs:
            print('cf : {} take off', cf.id)            
            cf.takeoff(targetHeight=self.height, duration=2.0)
        # self.cfs.takeoff(targetHeight=2.0, duration=1.0)
    
        self.timeHelper.sleep(3.0)
        # timeHelper.sleep(5.0)
        # cf.cmdVelocityWorld(np.array([0.1, 0.0, 0.0]), 0.0)
        # timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
        # timeHelper.sleep(2.0)    
        # timeHelper.sleep(2.0)
        # # cf.takeoff(1.0, 2.5)    
        
        # timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
        # cf.land(targetHeight=0.04, duration=2.5)
        # timeHelper.sleep(TAKEOFF_DURATION)


    def init_MAS(self):

        if self.ctrl_type == CtrlTypes.LQROutputFeedback:
            gain = self.synthesis.lqr_gain
        elif self.ctrl_type == CtrlTypes.SubOutpFeedback:
            gain = self.synthesis.sub_gain
        elif self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
            gain = self.synthesis.opt_gain
        elif self.ctrl_type == CtrlTypes.COMLQG:
            gain = self.synthesis.lqr_gain
        
        for i in range(self.N):
            init_pose_tmp = self.offset[i*2:(i+1)*2].copy()*2
            self.agents[i].set_x(init_pose_tmp)
            self.agents[i].set_gain(gain)
            if self.ctrl_type == CtrlTypes.CtrlEstFeedback or self.ctrl_type == CtrlTypes.LQGFeedback:
                self.agents[i].set_est_gain(self.synthesis.est_gains[i])
            elif self.ctrl_type == CtrlTypes.COMLQG:
                self.agents[i].set_est_gain(self.synthesis.kalman_gain[i])
            self.agents[i].set_offset(self.offset)

            self.cfs[i].goTo(np.append(init_pose_tmp,np.array([self.height])), 0, 2.0)
            
        self.timeHelper.sleep(2.5)
        
        self.get_states()        
        for i in range(self.N):
            self.agents[i].set_MAS_info(self.synthesis.Atilde,self.synthesis.Btilde, self.synthesis.w_covs, self.synthesis.v_covs)
            self.agents[i].set_xhat(self.X)
        
        self.centralized_xhat = self.X
        
       

    def get_states(self):
        gt_state_vector = []
        for i in range(self.N):
            gt_state = self.cfs[i].position() 
            self.agents[i].set_x(gt_state[:-1])
            gt_state_vector.append(gt_state[:-1])
        self.X =np.expand_dims(np.hstack(gt_state_vector).copy(),axis=1)

    def get_inputs(self):
        input_vector = []
        for i in range(self.N):
            input_tmp = self.agents[i].get_input()
            input_vector.append(input_tmp)
        self.U = np.vstack(input_vector)  

    def compute_cost(self,x,u):
        return np.dot(np.dot(np.transpose(x-self.offset),self.synthesis.Q_tilde),x-self.offset) + np.dot(np.dot(np.transpose(u),self.synthesis.R_tilde),u)
        
    
    def agent_process(self,agent):
        agent.record_state()
        agent.set_measurement(self.X)

    def init_operation(self):
        time_in_sec = 0
        self.init_time =  self.timeHelper.time()
        loop_count = 0

        while time_in_sec < self.mission_duration:
            loop_count +=1
            loop_init_time = time.time()
           
            self.get_states()
            self.get_inputs()            
            stage_cost = self.compute_cost(self.X, self.U)  
            self.eval.add_stage_cost(stage_cost)
            threads = []
            for i in range(self.N):                
                thread = threading.Thread(target=self.agent_process, args=(self.agents[i],))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

            ############## Estimation ################################################
            if self.ctrl_type == CtrlTypes.COMLQG:                
                roll_xhat = np.vstack([agent.xhat.copy() for agent in self.agents])
                for i in range(self.gamma):
                    tmp = self.consensus_mtx @ roll_xhat 
                    roll_xhat = roll_xhat + tmp.copy()
                for i in range(self.N):
                    consensus_est = roll_xhat[i*self.N*self.n:(i+1)*self.N*self.n]
                    self.agents[i].set_xhat(consensus_est.copy())
            
            elif self.ctrl_type == CtrlTypes.LQGFeedback:
                zis = []
                for agent in self.agents:
                    zis.append(agent.get_measurement())
                entire_z = np.vstack(zis)                            
                residual = (self.synthesis.centralized_kalman@(entire_z - self.synthesis.entireCmtx @ self.centralized_xhat)).copy()
                self.centralized_xhat = self.centralized_xhat + residual
                for agent in self.agents:
                    agent.set_xhat(self.centralized_xhat.copy())
                    agent.xhat_mem.append(self.centralized_xhat.copy())   

            elif self.ctrl_type == CtrlTypes.CtrlEstFeedback:
                for i in range(self.N):   
                    self.agents[i].est_step()                             
            ############## Estimation END ################################################


            for i in range(self.N):
                self.agents[i].compute_input()
                action_tmp = self.agents[i].get_input()        
                self.cfs[i].cmdVelocityWorld(np.append(action_tmp,np.array([0])), yawRate=0)
                ################################### TMP for Zaxi control ###########################
                # tmp_pose = self.cfs[i].position()
                # z = tmp_pose[-1]
                # error = -1*0.35*(z-self.height)
                # self.cfs[i].cmdVelocityWorld(np.append(action_tmp,np.array([0])), yawRate=0)
                ################################### TMP for Zaxi control ###########################

                


            time_in_sec = self.timeHelper.time() - self.init_time
            loop_end_time = time.time()
            if loop_end_time -loop_init_time > 1/self.Hz:
                print('While loop takes too long.. {}', loop_end_time -loop_init_time)
            self.timeHelper.sleepForRate(self.Hz)