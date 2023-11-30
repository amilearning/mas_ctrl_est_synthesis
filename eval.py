import numpy as np
import math
from mas_ws.utils import *
from datetime import datetime

import os
import pickle

class MASEval:
    # Constructor method (optional)
    def __init__(self, args):
        # Initialize instance variables here
        self.args = args

        self.stage_costs = []        
        self.trajs = []
        self.est_trajs = []
        self.xhats = []
        current_time = datetime.now().strftime("%m-%d_%H-%M-%S")                                
        if self.args['is_sim']:
            self.save_file_name = 'sim'+ self.args['gain_file_name'] +'_test#' +str(self.args['test_number']) + '_' + str(current_time)
        else:
            self.save_file_name = self.args['gain_file_name'] +'_test#' +str(self.args['test_number']) + '_' + str(current_time)
    
    
    
    
    def save_data(self):
        
        data = {
                'stage_cost': self.stage_costs,
                'trajs': self.trajs,
                'xhat': self.xhats,
                'est_trajs': self.est_trajs,
                'args': self.args
                }

        data_file_name = self.save_file_name + str('.pkl')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  # Create a 'data' directory in the same folder as your script
        file_path = os.path.join(data_dir, data_file_name)  # Create a 'data' directory in the same folder as your script
        if os.path.exists(file_path):
            file_path = file_path.split('.pkl')[0]+'_copy.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(data,file)

            

    def add_stage_cost(self,cost):
        self.stage_costs.append(cost.copy())
        

    def eval_init(self):
        self.save_data()
        plot_x_y(self.trajs)
        fig_file_name = self.save_file_name + 'xyplot' +str('.png')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'figure')  # Create a 'data' directory in the same folder as your script
        file_path = os.path.join(data_dir, fig_file_name)  # Create a 'data' directory in the same folder as your script
        plt.savefig(file_path)
        plt.show()
        # self.get_error_plot()
        plot_stage_cost(self.stage_costs)
        fig_file_name = self.save_file_name + 'stage_cost' +str('.png')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'figure')  # Create a 'data' directory in the same folder as your script
        file_path = os.path.join(data_dir, fig_file_name)  # Create a 'data' directory in the same folder as your script
        plt.savefig(file_path)       
        plt.show()
        
    def get_results(self):
        results = {}
        results['stage_cost'] = self.stage_costs
        results['trajs'] = self.trajs
        results['xhats'] = self.xhats
        results['est_trajs'] = self.est_trajs
        results['args'] = self.args
        return results
    
    def list_to_np(self):
        self.stage_costs = np.stack(self.stage_costs).squeeze()
        self.trajs = np.stack(self.trajs).squeeze()
        self.est_trajs = np.stack(self.est_trajs).squeeze()
        
    def plot_est_gt(self,id):
        
        est_traj = self.est_trajs[id,:,:]
        reshaped_est_traj = np.zeros(self.trajs.shape)
        n = self.trajs.shape[2]
        
        for i in range(self.trajs.shape[0]):
            reshaped_est_traj[i,:,:] = est_traj[:,i*n:(i+1)*n]
            
        plot_x_y(reshaped_est_traj)
        
    def get_error_plot(self):
        self.list_to_np()     
        traj = np.transpose(self.trajs, (1, 0, 2))
        gt_state = traj.reshape(traj.shape[0],-1)
        
        gt_state = np.repeat(gt_state[np.newaxis, :], self.est_trajs.shape[0], axis=0)
        err = gt_state - self.est_trajs
        pow_err = err
        err_sqrt = np.sqrt(np.sum(pow_err,axis=2))
        err_sqrt = pow_err[:,:,0]
        error_plot(err_sqrt)
        print(np.mean(err_sqrt))
        return
            
        
        