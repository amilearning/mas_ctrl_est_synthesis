import os
import numpy as np
import seaborn as sns
import pandas as pd
import pickle   
import matplotlib.pyplot as plt

    
class Analysis:
    # Constructor method (optional)
    def __init__(self, save_file_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')  # Create a 'data' directory in the same folder as your script
        file_path = os.path.join(data_dir, save_file_name)  # Create a 'data' directory in the same folder as your script
        
        if os.path.exists(file_path) is False:
            assert(1==2,"Pickle file is not found.")
        else:            
            with open(file_path, 'rb') as file:
                self.data = pickle.load(file)
                self.args = self.data['args']
                self.num_agent_list = self.data['num_agent_list'] 
                self.lqg_results = self.data['lqg_results']  
                self.opt_results = self.data['opt_results']  
                self.sub_results = self.data['sub_results']  
                self.comlqg_results = self.data['comlqg_results'] 
                
            
            self.draw_boxplot()
    
    def get_df(self,stage_costs, num_list, algorithm_name):        
        df = pd.DataFrame({
            'NumberofAgent': np.repeat(num_list, len(stage_costs[0])),            
            'Cost': np.concatenate(stage_costs),
            'Algorithm': [algorithm_name] * (len(stage_costs[0])*len(num_list))
        })
        return df 
    
    def draw_boxplot(self):
            # Extract stage costs from dictionaries
        stage_costs_lqr = [result['stage_cost'].reshape(-1) for result in self.lqg_results]
        stage_costs_sub = [result['stage_cost'].reshape(-1) for result in self.sub_results]
        stage_costs_opt = [result['stage_cost'].reshape(-1) for result in self.opt_results]
        stage_costs_comlqg = [result['stage_cost'].reshape(-1) for result in self.comlqg_results]

        
        
        lqr_df = self.get_df(stage_costs_lqr, self.num_agent_list, 'fullyconnected')
        sub_df = self.get_df(stage_costs_sub, self.num_agent_list, 'suboptimal')
        opt_df = self.get_df(stage_costs_opt, self.num_agent_list, 'optimal')
        comlqg_df = self.get_df(stage_costs_comlqg, self.num_agent_list, 'comlqg')
        
        merged_df = pd.concat([lqr_df, sub_df, opt_df,comlqg_df], ignore_index=True)


        
        df = pd.DataFrame({'FullyconnectedLQR': stage_costs_lqr, 'Suboptimal': stage_costs_sub, 'Proposed': stage_costs_opt, 'COMLQG' : stage_costs_comlqg})               
        sns.boxplot(data=merged_df)

        plt.figure(figsize=(10, 8))
        sns.boxplot(x='NumberofAgent', y='Cost', data=merged_df, hue = 'Algorithm')
        plt.title('Boxplot of Costs for Each Algorithm')
        plt.show()

       


    
if __name__ == "__main__":      
    save_file_name = 'mcmc_experiment_' +str(1) + str('.pkl')


    
    analysis_v1 = Analysis(save_file_name)

