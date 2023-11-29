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

       
    def draw_shaded_stage_cost(self):
        # stage_costs_lqr = [result['stage_cost'].reshape(-1) for result in self.lqg_results]
        # stage_costs_sub = [result['stage_cost'].reshape(-1) for result in self.sub_results]
        # stage_costs_opt = [result['stage_cost'].reshape(-1) for result in self.opt_results]
        # stage_costs_comlqg = [result['stage_cost'].reshape(-1) for result in self.comlqg_results]



        import matplotlib.pyplot as plt

        # Assuming A, B, C, D are defined as lists of lists with 1000 numerical values each
        # Example: A = [[1, 2, 3, ..., 1000], [...], [...]]

        # Extract the first subitem from each list
        first_A = self.lqg_results[0]['stage_cost']
        first_B = self.sub_results[0]['stage_cost']
        first_C = self.opt_results[0]['stage_cost']
        first_D = self.comlqg_results[0]['stage_cost']

        time_step = first_A.shape[1]
        sample_num = first_A.shape[0]
        # Function to calculate mean and 2-sigma rangedef mean_and_2sigma(data):
        def mean_and_2sigma(data):
            means = np.mean(data, axis=0)
            sigma = np.std(data, axis=0)
            lower_bound = means - 2.0* sigma
            upper_bound = means + 2.0* sigma
            return means, lower_bound, upper_bound

        # Calculate means and 2-sigma ranges for each dataset
        mean_A, lower_A, upper_A = mean_and_2sigma(self.lqg_results[0]['stage_cost'])
        mean_B, lower_B, upper_B = mean_and_2sigma(self.sub_results[0]['stage_cost'])
        mean_C, lower_C, upper_C = mean_and_2sigma(self.opt_results[0]['stage_cost'])
        mean_D, lower_D, upper_D = mean_and_2sigma(self.comlqg_results[0]['stage_cost'])

        # Create an x-axis for plotting
        x = np.arange(time_step)  # Assuming 500 time steps

        # Plotting
        plt.figure(figsize=(12, 6))

        plt.plot(x, mean_A, label='A', color='blue')
        plt.fill_between(x, lower_A, upper_A, color='blue', alpha=0.2)

        plt.plot(x, mean_B, label='B', color='red')
        plt.fill_between(x, lower_B, upper_B, color='red', alpha=0.2)

        plt.plot(x, mean_C, label='C', color='green')
        plt.fill_between(x, lower_C, upper_C, color='green', alpha=0.2)

        plt.plot(x, mean_D, label='D', color='purple')
        plt.fill_between(x, lower_D, upper_D, color='purple', alpha=0.2)

        # Labels and Title
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('Line Plot with 2-Sigma Shaded Area')
        plt.legend()

        # Display the plot
        plt.show()


        # tmp_lqg_result = lqg_result[i]
        # tmp_sub_result = sub_result[i]
        # tmp_opt_result = opt_result[i]
        # tmp_comlqg_result = comlqg_result[i]
        # tmp_comlqg_5_result = comlqg_5_result[i]
        # sub_result = sub_result[-1]
        # opt_result = opt_result[-1]


    
if __name__ == "__main__":      
    save_file_name = 'mcmc_experiment_tmp' +str(2) + str('.pkl')


    
    analysis_v1 = Analysis(save_file_name)
    # analysis_v1.draw_shaded_stage_cost()
    analysis_v1.draw_boxplot()

